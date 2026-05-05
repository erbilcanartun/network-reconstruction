"""
Temporally Multiplexed ESN (TM-ESN) for Rulkov map closed-loop prediction.

OPTIMIZATIONS vs. the original grid-search version:
  1. Search strategy: Optuna TPE sampler (~400 trials) instead of 72k grid
     - Adaptive sampling concentrates near the optimum
     - JournalFileBackend storage allows multiple worker processes to share
       the same study via a single log file
     - Median pruner kills underperforming trials early
  2. Per-trial speed:
     - Multi-ridge ridge sweep: one eigendecomposition of FtF, then each
       ridge is a cheap diagonal solve. Tries 4 ridges in roughly the cost
       of one fit.
     - scipy.sparse @ x (Accelerate-backed) instead of hand-rolled Numba CSR.
       On Apple Silicon, scipy.sparse is faster (Accelerate BLAS beats
       a single-threaded Numba loop). Numba dropped entirely -> no JIT
       warmup, simpler code.
     - Ring buffer in closed_loop -> O(1) bookkeeping per step.
     - Early-abort if predictions blow up after first 20 steps.
     - Search uses a 10k-step training window (vs 20k full); final
       validation reruns top-5 at full T with 3 seeds.
  3. Memory budget for 8 GB RAM:
     - N capped at 1500, L capped at 4 (feature dim <= 7500).
     - The fit forms FtF (P x P) and FtY (P x D), eigendecomposes FtF
       once, then sweeps ridges via diagonal solves. FtF at P=4500 is
       ~160 MB, comfortably fits.
     - Workers default to 1 in-process; for true parallelism, run multiple
       processes via separate terminals (each ~1-2 GB peak).

USAGE:
  Run this file once to start a master worker that creates the study.
  To add more workers, run additional copies in separate terminals; they
  attach to the same journal log via load_if_exists=True.

    python tm_esn_optuna.py             # first worker, creates study
    python tm_esn_optuna.py             # second worker, joins study
    python tm_esn_optuna.py             # third worker, joins study

  Or use the built-in --n-jobs flag for in-process threading workers
  (lighter overhead, but threads share memory).
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


# ==========================================================
# CONFIG
# ==========================================================
DATA_PATH = Path("../../data/chaotic_data/rulkov_map.csv")  # adjust if needed
STUDY_NAME = "tm_esn_rulkov_v2"
JOURNAL_PATH = "./tm_esn_optuna_journal_v2.log"
RESULTS_DIR = Path("./tm_esn_results_v2")
# Cross-process stop flag. Touching this file makes every worker exit
# after its current trial finishes. Workers remove it on startup so old
# flags don't poison new sessions.
STOP_FLAG_PATH = "./tm_esn_stop_flag"

TRAIN_LEN_FULL = 20_000      # used for final retrain
TRAIN_LEN_SEARCH = 10_000    # used during Optuna search (much faster fit)

# Multi-window evaluation: test on 3 disjoint windows and average the score.
# Reduces noise from the chaotic test-window-of-the-month effect.
# Windows are placed AFTER training (which uses [0:TRAIN_LEN_FULL] = [0:20000])
# and sized so all three fit within ~30000 total steps.
TEST_STARTS = [21_000, 24_000, 27_000]
TEST_LEN = 2_000

SEEDS_SEARCH = [42, 7]            # 2 seeds during search for speed
SEEDS_FINAL = [42, 7, 2024]       # 3 seeds for final validation
SHORT_HORIZON = 50
ALPHA_SPIKE = 0.5      # weight for spike-count penalty
ALPHA_ISI = 0.3        # weight for ISI-distribution Wasserstein distance
SPIKE_THRESHOLD = 0.0
EARLY_ABORT_HORIZON = 20
EARLY_ABORT_NRMSE = 2.0  # if NRMSE in first 20 steps > this, skip rest

# Ridge values swept "for free" via single eigh per architecture/seed.
# Wider sweep than before; cost is just a few diagonal solves per ridge.
RIDGE_VALUES = (1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


# ==========================================================
# SCALERS
# ==========================================================
def fit_scaler(X):
    X = np.asarray(X)
    return {"mu": X.mean(axis=0), "sd": X.std(axis=0) + 1e-12}


def transform(X, scaler):
    return (np.asarray(X) - scaler["mu"]) / scaler["sd"]


def inverse_transform(X, scaler):
    return np.asarray(X) * scaler["sd"] + scaler["mu"]


# ==========================================================
# RESERVOIR INIT (sparse, M2-friendly)
# ==========================================================
def init_reservoir(N, D, sr, input_scaling, connectivity, bias_scaling, rng):
    """Build a sparse reservoir matrix with target spectral radius `sr`.

    For larger N we estimate the spectral radius via power iteration on the
    sparse matrix instead of a dense eigendecomposition.
    """
    nnz = int(connectivity * N * N)
    rows = rng.integers(0, N, size=nnz)
    cols = rng.integers(0, N, size=nnz)
    vals = rng.uniform(-0.5, 0.5, size=nnz)
    W = csr_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    W.sum_duplicates()

    # Power iteration for largest |eigenvalue| of a real (non-symmetric) matrix.
    x = rng.standard_normal(N)
    x /= np.linalg.norm(x) + 1e-12
    for _ in range(30):
        y = W @ x
        y = W.T @ y
        nrm = np.linalg.norm(y)
        if nrm < 1e-12:
            break
        x = y / nrm
    sigma_max = np.sqrt(nrm) if nrm > 0 else 1.0
    if sigma_max > 0:
        W = W.multiply(sr / sigma_max).tocsr()

    Win = rng.uniform(-input_scaling, input_scaling, size=(N, D))
    bias = rng.uniform(-bias_scaling, bias_scaling, size=(N,))
    return W, Win, bias


# ==========================================================
# TM-ESN
# ==========================================================
class TMESN:
    """Temporally Multiplexed ESN.

    Readout features at time t:
        f(t) = [x(t), x(t-tau), x(t-2*tau), ..., x(t-L*tau)]

    Trained as ridge regression on these stacked features. The fit accepts
    a list/tuple of ridge values and returns a dict {ridge: Wout} computed
    via a single SVD of F (cheap ridge sweep).
    """

    def __init__(
        self,
        N=1000, D=1, sr=0.95, lr=0.4, input_scaling=1.0,
        connectivity=0.1, bias_scaling=0.5,
        tau=4, L=2, noise_sigma=0.005, seed=42,
    ):
        self.N = int(N)
        self.D = int(D)
        self.sr = float(sr)
        self.lr = float(lr)
        self.input_scaling = float(input_scaling)
        self.connectivity = float(connectivity)
        self.bias_scaling = float(bias_scaling)
        self.tau = int(tau)
        self.L = int(L)
        self.noise_sigma = float(noise_sigma)
        self.seed = int(seed)

        rng = np.random.default_rng(seed)
        self.W, self.Win, self.bias = init_reservoir(
            self.N, self.D, self.sr, self.input_scaling,
            self.connectivity, self.bias_scaling, rng,
        )
        self.rng_noise = np.random.default_rng(seed + 100)
        self.Wout_dict = None  # filled by fit

    # ---------- Reservoir update (vectorized, scipy.sparse) ----------
    def _step(self, x_prev, u):
        # scipy.sparse @ x dispatches to Accelerate on M2 -> fast
        pre = self.W @ x_prev + self.Win @ u + self.bias
        return (1.0 - self.lr) * x_prev + self.lr * np.tanh(pre)

    def collect_states(self, U):
        T = len(U)
        X = np.empty((T, self.N), dtype=np.float64)
        x = np.zeros(self.N, dtype=np.float64)
        for t in range(T):
            x = self._step(x, U[t])
            X[t] = x
        return X

    def stack_features(self, X):
        T, N = X.shape
        P = (self.L + 1) * N
        F = np.zeros((T, P), dtype=np.float64)
        F[:, :N] = X
        for j in range(1, self.L + 1):
            shift = j * self.tau
            if shift < T:
                F[shift:, j * N:(j + 1) * N] = X[: T - shift]
        return F

    # ---------- Training: SVD-based multi-ridge fit ----------
    def fit_from_states(self, states, Y, warmup, ridge_values=RIDGE_VALUES):
        F = self.stack_features(states)
        F_use = F[warmup:]
        Y_use = Y[warmup:]

        # Economy SVD: F_use = U @ diag(s) @ Vt, shapes (T, k), (k,), (k, P).
        # For ridge regression, Wout.T = V @ diag(s/(s^2 + lambda)) @ U.T @ Y.
        # T_use is typically ~19500 and P up to 7500, so k = min(T,P) = P.
        # Memory: U is (T, P) ~ 1.2 GB at P=7500, T=19500. That's heavy.
        # Trick: compute FtF = F.T F (P, P) and FtY = F.T Y (P, D), then
        # eigendecompose FtF = V S V.T with S = sigma^2. This is much smaller.
        FtF = F_use.T @ F_use                       # (P, P)
        FtY = F_use.T @ Y_use                       # (P, D)
        # symmetrize (numerical safety)
        FtF = 0.5 * (FtF + FtF.T)
        eigvals, eigvecs = np.linalg.eigh(FtF)      # ascending order
        # clip tiny negatives from rounding
        eigvals = np.clip(eigvals, 0.0, None)

        VtFtY = eigvecs.T @ FtY                     # (P, D)

        Wout_dict = {}
        for lam in ridge_values:
            # Wout.T = V @ diag(1/(s^2 + lam)) @ V.T @ FtY
            #       = V @ ((1/(eigvals+lam))[:, None] * VtFtY)
            inv_diag = 1.0 / (eigvals + lam)
            tmp = inv_diag[:, None] * VtFtY         # (P, D)
            Wout_T = eigvecs @ tmp                  # (P, D)
            Wout_dict[lam] = Wout_T.T.copy()        # (D, P)

        self.Wout_dict = Wout_dict
        return Wout_dict

    def fit(self, U, Y, warmup=500, ridge_values=RIDGE_VALUES):
        U = np.asarray(U, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if self.noise_sigma > 0:
            U = U + self.noise_sigma * self.rng_noise.standard_normal(U.shape)
        states = self.collect_states(U)
        return self.fit_from_states(states, Y, warmup, ridge_values)

    # ---------- Closed-loop with ring buffer ----------
    def closed_loop(self, U_warmup, n_steps, Wout):
        """Run autonomously for n_steps after syncing on U_warmup.

        Wout: (D, (L+1)*N) readout to use for this rollout.
        """
        sync_states = self.collect_states(U_warmup)
        max_lag = self.L * self.tau
        buf_size = max_lag + 1

        # Ring buffer: most recent at index `head`
        buffer = np.zeros((buf_size, self.N), dtype=np.float64)
        T_w = sync_states.shape[0]
        if T_w >= buf_size:
            buffer[:] = sync_states[-buf_size:]
            head = buf_size - 1
        else:
            buffer[-T_w:] = sync_states
            head = buf_size - 1

        x_curr = sync_states[-1].copy()
        u_t = U_warmup[-1].copy()

        N, L = self.N, self.L
        preds = np.zeros((n_steps, self.D), dtype=np.float64)
        feat = np.empty((L + 1) * N, dtype=np.float64)

        for step in range(n_steps):
            x_curr = self._step(x_curr, u_t)
            head = (head + 1) % buf_size
            buffer[head] = x_curr

            # Build feature: lag j -> buffer index (head - j*tau) mod buf_size
            for j in range(L + 1):
                idx = (head - j * self.tau) % buf_size
                feat[j * N:(j + 1) * N] = buffer[idx]

            y_hat = Wout @ feat
            preds[step] = y_hat
            u_t = y_hat

            # Early abort on blow-up
            if step == EARLY_ABORT_HORIZON and (
                np.any(np.isnan(y_hat)) or np.max(np.abs(y_hat)) > 1e6
            ):
                preds[step + 1:] = np.nan
                break

        return preds


# ==========================================================
# METRICS
# ==========================================================
def count_spikes(signal, threshold=0.0):
    s = np.asarray(signal).ravel()
    if len(s) < 3:
        return 0
    is_peak = (s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]) & (s[1:-1] > threshold)
    return int(np.sum(is_peak))


def spike_indices(signal, threshold=0.0):
    """Return integer indices of detected spikes (local maxima above threshold)."""
    s = np.asarray(signal).ravel()
    if len(s) < 3:
        return np.array([], dtype=int)
    is_peak = (s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]) & (s[1:-1] > threshold)
    # +1 because is_peak is computed on s[1:-1]
    return np.flatnonzero(is_peak) + 1


def isi_array(signal, threshold=0.0):
    """Inter-spike intervals (ISIs) as a 1D array. Empty if <2 spikes."""
    idx = spike_indices(signal, threshold)
    if len(idx) < 2:
        return np.array([], dtype=float)
    return np.diff(idx).astype(float)


def isi_wasserstein_distance(isi_true, isi_pred):
    """1D Wasserstein-1 (earth mover's) distance between two ISI samples,
    normalized by the standard deviation of isi_true so the score is
    dimensionless and comparable across configs.

    For 1D distributions, W1 = mean(|sorted(a) - sorted(b)|) when |a|=|b|;
    when sample sizes differ we resample to the smaller length via quantile
    matching. Cheap and robust.
    """
    if len(isi_true) < 2:
        # Truth has too few spikes to define an ISI distribution: treat as no
        # penalty (the spike-count term will already penalize this case).
        return 0.0
    if len(isi_pred) < 2:
        # Prediction failed to spike: large penalty.
        return 5.0

    n = min(len(isi_true), len(isi_pred))
    # Quantile matching: compare sorted samples at common quantiles.
    q = (np.arange(n) + 0.5) / n
    a = np.quantile(isi_true, q)
    b = np.quantile(isi_pred, q)
    w1 = float(np.mean(np.abs(a - b)))
    scale = float(np.std(isi_true)) + 1e-9
    return w1 / scale


def composite_score(
    y_true, y_pred,
    short_horizon=50,
    alpha_spike=0.5, alpha_isi=0.3,
    threshold=0.0,
):
    """Multi-component score for a single (truth, prediction) pair.

    Components:
      - NRMSE on the first `short_horizon` steps (trajectory accuracy near
        the warmup boundary; this is the strict trajectory metric).
      - |log(spike_count_ratio)|: penalizes wrong spike density on the full
        prediction window.
      - Wasserstein-1 distance between ISI distributions (predicted vs true),
        normalized by std(ISI_true): penalizes wrong burst structure.

    Returns (combined_score, nrmse_short, spike_pen, isi_dist).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        return np.inf, np.inf, np.inf, np.inf

    h = min(short_horizon, len(y_true))
    err = y_true[:h] - y_pred[:h]
    rmse = np.sqrt(np.mean(err ** 2))
    denom = np.std(y_true[:h])
    nrmse_short = rmse / denom if denom > 0 else np.inf

    n_true = count_spikes(y_true, threshold=threshold)
    n_pred = count_spikes(y_pred, threshold=threshold)
    if n_true == 0:
        spike_pen = 0.0 if n_pred == 0 else 5.0
    else:
        ratio = max(n_pred, 1) / n_true
        spike_pen = abs(np.log(ratio))

    isi_true = isi_array(y_true, threshold=threshold)
    isi_pred = isi_array(y_pred, threshold=threshold)
    isi_dist = isi_wasserstein_distance(isi_true, isi_pred)

    combined = nrmse_short + alpha_spike * spike_pen + alpha_isi * isi_dist
    return combined, nrmse_short, spike_pen, isi_dist


# ==========================================================
# DATA (loaded once per process at import time)
# ==========================================================
def _load_data():
    arr = np.loadtxt(DATA_PATH, delimiter=",")
    if arr.ndim > 1:
        arr = arr[:, 0]
    data = arr.reshape(-1, 1)
    X_raw = data[:-1]
    Y_raw = data[1:]
    return X_raw, Y_raw


X_raw, Y_raw = _load_data()
X_train_full_raw = X_raw[:TRAIN_LEN_FULL]
Y_train_full_raw = Y_raw[:TRAIN_LEN_FULL]

# Multiple test windows for noise-robust evaluation.
# Sanity-check that all requested windows fit in the data.
_max_idx = max(s + TEST_LEN for s in TEST_STARTS)
if _max_idx > len(X_raw):
    raise ValueError(
        f"Data has {len(X_raw)} rows but test windows need up to {_max_idx}. "
        f"Either lower TEST_STARTS or generate longer data."
    )

# Scaler is fit on the full training window (more representative statistics)
_scaler = fit_scaler(X_train_full_raw)
X_train_full = transform(X_train_full_raw, _scaler)
Y_train_full = transform(Y_train_full_raw, _scaler)
# Search uses a subset for faster fits
X_train_search = X_train_full[:TRAIN_LEN_SEARCH]
Y_train_search = Y_train_full[:TRAIN_LEN_SEARCH]

# List of (X_test_scaled, Y_test_scaled, X_test_raw, Y_test_raw) per window
TEST_WINDOWS = []
for s in TEST_STARTS:
    X_t_raw = X_raw[s:s + TEST_LEN]
    Y_t_raw = Y_raw[s:s + TEST_LEN]
    TEST_WINDOWS.append((
        transform(X_t_raw, _scaler),
        transform(Y_t_raw, _scaler),
        X_t_raw,
        Y_t_raw,
    ))
# For backward compatibility (the old single-window aliases used elsewhere in
# the file). The first window is treated as the "primary" plotting window.
X_test, Y_test, X_test_raw, Y_test_raw = TEST_WINDOWS[0]


# ==========================================================
# OBJECTIVE
# ==========================================================
def evaluate_arch_one_seed(arch, ridge_values, seed, X_train, Y_train):
    """Train + closed-loop for one architecture, one seed, all ridges,
    averaged across ALL test windows in TEST_WINDOWS.

    Returns dict {ridge: (score, nrmse_short, spike_pen, isi_dist)} where
    each value is the mean across test windows.
    """
    test_warmup = arch["test_warmup"]
    train_warmup = arch["train_warmup"]
    pred_len = TEST_LEN - test_warmup

    fail = (np.inf, np.inf, np.inf, np.inf)
    out = {r: fail for r in ridge_values}

    if test_warmup >= TEST_LEN or train_warmup >= len(X_train):
        return out

    try:
        model = TMESN(
            N=arch["units"],
            D=1,
            sr=arch["sr"],
            lr=arch["lr"],
            input_scaling=arch["input_scaling"],
            connectivity=arch["connectivity"],
            bias_scaling=arch["bias_scaling"],
            tau=arch["tau"],
            L=arch["L"],
            noise_sigma=arch["noise_sigma"],
            seed=seed,
        )

        # Single fit for all ridges via shared eigendecomposition.
        # Training data is the same across windows; only the rollout differs.
        Wout_dict = model.fit(
            X_train, Y_train, warmup=train_warmup, ridge_values=ridge_values,
        )

        for ridge, Wout in Wout_dict.items():
            scores_w = []
            nrmses_w = []
            spikes_w = []
            isis_w = []
            for X_t, Y_t, _, _ in TEST_WINDOWS:
                Y_true = Y_t[test_warmup:test_warmup + pred_len, 0]
                preds = model.closed_loop(
                    X_t[:test_warmup], n_steps=pred_len, Wout=Wout,
                )
                y_pred = preds[:, 0]
                score, nrmse_s, spike_p, isi_d = composite_score(
                    Y_true, y_pred,
                    short_horizon=SHORT_HORIZON,
                    alpha_spike=ALPHA_SPIKE,
                    alpha_isi=ALPHA_ISI,
                    threshold=SPIKE_THRESHOLD,
                )
                scores_w.append(score)
                nrmses_w.append(nrmse_s)
                spikes_w.append(spike_p)
                isis_w.append(isi_d)
            out[ridge] = (
                float(np.mean(scores_w)),
                float(np.mean(nrmses_w)),
                float(np.mean(spikes_w)),
                float(np.mean(isis_w)),
            )
    except Exception:
        pass

    return out


def evaluate_arch(arch, ridge_values, seeds, X_train, Y_train):
    """Average per-seed results into a single dict keyed by ridge."""
    per_seed = [
        evaluate_arch_one_seed(arch, ridge_values, s, X_train, Y_train)
        for s in seeds
    ]
    averaged = {}
    for r in ridge_values:
        scores = [p[r][0] for p in per_seed]
        nrmses = [p[r][1] for p in per_seed]
        spikes = [p[r][2] for p in per_seed]
        isis = [p[r][3] for p in per_seed]
        averaged[r] = (
            float(np.mean(scores)),
            float(np.mean(nrmses)),
            float(np.mean(spikes)),
            float(np.mean(isis)),
        )
    return averaged


def make_objective():
    """Optuna objective. Suggests an architecture; evaluates all ridges via
    a shared eigendecomposition and reports the best ridge's composite
    score as the trial value. Score is averaged across all TEST_WINDOWS.
    """
    def objective(trial: optuna.Trial) -> float:
        arch = {
            # Capacity (wider; memory guard catches infeasible combos)
            "units":         trial.suggest_categorical(
                                 "units", [500, 750, 1000, 1500, 2000]),
            "L":             trial.suggest_categorical(
                                 "L", [1, 2, 3, 4, 6]),

            # Connectivity (sparser options for better generalization)
            "connectivity":  trial.suggest_categorical(
                                 "connectivity", [0.02, 0.05, 0.1, 0.2]),

            # Reservoir dynamics (wider ranges; allows edge-of-chaos sr>1)
            "sr":            trial.suggest_float("sr", 0.70, 1.25, step=0.025),
            "lr":            trial.suggest_float("lr", 0.05, 0.99, step=0.02),
            "input_scaling": trial.suggest_float(
                                 "input_scaling", 0.2, 2.0, step=0.05),
            "bias_scaling":  trial.suggest_float(
                                 "bias_scaling", 0.0, 1.0, step=0.05),

            # Multiplexing lags (added non-power-of-2 values for resonance
            # with bursting periods that powers of 2 can miss)
            "tau":           trial.suggest_categorical(
                                 "tau", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]),

            # Regularization
            "noise_sigma":   trial.suggest_float(
                                 "noise_sigma", 0.0, 0.02, step=0.001),

            "train_warmup":  500,
            "test_warmup":   500,
        }

        # Memory guard: FtF size = ((L+1)*N)^2 * 8 bytes. Keep under ~1.5 GB
        # so two parallel workers fit in 8 GB RAM.
        feat_dim = (arch["L"] + 1) * arch["units"]
        if feat_dim * feat_dim * 8 > 1.5e9:
            raise optuna.TrialPruned()

        results = evaluate_arch(
            arch, RIDGE_VALUES, seeds=SEEDS_SEARCH,
            X_train=X_train_search, Y_train=Y_train_search,
        )
        best_ridge, best = min(results.items(), key=lambda kv: kv[1][0])
        score, nrmse_s, spike_p, isi_d = best

        # Log everything for later analysis
        trial.set_user_attr("best_ridge", best_ridge)
        trial.set_user_attr("best_nrmse_short", nrmse_s)
        trial.set_user_attr("best_spike_pen", spike_p)
        trial.set_user_attr("best_isi_dist", isi_d)
        trial.set_user_attr(
            "all_ridge_scores", {str(k): v for k, v in results.items()}
        )

        return score

    return objective


# ==========================================================
# RUN STUDY
# ==========================================================
def _stop_flag_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Optuna calls this after every completed trial. If the stop flag has
    appeared, request the optimize loop to exit. study.stop() works here
    because we are inside the optimize() call, on the worker's own thread.
    """
    if os.path.exists(STOP_FLAG_PATH):
        print(f"\n[PID {os.getpid()}] Stop flag detected; "
              f"exiting after this trial.")
        study.stop()


def run_study(n_trials: int, n_jobs: int, fresh: bool):
    RESULTS_DIR.mkdir(exist_ok=True)
    if fresh and os.path.exists(JOURNAL_PATH):
        os.remove(JOURNAL_PATH)
        print(f"Removed old journal: {JOURNAL_PATH}")

    # Always clear a stale stop flag at startup. Without this, a flag left
    # over from a previous session would immediately stop the new worker.
    if os.path.exists(STOP_FLAG_PATH):
        os.remove(STOP_FLAG_PATH)
        print(f"Cleared stale stop flag: {STOP_FLAG_PATH}")

    storage = JournalStorage(JournalFileBackend(JOURNAL_PATH))
    # Worker-specific seed so multiple parallel processes explore distinct
    # random startup points. After n_startup_trials, TPE consults the full
    # shared trial history, so workers naturally cooperate.
    sampler_seed = (os.getpid() * 31 + int(time.time())) % (2**31 - 1)
    # Settings tuned for the wider v2 search space:
    #   n_startup_trials=80: broad random exploration before TPE engages.
    #     With ~9 hyperparameters and many categoricals, fewer startups
    #     leave large regions unexplored.
    #   multivariate=True + group=True: model parameter interactions
    #     (especially between categoricals like tau and L) instead of
    #     treating each dimension independently.
    #   constant_liar=True: when running in parallel, treat in-flight
    #     trials as if they returned the running mean, so workers don't
    #     all suggest the same point at once.
    #   gamma=0.15: split top 15% as "good" instead of default 25%.
    #     Tighter focus when objective is noisy.
    sampler = TPESampler(
        seed=sampler_seed,
        multivariate=True,
        group=True,
        constant_liar=True,
        n_startup_trials=80,
        gamma=lambda n: min(int(np.ceil(0.15 * n)), 25),
    )
    pruner = MedianPruner(n_startup_trials=80, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    print(f"Study: {STUDY_NAME}  (existing trials: {len(study.trials)})")
    print(f"Worker PID {os.getpid()}, sampler seed {sampler_seed}")
    print(f"Running {n_trials} trials with n_jobs={n_jobs}")
    t0 = time.time()
    study.optimize(
        make_objective(),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
        callbacks=[_stop_flag_callback],
    )
    print(f"\nElapsed: {time.time() - t0:.1f}s")
    return study


def report(study: optuna.Study):
    print("\n" + "=" * 80)
    print("OPTUNA STUDY COMPLETE")
    print(f"Trials run        : {len(study.trials)}")
    print(f"Best score        : {study.best_value:.6f}")
    print("Best parameters   :")
    for k, v in study.best_params.items():
        print(f"  {k:18s} = {v}")
    ua = study.best_trial.user_attrs
    print(f"Best ridge        : {ua.get('best_ridge')}")
    print(f"Best NRMSE(short) : {ua.get('best_nrmse_short', float('nan')):.6f}")
    print(f"Best spike pen    : {ua.get('best_spike_pen', float('nan')):.6f}")
    print(f"Best ISI dist     : {ua.get('best_isi_dist', float('nan')):.6f}")
    print("=" * 80)

    # Top 10
    completed = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    completed.sort(key=lambda t: t.value)
    print("\nTop 10 trials:")
    print(f"{'Rank':>4}  {'units':>5}  {'sr':>5}  {'lr':>5}  {'in_sc':>6}  "
          f"{'bias':>5}  {'tau':>4}  {'L':>3}  {'noise':>6}  "
          f"{'ridge':>9}  {'score':>8}")
    print("-" * 95)
    for rank, t in enumerate(completed[:10], 1):
        p = t.params
        print(
            f"{rank:4d}  {p.get('units', 0):5d}  {p.get('sr', 0):5.2f}  "
            f"{p.get('lr', 0):5.2f}  {p.get('input_scaling', 0):6.2f}  "
            f"{p.get('bias_scaling', 0):5.2f}  {p.get('tau', 0):4d}  "
            f"{p.get('L', 0):3d}  {p.get('noise_sigma', 0):6.3f}  "
            f"{t.user_attrs.get('best_ridge', 0):9.1e}  {t.value:8.4f}"
        )
    return completed


def validate_top_k(study: optuna.Study, k: int = 5):
    """Re-evaluate the top-k Optuna trials at the FULL training length and
    with all SEEDS_FINAL. The search used a shorter T and fewer seeds for
    speed; this step picks the most reliable winner. Score is averaged
    across all TEST_WINDOWS and seeds."""
    completed = [t for t in study.trials
                 if t.value is not None and np.isfinite(t.value)]
    completed.sort(key=lambda t: t.value)
    top = completed[:k]
    if not top:
        print("No completed trials to validate.")
        return None

    print(f"\nValidating top-{k} trials at full T={TRAIN_LEN_FULL}, "
          f"{len(SEEDS_FINAL)} seeds, {len(TEST_WINDOWS)} test windows...")
    results = []
    for rank, t in enumerate(top, 1):
        arch = {**t.params, "train_warmup": 500, "test_warmup": 500}
        scores = evaluate_arch(
            arch, RIDGE_VALUES, seeds=SEEDS_FINAL,
            X_train=X_train_full, Y_train=Y_train_full,
        )
        best_ridge, (score, nrmse_s, spike_p, isi_d) = min(
            scores.items(), key=lambda kv: kv[1][0]
        )
        results.append({
            "trial_number": t.number,
            "search_score": t.value,
            "validation_score": score,
            "nrmse_short": nrmse_s,
            "spike_pen": spike_p,
            "isi_dist": isi_d,
            "ridge": best_ridge,
            "params": t.params,
        })
        print(f"  Trial #{t.number} (search={t.value:.4f}): "
              f"val={score:.4f}, NRMSE={nrmse_s:.4f}, "
              f"spike={spike_p:.4f}, ISI={isi_d:.4f}, "
              f"ridge={best_ridge:.0e}")

    results.sort(key=lambda r: r["validation_score"])
    print(f"\nBest after validation: trial #{results[0]['trial_number']} "
          f"with score {results[0]['validation_score']:.4f}")
    return results


def _plot_specific(val_result: dict):
    """Plot the closed-loop prediction across ALL test windows and save
    each as a separate panel. The validation winner is fit once at full T
    and rolled out separately on each window."""
    p = val_result["params"]
    best_ridge = val_result["ridge"]
    test_warmup = 500
    train_warmup = 500
    pred_len = TEST_LEN - test_warmup

    model = TMESN(
        N=p["units"], D=1, sr=p["sr"], lr=p["lr"],
        input_scaling=p["input_scaling"],
        connectivity=p["connectivity"],
        bias_scaling=p.get("bias_scaling", 0.5),
        tau=p["tau"], L=p["L"],
        noise_sigma=p["noise_sigma"],
        seed=SEEDS_FINAL[0],
    )
    Wout_dict = model.fit(
        X_train_full, Y_train_full,
        warmup=train_warmup, ridge_values=(best_ridge,),
    )

    n_win = len(TEST_WINDOWS)
    fig, axes = plt.subplots(n_win, 1, figsize=(14, 4 * n_win), sharex=True)
    if n_win == 1:
        axes = [axes]

    nrmse_short_list = []
    nrmse_full_list = []
    spike_ratio_list = []

    for w_idx, ((X_t, Y_t, X_t_raw, Y_t_raw), ax) in enumerate(
        zip(TEST_WINDOWS, axes)
    ):
        preds_scaled = model.closed_loop(
            X_t[:test_warmup], n_steps=pred_len, Wout=Wout_dict[best_ridge],
        )
        Y_pred = inverse_transform(preds_scaled, _scaler).ravel()
        Y_true = inverse_transform(
            Y_t[test_warmup:test_warmup + pred_len], _scaler,
        ).ravel()

        rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2))
        nrmse_full = rmse / (np.std(Y_true) + 1e-12)
        short = min(SHORT_HORIZON, len(Y_true))
        nrmse_short = (
            np.sqrt(np.mean((Y_true[:short] - Y_pred[:short]) ** 2))
            / (np.std(Y_true[:short]) + 1e-12)
        )
        n_true_spk = count_spikes(Y_true, SPIKE_THRESHOLD)
        n_pred_spk = count_spikes(Y_pred, SPIKE_THRESHOLD)
        nrmse_short_list.append(nrmse_short)
        nrmse_full_list.append(nrmse_full)
        spike_ratio_list.append(
            n_pred_spk / max(n_true_spk, 1)
        )

        ax.plot(np.arange(TEST_LEN), Y_t_raw[:, 0], color="green",
                label="Ground truth", linewidth=1.0)
        ax.axvline(test_warmup, linestyle="--", c="k", linewidth=0.8,
                   label="Warmup end")
        ax.plot(np.arange(test_warmup, test_warmup + pred_len), Y_pred,
                linestyle="--", c="red", linewidth=1.0,
                label="TM-ESN closed-loop")
        ax.set_title(
            f"Window {w_idx + 1} (start={TEST_STARTS[w_idx]}) | "
            f"NRMSE(50)={nrmse_short:.3f} | spikes {n_pred_spk}/{n_true_spk}"
        )
        ax.set_ylabel("Signal")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Step within test window")

    fig.suptitle(
        f"TM-ESN (validated) | val_score={val_result['validation_score']:.3f} | "
        f"N={p['units']}, tau={p['tau']}, L={p['L']}, "
        f"sr={p['sr']:.2f}, lr={p['lr']:.2f}, noise={p['noise_sigma']:.3f}",
        y=1.0,
    )
    plt.tight_layout()

    out_path = RESULTS_DIR / "tm_esn_best.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved figure: {out_path}")
    print(f"Per-window NRMSE(50): "
          f"{[f'{x:.3f}' for x in nrmse_short_list]}")
    print(f"Per-window NRMSE(full): "
          f"{[f'{x:.3f}' for x in nrmse_full_list]}")
    print(f"Per-window spike ratio: "
          f"{[f'{x:.3f}' for x in spike_ratio_list]}")
    plt.show()


# ==========================================================
# CLI / NOTEBOOK ENTRY POINTS
# ==========================================================
def _load_existing_study():
    """Load the on-disk study without modifying it."""
    storage = JournalStorage(JournalFileBackend(JOURNAL_PATH))
    return optuna.load_study(study_name=STUDY_NAME, storage=storage)


def status(plot_convergence: bool = True):
    """Quick snapshot of the current study state. Safe to call repeatedly
    while workers are running. Cheap (just reads the journal)."""
    study = _load_existing_study()
    completed = [t for t in study.trials
                 if t.value is not None and np.isfinite(t.value)]
    running = [t for t in study.trials if t.state.name == "RUNNING"]
    print(f"Trials in journal : {len(study.trials)}")
    print(f"  completed       : {len(completed)}")
    print(f"  running         : {len(running)}")
    if completed:
        best = min(completed, key=lambda t: t.value)
        print(f"Best score so far : {best.value:.4f}  (trial #{best.number})")
        print("Best params       :")
        for k, v in best.params.items():
            print(f"  {k:14s} = {v}")
    if plot_convergence and len(completed) >= 2:
        values = [t.value for t in sorted(completed, key=lambda t: t.number)]
        running_best = np.minimum.accumulate(values)
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(running_best, color="steelblue")
        ax.set_xlabel("Completed trial #")
        ax.set_ylabel("Best score so far")
        ax.set_title(f"Convergence ({len(completed)} completed trials)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    return study


def request_stop():
    """Ask all workers to stop AFTER their current trial finishes.

    Implementation: writes a small flag file at STOP_FLAG_PATH. Each
    worker has registered a callback that runs after every trial; the
    callback checks for this file and calls study.stop() if it exists.
    Currently-running trials complete normally and their results are
    saved. Workers then exit on their own and clean up the flag file.

    Use this instead of Ctrl+C when you want to keep every completed
    trial. To cancel a stop request before any worker has noticed it,
    delete STOP_FLAG_PATH manually.
    """
    Path(STOP_FLAG_PATH).touch()
    print(f"Stop flag written: {STOP_FLAG_PATH}")
    print("Workers will exit after their current trial completes.")
    print("Wait until each terminal prints 'Elapsed: ...s' before calling")
    print("final_report() to get the validated best model.")


def cancel_stop_request():
    """Delete the stop flag file, in case you changed your mind before any
    worker noticed it. Has no effect on workers that already saw the flag
    and are exiting."""
    if os.path.exists(STOP_FLAG_PATH):
        os.remove(STOP_FLAG_PATH)
        print(f"Removed stop flag: {STOP_FLAG_PATH}")
    else:
        print("No stop flag is currently set.")


def final_report(validate_k: int = 5):
    """Print the report, validate the top-k configs at full T with all seeds,
    plot the validation winner and save to PNG. Use this AFTER all workers
    have stopped (either via Ctrl+C, request_stop(), or natural completion).

    Also removes the stop flag file if it was left behind by request_stop(),
    so subsequent worker runs aren't immediately interrupted.
    """
    if os.path.exists(STOP_FLAG_PATH):
        os.remove(STOP_FLAG_PATH)
    study = _load_existing_study()
    report(study)
    if len(study.trials) > 0 and study.best_value is not None:
        val_results = validate_top_k(study, k=validate_k)
        if val_results:
            _plot_specific(val_results[0])
    return study


def _running_in_jupyter() -> bool:
    """True if we're inside an IPython kernel (Jupyter / VS Code notebook).

    argparse does not play well with kernel-injected arguments such as
    `-f /path/to/kernel.json`, so we detect that case and skip argparse.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return "IPKernelApp" in ip.config or "ipykernel" in str(type(ip)).lower()
    except Exception:
        return False


def run(n_trials: int = 400, n_jobs: int = 1, fresh: bool = False,
        report_only: bool = False, validate_k: int = 5):
    """Programmatic entry point (use this from a notebook).

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials this worker will contribute.
    n_jobs : int
        In-process Optuna threads. Keep at 1 on the M2 8GB; for multi-core
        parallelism, run multiple processes (separate terminals or
        subprocess.Popen) instead.
    fresh : bool
        If True, delete the existing journal file and start over.
    report_only : bool
        If True, skip optimization and just summarize the existing study.
    validate_k : int
        After search, re-evaluate the top-k configs at full T and full
        seeds, and plot the validation winner.
    """
    if report_only:
        storage = JournalStorage(JournalFileBackend(JOURNAL_PATH))
        study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    else:
        study = run_study(n_trials, n_jobs, fresh)

    report(study)
    if len(study.trials) > 0 and study.best_value is not None:
        val_results = validate_top_k(study, k=validate_k)
        if val_results:
            _plot_specific(val_results[0])
    return study


def main():
    if _running_in_jupyter():
        # In a notebook: argparse would fail on kernel args. Use defaults
        # and print a hint so the user knows how to customize.
        print("Detected Jupyter kernel; using default arguments.")
        print("To customize from a notebook, call:")
        print("    import tm_esn_optuna")
        print("    tm_esn_optuna.run(n_trials=200, fresh=True)")
        return run()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=400,
                        help="Number of Optuna trials this worker contributes")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Threads per worker (Optuna in-process). "
                             "On M2 with 8GB, use 1 here and run multiple "
                             "processes via separate terminals for true "
                             "parallelism.")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing journal and start over")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip optimization; just report current study")
    args = parser.parse_args()

    return run(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        fresh=args.fresh,
        report_only=args.report_only,
    )


if __name__ == "__main__":
    main()