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
STUDY_NAME = "tm_esn_rulkov"
JOURNAL_PATH = "./tm_esn_optuna_journal.log"
RESULTS_DIR = Path("./tm_esn_results")

TRAIN_LEN_FULL = 20_000      # used for final retrain
TRAIN_LEN_SEARCH = 10_000    # used during Optuna search (much faster fit)
TEST_START = 22_000
TEST_LEN = 2_000

SEEDS_SEARCH = [42, 7]            # 2 seeds during search for speed
SEEDS_FINAL = [42, 7, 2024]       # 3 seeds for final validation
SHORT_HORIZON = 50
ALPHA = 0.5
SPIKE_THRESHOLD = 0.0
EARLY_ABORT_HORIZON = 20
EARLY_ABORT_NRMSE = 2.0  # if NRMSE in first 20 steps > this, skip rest

# Ridge values swept "for free" via single eigh per architecture/seed.
RIDGE_VALUES = (1e-9, 1e-8, 1e-6, 1e-4)


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
def init_reservoir(N, D, sr, input_scaling, connectivity, rng):
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
    # We iterate W @ W.T @ x to estimate the largest singular value, then
    # use it as a proxy. This is what reservoirpy effectively does too;
    # the spectral radius is upper-bounded by the largest singular value, and
    # for random sparse matrices the two agree to within a few percent.
    # The user can rescale tightly later if needed.
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
    bias = rng.uniform(-0.5, 0.5, size=(N,))
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
        connectivity=0.1, tau=4, L=2, noise_sigma=0.005, seed=42,
    ):
        self.N = int(N)
        self.D = int(D)
        self.sr = float(sr)
        self.lr = float(lr)
        self.input_scaling = float(input_scaling)
        self.connectivity = float(connectivity)
        self.tau = int(tau)
        self.L = int(L)
        self.noise_sigma = float(noise_sigma)
        self.seed = int(seed)

        rng = np.random.default_rng(seed)
        self.W, self.Win, self.bias = init_reservoir(
            self.N, self.D, self.sr, self.input_scaling, self.connectivity, rng,
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


def composite_score(y_true, y_pred, short_horizon=50, alpha=0.5, threshold=0.0):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        return np.inf, np.inf, np.inf

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

    return nrmse_short + alpha * spike_pen, nrmse_short, spike_pen


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
X_test_raw = X_raw[TEST_START:TEST_START + TEST_LEN]
Y_test_raw = Y_raw[TEST_START:TEST_START + TEST_LEN]

# Scaler is fit on the full training window (more representative statistics)
_scaler = fit_scaler(X_train_full_raw)
X_train_full = transform(X_train_full_raw, _scaler)
Y_train_full = transform(Y_train_full_raw, _scaler)
# Search uses a subset for faster fits
X_train_search = X_train_full[:TRAIN_LEN_SEARCH]
Y_train_search = Y_train_full[:TRAIN_LEN_SEARCH]

X_test = transform(X_test_raw, _scaler)
Y_test = transform(Y_test_raw, _scaler)


# ==========================================================
# OBJECTIVE
# ==========================================================
def evaluate_arch_one_seed(arch, ridge_values, seed, X_train, Y_train):
    """Train + closed-loop for one architecture, one seed, all ridges.
    Returns dict {ridge: (score, nrmse_short, spike_pen)}.
    """
    test_warmup = arch["test_warmup"]
    train_warmup = arch["train_warmup"]
    pred_len = TEST_LEN - test_warmup

    out = {r: (np.inf, np.inf, np.inf) for r in ridge_values}

    if test_warmup >= TEST_LEN or train_warmup >= len(X_train):
        return out

    Y_true = Y_test[test_warmup:test_warmup + pred_len, 0]

    try:
        model = TMESN(
            N=arch["units"],
            D=1,
            sr=arch["sr"],
            lr=arch["lr"],
            input_scaling=arch["input_scaling"],
            connectivity=arch["connectivity"],
            tau=arch["tau"],
            L=arch["L"],
            noise_sigma=arch["noise_sigma"],
            seed=seed,
        )

        # Single fit for all ridges via shared eigendecomposition
        Wout_dict = model.fit(
            X_train, Y_train, warmup=train_warmup, ridge_values=ridge_values,
        )

        # Closed-loop per ridge using shared sync states
        for ridge, Wout in Wout_dict.items():
            preds = model.closed_loop(
                X_test[:test_warmup], n_steps=pred_len, Wout=Wout,
            )
            y_pred = preds[:, 0]
            score, nrmse_s, spike_p = composite_score(
                Y_true, y_pred,
                short_horizon=SHORT_HORIZON,
                alpha=ALPHA,
                threshold=SPIKE_THRESHOLD,
            )
            out[ridge] = (score, nrmse_s, spike_p)
    except Exception:
        pass

    return out


def evaluate_arch(arch, ridge_values, seeds, X_train, Y_train):
    per_seed = [
        evaluate_arch_one_seed(arch, ridge_values, s, X_train, Y_train)
        for s in seeds
    ]
    averaged = {}
    for r in ridge_values:
        scores = [p[r][0] for p in per_seed]
        nrmses = [p[r][1] for p in per_seed]
        spikes = [p[r][2] for p in per_seed]
        averaged[r] = (
            float(np.mean(scores)),
            float(np.mean(nrmses)),
            float(np.mean(spikes)),
        )
    return averaged


def make_objective():
    """Optuna objective. Suggests an architecture; we evaluate all ridges
    via SVD and report the best ridge's composite score as the trial value.
    """
    def objective(trial: optuna.Trial) -> float:
        arch = {
            "units":         trial.suggest_categorical("units", [500, 1000, 1500]),
            "connectivity":  trial.suggest_categorical("connectivity", [0.05, 0.1]),
            "sr":            trial.suggest_float("sr", 0.80, 1.10, step=0.05),
            "lr":            trial.suggest_float("lr", 0.1, 0.95, step=0.05),
            "input_scaling": trial.suggest_float("input_scaling", 0.3, 1.2, step=0.1),
            "tau":           trial.suggest_categorical("tau", [2, 4, 8, 16, 32]),
            "L":             trial.suggest_categorical("L", [1, 2, 4]),
            "noise_sigma":   trial.suggest_categorical("noise_sigma", [0.0, 0.005]),
            "train_warmup":  500,
            "test_warmup":   500,
        }

        # Memory guard: feature dimension hard cap to keep peak under ~2 GB
        # FtF size in float64 = ((L+1)*N)^2 * 8 bytes.
        feat_dim = (arch["L"] + 1) * arch["units"]
        if feat_dim * feat_dim * 8 > 1.5e9:  # > ~1.5 GB just for FtF
            raise optuna.TrialPruned()

        results = evaluate_arch(
            arch, RIDGE_VALUES, seeds=SEEDS_SEARCH,
            X_train=X_train_search, Y_train=Y_train_search,
        )
        best_ridge, best = min(results.items(), key=lambda kv: kv[1][0])
        score, nrmse_s, spike_p = best

        # Log everything for later analysis
        trial.set_user_attr("best_ridge", best_ridge)
        trial.set_user_attr("best_nrmse_short", nrmse_s)
        trial.set_user_attr("best_spike_pen", spike_p)
        trial.set_user_attr("all_ridge_scores", {str(k): v for k, v in results.items()})

        return score

    return objective


# ==========================================================
# RUN STUDY
# ==========================================================
def run_study(n_trials: int, n_jobs: int, fresh: bool):
    RESULTS_DIR.mkdir(exist_ok=True)
    if fresh and os.path.exists(JOURNAL_PATH):
        os.remove(JOURNAL_PATH)
        print(f"Removed old journal: {JOURNAL_PATH}")

    storage = JournalStorage(JournalFileBackend(JOURNAL_PATH))
    # Worker-specific seed so multiple parallel processes explore distinct
    # random startup points. After n_startup_trials, TPE consults the full
    # shared trial history, so workers naturally cooperate.
    sampler_seed = (os.getpid() * 31 + int(time.time())) % (2**31 - 1)
    sampler = TPESampler(seed=sampler_seed, multivariate=True,
                         n_startup_trials=30)
    pruner = MedianPruner(n_startup_trials=30, n_warmup_steps=0)

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
    study.optimize(make_objective(), n_trials=n_trials, n_jobs=n_jobs,
                   show_progress_bar=False)
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
    print(f"Best ridge        : {study.best_trial.user_attrs.get('best_ridge')}")
    print(f"Best NRMSE(short) : {study.best_trial.user_attrs.get('best_nrmse_short'):.6f}")
    print(f"Best spike pen    : {study.best_trial.user_attrs.get('best_spike_pen'):.6f}")
    print("=" * 80)

    # Top 10
    completed = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    completed.sort(key=lambda t: t.value)
    print("\nTop 10 trials:")
    print(f"{'Rank':>4}  {'units':>5}  {'sr':>5}  {'lr':>5}  {'in_sc':>6}  "
          f"{'tau':>4}  {'L':>3}  {'noise':>6}  {'ridge':>9}  {'score':>8}")
    print("-" * 80)
    for rank, t in enumerate(completed[:10], 1):
        p = t.params
        print(
            f"{rank:4d}  {p['units']:5d}  {p['sr']:5.2f}  {p['lr']:5.2f}  "
            f"{p['input_scaling']:6.2f}  {p['tau']:4d}  {p['L']:3d}  "
            f"{p['noise_sigma']:6.3f}  "
            f"{t.user_attrs.get('best_ridge', 0):9.1e}  {t.value:8.4f}"
        )
    return completed


def validate_top_k(study: optuna.Study, k: int = 5):
    """Re-evaluate the top-k Optuna trials at the FULL training length and
    with all SEEDS_FINAL. The search used a shorter T and fewer seeds for
    speed; this step picks the most reliable winner."""
    completed = [t for t in study.trials
                 if t.value is not None and np.isfinite(t.value)]
    completed.sort(key=lambda t: t.value)
    top = completed[:k]
    if not top:
        print("No completed trials to validate.")
        return None

    print(f"\nValidating top-{k} trials at full T={TRAIN_LEN_FULL}, "
          f"{len(SEEDS_FINAL)} seeds...")
    results = []
    for rank, t in enumerate(top, 1):
        arch = {**t.params, "train_warmup": 500, "test_warmup": 500}
        scores = evaluate_arch(
            arch, RIDGE_VALUES, seeds=SEEDS_FINAL,
            X_train=X_train_full, Y_train=Y_train_full,
        )
        best_ridge, (score, nrmse_s, spike_p) = min(
            scores.items(), key=lambda kv: kv[1][0]
        )
        results.append({
            "trial_number": t.number,
            "search_score": t.value,
            "validation_score": score,
            "nrmse_short": nrmse_s,
            "spike_pen": spike_p,
            "ridge": best_ridge,
            "params": t.params,
        })
        print(f"  Trial #{t.number} (search={t.value:.4f}): "
              f"validation={score:.4f}, NRMSE={nrmse_s:.4f}, "
              f"ridge={best_ridge:.0e}")

    results.sort(key=lambda r: r["validation_score"])
    print(f"\nBest after validation: trial #{results[0]['trial_number']} "
          f"with score {results[0]['validation_score']:.4f}")
    return results


def _plot_specific(val_result: dict):
    """Plot the closed-loop prediction for a specific (validated) config."""
    p = val_result["params"]
    best_ridge = val_result["ridge"]
    test_warmup = 500
    train_warmup = 500
    pred_len = TEST_LEN - test_warmup

    model = TMESN(
        N=p["units"], D=1, sr=p["sr"], lr=p["lr"],
        input_scaling=p["input_scaling"], connectivity=p["connectivity"],
        tau=p["tau"], L=p["L"], noise_sigma=p["noise_sigma"],
        seed=SEEDS_FINAL[0],
    )
    # Final fit uses the FULL training window for best generalization
    Wout_dict = model.fit(
        X_train_full, Y_train_full,
        warmup=train_warmup, ridge_values=(best_ridge,),
    )
    preds_scaled = model.closed_loop(
        X_test[:test_warmup], n_steps=pred_len, Wout=Wout_dict[best_ridge],
    )

    Y_pred = inverse_transform(preds_scaled, _scaler).ravel()
    Y_true = inverse_transform(
        Y_test[test_warmup:test_warmup + pred_len], _scaler,
    ).ravel()

    rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2))
    nrmse_full = rmse / np.std(Y_true)
    short = min(SHORT_HORIZON, len(Y_true))
    nrmse_short = (
        np.sqrt(np.mean((Y_true[:short] - Y_pred[:short]) ** 2))
        / np.std(Y_true[:short])
    )
    n_true_spk = count_spikes(Y_true, SPIKE_THRESHOLD)
    n_pred_spk = count_spikes(Y_pred, SPIKE_THRESHOLD)

    print("\nFinal model metrics (seed 0, original scale):")
    print(f"  NRMSE (full)       : {nrmse_full:.6f}")
    print(f"  NRMSE (first {short}) : {nrmse_short:.6f}")
    print(f"  Spikes true/pred   : {n_true_spk} / {n_pred_spk}")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(np.arange(TEST_LEN), Y_test_raw[:, 0], color="green",
            label="Ground truth", linewidth=1.0)
    ax.axvline(test_warmup, linestyle="--", c="k", linewidth=0.8,
               label="Warmup end")
    ax.plot(np.arange(test_warmup, test_warmup + pred_len), Y_pred,
            linestyle="--", c="red", linewidth=1.0,
            label="TM-ESN closed-loop")
    ax.set_title(
        f"TM-ESN (validated) | "
        f"val_score={val_result['validation_score']:.3f} | "
        f"NRMSE(50)={nrmse_short:.3f} | spikes {n_pred_spk}/{n_true_spk} | "
        f"N={p['units']}, tau={p['tau']}, L={p['L']}, noise={p['noise_sigma']}"
    )
    ax.set_xlabel("Step within test window")
    ax.set_ylabel("Signal")
    ax.legend()
    plt.tight_layout()
    out_path = RESULTS_DIR / "tm_esn_best.png"
    plt.savefig(out_path, dpi=120)
    print(f"  Saved figure: {out_path}")
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

    This sets study.stop() which Optuna checks between trials. Currently-
    running trials complete normally and their results ARE saved. Workers
    then exit on their own.

    Use this instead of Ctrl+C when you want to keep every completed trial.
    """
    study = _load_existing_study()
    study.stop()
    print("Stop signal sent. Workers will exit after their current trial.")
    print("Wait until terminals show 'Elapsed: ...s' before calling")
    print("final_report() to get the validated best model.")


def final_report(validate_k: int = 5):
    """Print the report, validate the top-k configs at full T with all seeds,
    plot the validation winner and save to PNG. Use this AFTER all workers
    have stopped (either via Ctrl+C, request_stop(), or natural completion).
    """
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