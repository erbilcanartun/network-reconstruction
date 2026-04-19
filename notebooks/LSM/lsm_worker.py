"""
LSM Grid Search Worker — importable module for multiprocessing.

Place this file next to your notebook (or anywhere on sys.path).
Each spawned child process imports this module and calls worker_evaluate().
"""

import numpy as np
from brian2 import (
    NeuronGroup, Synapses, SpikeMonitor, Network,
    ms, mV, defaultclock, prefs, start_scope
)
from sklearn.linear_model import Ridge, Lasso


# ==========================================================
# FIXED STRUCTURAL PARAMETERS
# ==========================================================
TAU_SYN_FIX         = 5 * ms
V_REST_FIX          = -65 * mV
V_RESET_FIX         = -65 * mV
V_THRESH_FIX        = -50 * mV
REFRACTORY_FIX      = 2 * ms
CONN_PROB_FIX       = 0.1
INPUT_CONN_PROB_FIX = 0.3
N_INPUT_NEURONS_FIX = 50
SIGMA_ENC_FIX       = 0.2
MU = np.linspace(-1.0, 1.0, N_INPUT_NEURONS_FIX)

TRAIN_LEN  = 4000
TEST_START = 4000
TEST_LEN   = 1000


# ==========================================================
# ENCODER
# ==========================================================
def encode(x_scalar, encoding, input_gain_mV):
    if encoding == "direct":
        arr = np.full(N_INPUT_NEURONS_FIX, float(x_scalar)) * input_gain_mV
    else:  # population
        bumps = np.exp(-((MU - float(x_scalar)) ** 2) / (2 * SIGMA_ENC_FIX ** 2))
        arr = bumps * input_gain_mV
    return arr * mV


# ==========================================================
# BUILD LIQUID
# ==========================================================
def build_liquid(N_liquid, tau_mem, w_exc, w_inh, w_input, rng_seed):
    np.random.seed(rng_seed)

    liquid_ns = {
        'V_rest': V_REST_FIX, 'V_thresh': V_THRESH_FIX,
        'V_reset': V_RESET_FIX, 'tau_mem': tau_mem, 'tau_syn': TAU_SYN_FIX,
    }
    eqs = '''
    dV/dt = ((V_rest - V) + I_syn + I_input) / tau_mem : volt (unless refractory)
    dI_syn/dt = -I_syn / tau_syn : volt
    I_input : volt
    '''
    liquid = NeuronGroup(
        N_liquid, eqs, threshold='V > V_thresh', reset='V = V_reset',
        refractory=REFRACTORY_FIX, method='euler', namespace=liquid_ns,
    )
    liquid.V = V_REST_FIX

    S_exc = Synapses(liquid, liquid, on_pre='I_syn_post += w_exc',
                     namespace={'w_exc': w_exc})
    S_exc.connect(condition='i != j', p=CONN_PROB_FIX)

    S_inh = Synapses(liquid, liquid, on_pre='I_syn_post += w_inh',
                     namespace={'w_inh': w_inh})
    S_inh.connect(condition='i != j', p=CONN_PROB_FIX)

    input_ns = {
        'V_rest': V_REST_FIX, 'V_thresh': V_THRESH_FIX,
        'V_reset': V_RESET_FIX, 'tau_mem': tau_mem,
    }
    input_eqs = '''
    dV/dt = (V_rest - V + I_drive) / tau_mem : volt (unless refractory)
    I_drive : volt
    '''
    input_layer = NeuronGroup(
        N_INPUT_NEURONS_FIX, input_eqs, threshold='V > V_thresh',
        reset='V = V_reset', refractory=REFRACTORY_FIX,
        method='euler', namespace=input_ns,
    )
    input_layer.V = V_REST_FIX

    S_in = Synapses(input_layer, liquid, on_pre='I_syn_post += w_input',
                    namespace={'w_input': w_input})
    S_in.connect(p=INPUT_CONN_PROB_FIX)

    spike_mon = SpikeMonitor(liquid)
    net = Network(liquid, input_layer, S_exc, S_inh, S_in, spike_mon)
    net.store('initial')
    return net, input_layer, spike_mon


# ==========================================================
# COLLECT STATES
# ==========================================================
def collect_states(net, input_layer, spike_mon, N_liquid,
                   step_duration, input_sequence, encoding,
                   input_gain_mV, continue_from=None):
    T = len(input_sequence)
    states = np.zeros((T, N_liquid))
    filt = np.zeros(N_liquid) if continue_from is None else continue_from.copy()
    decay = np.exp(-float(step_duration / TAU_SYN_FIX))

    for t in range(T):
        input_layer.I_drive = encode(input_sequence[t, 0], encoding, input_gain_mV)
        spikes_before = len(spike_mon.t)
        net.run(step_duration)
        new_idx = np.array(spike_mon.i[spikes_before:])
        counts = np.bincount(new_idx, minlength=N_liquid).astype(float)
        filt = filt * decay + counts
        states[t] = filt
    return states, filt


# ==========================================================
# WORKER FUNCTION (called by multiprocessing)
# ==========================================================
def worker_evaluate(param_tuple):
    """
    Evaluate one LSM configuration in an isolated process.
    
    Input:  (trial_index, param_dict, data_path)
    Output: (trial_index, param_dict, nrmse)
    """
    trial_idx, params, data_path = param_tuple

    try:
        prefs.codegen.target = "numpy"
        defaultclock.dt = 1 * ms
        start_scope()

        # ---- Load data ----
        dataset = np.loadtxt(data_path, delimiter=',')
        dataset = dataset[:, 0]
        data = dataset.reshape(-1, 1)
        X_raw = data[:-1]
        Y_raw = data[1:]

        X_train_raw = X_raw[:TRAIN_LEN]
        Y_train_raw = Y_raw[:TRAIN_LEN]
        X_test_raw  = X_raw[TEST_START:TEST_START + TEST_LEN]
        Y_test_raw  = Y_raw[TEST_START:TEST_START + TEST_LEN]

        # ---- Import scaler utils ----
        import sys, os
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..',)))
        from lib.utils.reservoirpy import fit_scaler, transform_array

        # ---- Parse params ----
        p = params
        if p["test_warmup"] >= TEST_LEN:
            return (trial_idx, params, np.inf)

        tau_mem       = p["tau_mem_ms"] * ms
        step_duration = p["step_duration_ms"] * ms
        w_exc         = p["w_exc_mV"] * mV
        w_inh         = -p["inh_ratio"] * p["w_exc_mV"] * mV
        w_input       = p["w_input_mV"] * mV
        input_gain    = p["input_gain_mV"]

        scaler  = fit_scaler(X_train_raw, method=p["normalization"])
        X_train = transform_array(X_train_raw, scaler)
        Y_train = transform_array(Y_train_raw, scaler)
        X_test  = transform_array(X_test_raw, scaler)
        Y_test  = transform_array(Y_test_raw, scaler)

        pred_len      = TEST_LEN - p["test_warmup"]
        Y_true_scaled = Y_test[p["test_warmup"]:p["test_warmup"] + pred_len, 0]

        # ---- Build & train ----
        net, input_layer, spike_mon = build_liquid(
            p["N_liquid"], tau_mem, w_exc, w_inh, w_input, 42
        )

        net.restore('initial')
        S_train, _ = collect_states(
            net, input_layer, spike_mon, p["N_liquid"],
            step_duration, X_train, p["encoding"], input_gain,
        )

        mean_rate = S_train.mean()
        if mean_rate < 1e-3 or mean_rate > 1e4 or not np.isfinite(mean_rate):
            return (trial_idx, params, np.inf)

        S_train_fit = S_train[p["train_warmup"]:]
        Y_train_fit = Y_train[p["train_warmup"]:, 0]

        regression_model = p.get("regression_model", "ridge")
        if regression_model == "ridge":
            readout = Ridge(alpha=p["regression"])
        else:
            readout = Lasso(alpha=p["regression"], max_iter=10000)
        readout.fit(S_train_fit, Y_train_fit)

        # ---- Sync ----
        net.restore('initial')
        _, filt = collect_states(
            net, input_layer, spike_mon, p["N_liquid"],
            step_duration, X_test[:p["test_warmup"]],
            p["encoding"], input_gain,
        )

        # ---- Closed-loop ----
        Y_pred_scaled = np.zeros(pred_len)
        decay = np.exp(-float(step_duration / TAU_SYN_FIX))
        current_input = X_test[p["test_warmup"], 0]

        for k in range(pred_len):
            input_layer.I_drive = encode(current_input, p["encoding"], input_gain)
            spikes_before = len(spike_mon.t)
            net.run(step_duration)
            new_idx = np.array(spike_mon.i[spikes_before:])
            counts = np.bincount(new_idx, minlength=p["N_liquid"]).astype(float)
            filt = filt * decay + counts
            pred = readout.predict(filt.reshape(1, -1))[0]
            if not np.isfinite(pred) or abs(pred) > 1e6:
                return (trial_idx, params, np.inf)
            Y_pred_scaled[k] = pred
            current_input = pred

        rmse  = np.sqrt(np.mean((Y_true_scaled - Y_pred_scaled) ** 2))
        denom = np.std(Y_true_scaled)
        if denom == 0:
            return (trial_idx, params, np.inf)
        nrmse = rmse / denom

        return (trial_idx, params, float(nrmse))

    except Exception:
        return (trial_idx, params, np.inf)