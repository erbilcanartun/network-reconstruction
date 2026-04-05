import numpy as np


# ESN FUNCTIONS
def generate_esn(n_inputs, n_reservoir, spectral_radius=0.9, input_scaling=1.0, bias_scaling=1.0, connectivity=1.0, seed=42,):
    
    rng = np.random.default_rng(seed)

    Win = rng.uniform(low=-0.5 * input_scaling, high=0.5 * input_scaling, size=(n_reservoir, n_inputs))
    Wbias = rng.uniform(low=-0.5 * bias_scaling, high=0.5 * bias_scaling, size=(n_reservoir, 1))
    W = rng.uniform(low=-0.5, high=0.5, size=(n_reservoir, n_reservoir))

    # Sparsify if needed
    if connectivity < 1.0:
        mask = rng.random((n_reservoir, n_reservoir)) < connectivity
        W *= mask

    # Scale to desired spectral radius
    eigvals = np.linalg.eigvals(W)
    rho = np.max(np.abs(eigvals))
    W *= spectral_radius / rho

    return Win, W, Wbias


def esn_step(x, u, Win, W, Wbias, leak_rate):
    preactivation = Win @ u + W @ x + Wbias
    x_new = (1.0 - leak_rate) * x + leak_rate * np.tanh(preactivation)
    return x_new


def fit_esn(X_train, Y_train, Win, W, Wbias, leak_rate=1.0, washout=100, ridge=1e-8):
    """
    Train readout using ridge regression.

    X_train: (T, n_inputs)
    Y_train: (T, n_outputs)
    """
    T = X_train.shape[0]
    n_res = W.shape[0]
    n_inputs = X_train.shape[1]

    x = np.zeros((n_res, 1))

    X_collect = []
    Y_collect = []

    for t in range(T):
        u = X_train[t].reshape(n_inputs, 1)
        x = esn_step(x, u, Win, W, Wbias, leak_rate)

        if t >= washout:
            extended_state = np.vstack([np.ones((1, 1)), u, x])[:, 0]
            X_collect.append(extended_state)
            Y_collect.append(Y_train[t])

    X_design = np.array(X_collect).T
    Y_target = np.array(Y_collect).T

    A = X_design @ X_design.T + ridge * np.eye(X_design.shape[0])
    B = Y_target @ X_design.T

    # Equivalent to ridge regression formula, but numerically a bit cleaner
    Wout = np.linalg.solve(A.T, B.T).T

    return Wout


def run_esn_open_loop(X_seq, Win, W, Wbias, Wout=None, leak_rate=1.0, x0=None):
    """
    Run ESN using true inputs.
    Returns final state, and optionally outputs.
    """
    T = X_seq.shape[0]
    n_res = W.shape[0]
    n_inputs = X_seq.shape[1]

    x = np.zeros((n_res, 1)) if x0 is None else x0.copy()

    states = []
    outputs = []

    for t in range(T):
        u = X_seq[t].reshape(n_inputs, 1)
        x = esn_step(x, u, Win, W, Wbias, leak_rate)
        states.append(x[:, 0].copy())

        if Wout is not None:
            extended_state = np.vstack([np.ones((1, 1)), u, x])
            y = Wout @ extended_state
            outputs.append(y[:, 0].copy())

    states = np.array(states)

    if Wout is not None:
        outputs = np.array(outputs)
        return states, outputs, x

    return states, x


def predict_closed_loop(initial_input, steps, Win, W, Wbias, Wout, leak_rate=1.0, x0=None):
    """
    Closed-loop prediction:
    predicted output becomes next input.
    """
    n_res = W.shape[0]
    n_inputs = Win.shape[1]

    x = np.zeros((n_res, 1)) if x0 is None else x0.copy()
    u = np.asarray(initial_input).reshape(n_inputs, 1)

    Y_pred = []

    for _ in range(steps):
        x = esn_step(x, u, Win, W, Wbias, leak_rate)

        extended_state = np.vstack([np.ones((1, 1)), u, x])
        y = Wout @ extended_state
        Y_pred.append(y[:, 0].copy())

        u = y.reshape(n_inputs, 1)

    return np.array(Y_pred), x


def compute_nrmse(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / (np.std(y_true) + eps)