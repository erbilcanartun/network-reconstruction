import numpy as np


# ==========================================================
# NORMALIZATION UTILITIES
# ==========================================================
def fit_scaler(train_array, method="zscore"):
    """
    Fit scaler using training data only.
    """
    scaler = {"method": method}

    if method == "none":
        return scaler

    if method == "zscore":
        mu = np.mean(train_array, axis=0)
        sigma = np.std(train_array, axis=0)
        sigma[sigma == 0] = 1.0
        scaler["mu"] = mu
        scaler["sigma"] = sigma
        return scaler

    if method == "minmax01":
        xmin = np.min(train_array, axis=0)
        xmax = np.max(train_array, axis=0)
        span = xmax - xmin
        span[span == 0] = 1.0
        scaler["xmin"] = xmin
        scaler["span"] = span
        return scaler

    if method == "minmax11":
        xmin = np.min(train_array, axis=0)
        xmax = np.max(train_array, axis=0)
        span = xmax - xmin
        span[span == 0] = 1.0
        scaler["xmin"] = xmin
        scaler["span"] = span
        return scaler

    raise ValueError(f"Unknown normalization method: {method}")


def transform_array(array, scaler):
    method = scaler["method"]

    if method == "none":
        return array.copy()

    if method == "zscore":
        return (array - scaler["mu"]) / scaler["sigma"]

    if method == "minmax01":
        return (array - scaler["xmin"]) / scaler["span"]

    if method == "minmax11":
        return 2.0 * (array - scaler["xmin"]) / scaler["span"] - 1.0

    raise ValueError(f"Unknown normalization method: {method}")


def inverse_transform_array(array, scaler):
    method = scaler["method"]

    if method == "none":
        return array.copy()

    if method == "zscore":
        return array * scaler["sigma"] + scaler["mu"]

    if method == "minmax01":
        return array * scaler["span"] + scaler["xmin"]

    if method == "minmax11":
        return 0.5 * (array + 1.0) * scaler["span"] + scaler["xmin"]

    raise ValueError(f"Unknown normalization method: {method}")