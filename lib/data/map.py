import numpy as np


def add_input_noise(data, noise_std):
    return data + noise_std * np.random.randn(len(data))


# ----------------------------
# Discrete systems
# ----------------------------
def logistic_map(r, x0, T):
    x = np.zeros(T)
    x[0] = x0
    for t in range(1, T):
        x[t] = r * x[t - 1] * (1 - x[t - 1])
    return x


def rulkov_map(T, alpha=4.1, mu=0.001, sigma=0.001, x0=-1.5, y0=-2.0):
    x = np.zeros(T)
    y = np.zeros(T)

    x[0] = x0
    y[0] = y0

    for n in range(T - 1):
        x[n + 1] = alpha / (1 + x[n] ** 2) + y[n]
        y[n + 1] = y[n] - mu * (x[n] - sigma)

    return x


def doubling_map(x0, T):
    """
    Doubling / binary shift map:
      x_{t+1} = (2 x_t) mod 1
    Implemented in the same piecewise style as triplet_map.
    """
    x = np.zeros(T)
    x[0] = x0 % 1.0

    for t in range(1, T):
        if x[t - 1] < 0.5:
            x[t] = 2.0 * x[t - 1]
        else:
            x[t] = 2.0 * x[t - 1] - 1.0

    return x


def triplet_map(x0, T):
    """
    Triplet / ternary shift map:
      x_{t+1} = (3 x_t) mod 1
    Implemented piecewise for clarity.
    """
    x = np.zeros(T)
    x[0] = x0 % 1.0

    for t in range(1, T):
        if x[t - 1] < 1 / 3:
            x[t] = 3 * x[t - 1]
        elif x[t - 1] < 2 / 3:
            x[t] = 3 * x[t - 1] - 1
        else:
            x[t] = 3 * x[t - 1] - 2

    return x


# ----------------------------
# Continuous systems
# ----------------------------
def lorenz_system(T, dt=0.01, sigma=10.0, rho=28.0, beta=8 / 3, x0=(1.0, 1.0, 1.0)):
    x = np.zeros(T)
    y = np.zeros(T)
    z = np.zeros(T)

    x[0], y[0], z[0] = x0

    for t in range(T - 1):
        dx = sigma * (y[t] - x[t])
        dy = x[t] * (rho - z[t]) - y[t]
        dz = x[t] * y[t] - beta * z[t]

        x[t + 1] = x[t] + dt * dx
        y[t + 1] = y[t] + dt * dy
        z[t + 1] = z[t] + dt * dz

    return x


def rossler_system(T, dt=0.05, a=0.2, b=0.2, c=5.7, x0=(1.0, 0.0, 0.0)):
    x = np.zeros(T)
    y = np.zeros(T)
    z = np.zeros(T)

    x[0], y[0], z[0] = x0

    for t in range(T - 1):
        dx = -y[t] - z[t]
        dy = x[t] + a * y[t]
        dz = b + z[t] * (x[t] - c)

        x[t + 1] = x[t] + dt * dx
        y[t + 1] = y[t] + dt * dy
        z[t + 1] = z[t] + dt * dz

    return x


def mackey_glass(T, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0, x0=1.2):
    max_delay = int(tau / dt)
    x = np.zeros(T + max_delay + 1)
    x[: max_delay + 1] = x0

    for t in range(max_delay, T + max_delay):
        x_tau = x[t - max_delay]
        dx = beta * x_tau / (1 + x_tau ** n) - gamma * x[t]
        x[t + 1] = x[t] + dt * dx

    return x[max_delay + 1 : max_delay + 1 + T]