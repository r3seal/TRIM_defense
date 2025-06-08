import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from scipy.optimize import minimize
from utils import timeit


@timeit
def optp_attack(X_train, y_train, X_val, y_val, epsilon=0.1, max_iter=3, base_model=None):
    n_poison = int(epsilon * len(X_train))
    n_features = X_train.shape[1]

    if n_poison == 0:
        return X_train, y_train

    X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
    y_min, y_max = y_train.min(), y_train.max()

    X_p = np.random.uniform(X_min, X_max, size=(n_poison, n_features))
    y_p = np.random.uniform(y_min, y_max, size=n_poison)

    theta_init = np.hstack([X_p.ravel(), y_p])

    def objective(theta):
        X_p_flat = theta[:n_poison * n_features].reshape((n_poison, n_features))
        y_p_flat = theta[n_poison * n_features:]

        X_poisoned = np.vstack([X_train, X_p_flat])
        y_poisoned = np.hstack([y_train, y_p_flat])

        model = clone(base_model)
        model.fit(X_poisoned, y_poisoned)
        y_pred_val = model.predict(X_val)
        return mean_squared_error(y_val, y_pred_val)

    bounds = [(low, high) for low, high in zip(X_min, X_max)] * n_poison + [(y_min, y_max)] * n_poison

    result = minimize(
        objective,
        theta_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'disp': False}
    )

    theta_opt = result.x
    X_p_opt = theta_opt[:n_poison * n_features].reshape((n_poison, n_features))
    y_p_opt = theta_opt[n_poison * n_features:]

    X_poisoned = np.vstack([X_train, X_p_opt])
    y_poisoned = np.hstack([y_train, y_p_opt])

    return X_poisoned, y_poisoned


@timeit
def statp_attack(X, y, epsilon=0.1, rounding=True):
    n_samples = int(epsilon * len(X))

    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    X_poison = np.random.multivariate_normal(mean, cov, size=n_samples)

    if rounding:
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        corners = np.random.choice([0, 1], size=X_poison.shape)
        X_poison = corners * X_max + (1 - corners) * X_min

    y_min, y_max = y.min(), y.max()
    y_poison = np.random.choice([y_min, y_max], size=n_samples)

    X_attacked = np.vstack([X, X_poison])
    y_attacked = np.hstack([y, y_poison])
    return X_attacked, y_attacked
