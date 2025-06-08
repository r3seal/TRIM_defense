import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.base import clone
from sklearn.linear_model import HuberRegressor

def huber_regression(X, y, epsilon=1.35):
    model = HuberRegressor(epsilon=epsilon, max_iter=1000)
    model.fit(X, y)
    return model


def trim_regression(X, y, model, keep_ratio=0.9, max_iter=5000, tol=1e-5):
    n_samples = X.shape[0]
    n_keep = int(keep_ratio * n_samples)
    indices = np.arange(n_samples)
    prev_error = np.inf
    clf = clone(model)

    for _ in range(max_iter):
        clf.fit(X[indices], y[indices])
        preds = clf.predict(X)
        residuals = (preds - y) ** 2
        sorted_indices = np.argsort(residuals)[:n_keep]
        current_error = residuals[sorted_indices].sum()
        if np.abs(prev_error - current_error) < tol:
            break
        prev_error = current_error
        indices = sorted_indices

    clf.fit(X[indices], y[indices])
    return clf


def ransac_regression(X, y, base_model, min_samples=0.9):
    model = RANSACRegressor(estimator=clone(base_model), min_samples=min_samples,
                            max_trials=100, random_state=42)
    model.fit(X, y)
    return model