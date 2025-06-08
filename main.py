import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler


def load_data(file_path, target_column, sep=','):
    df = pd.read_csv(file_path, sep=sep)

    if target_column not in df.columns:
        raise ValueError()

    X = df.drop(columns=[target_column])
    y = df[target_column]\

    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        X = X.fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X, y

def statp_attack(X, y, epsilon=0.1, rounding=True):
    n_samples = int(epsilon * len(X))

    # Estimate mean and covariance
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    # Sample from multivariate normal
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

def trim_regression(X, y, model, keep_ratio=0.9, max_iter=1000, tol=1e-5):
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

models = {
    'OLS': LinearRegression(),
    'LASSO': Lasso(alpha=0.1),
    'Ridge': Ridge(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

datasets = {
    'datasets/house-processed.csv': 'SalePrice',
    'datasets/loan-processed.csv': 'int_rate',
    'datasets/pharm-preproc.csv': 'TherapeuticDoseofWarfarin'
}

poisoning_rates = [i/50 for i in range(11)]

for dataset_path, target_column in datasets.items():
    print(f"\nPrzetwarzanie: {dataset_path}")
    X, y = load_data(dataset_path, target_column, sep=',')
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'MSE vs Poisoning Rate dla zbioru: {dataset_path}', fontsize=16)

    axs = axs.flatten()

    for ax, (model_name, model_proto) in zip(axs, models.items()):
        mse_clean_list = []
        mse_poisoned_list = []
        mse_defended_list = []

        for epsilon in poisoning_rates:
            model = model_proto.__class__(**model_proto.get_params())

            if epsilon == 0.0:
                X_train_poisoned, y_train_poisoned = X_train, y_train
            else:
                X_train_poisoned, y_train_poisoned = statp_attack(X_train, y_train, epsilon)

            # 1) Czysty model
            if epsilon == 0.0:
                model.fit(X_train, y_train)
                y_pred_clean = model.predict(X_test)
                mse_clean = mean_squared_error(y_test, y_pred_clean)
            mse_clean_list.append(mse_clean)

            # 2) Model po zatruciu
            model.fit(X_train_poisoned, y_train_poisoned)
            y_pred_poisoned = model.predict(X_test)
            mse_poisoned = mean_squared_error(y_test, y_pred_poisoned)
            mse_poisoned_list.append(mse_poisoned)

            # 3) Obrona TRIM
            defended_model = trim_regression(X_train_poisoned, y_train_poisoned, model, keep_ratio=0.9)
            y_pred_defended = defended_model.predict(X_test)
            mse_defended = mean_squared_error(y_test, y_pred_defended)
            mse_defended_list.append(mse_defended)

        ax.plot(poisoning_rates, mse_clean_list, label='MSE czyste dane', marker='o')
        ax.plot(poisoning_rates, mse_poisoned_list, label='MSE po ataku', marker='o')
        ax.plot(poisoning_rates, mse_defended_list, label='MSE po obronie TRIM', marker='o')

        ax.set_title(model_name)
        ax.set_xlabel('Poisoning rate (ε)')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
