import os
import time
from datetime import timedelta
import copy
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, RANSACRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from scipy.optimize import minimize
from defense import trim_regression, ransac_regression, huber_regression
from attack import optp_attack, statp_attack
from utils import get_cpu_info, load_data


def evaluate_poisoning_rate(args, current_count, total_count):
    attack_name, attack_func, model_proto, epsilon, dataset_path, target_column, min_len, begin_time = args
    process_start_time =  time.time()

    # Load data fresh here inside each process to avoid memory sharing issues:
    X, y = load_data(dataset_path, target_column, n_samples=min_len)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42)

    model = clone(model_proto)

    # Run attack and get poisoning time
    if attack_name == 'StatP':
        X_train_poisoned, y_train_poisoned, attack_time = attack_func(X_train, y_train, epsilon)
    elif attack_name == 'OPTP':
        X_train_poisoned, y_train_poisoned, attack_time = attack_func(
            X_train, y_train, X_val, y_val, epsilon=epsilon, base_model=model)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    # Save poisoned X and y to file
    poisoned_dir = "poisoned_data"
    os.makedirs(poisoned_dir, exist_ok=True)

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    model_name = type(model_proto).__name__
    filename = f"{dataset_name}-{attack_name}-{model_name}-eps{epsilon:.2f}.csv"

    df_poisoned = pd.DataFrame(X_train_poisoned, columns=X.columns)
    df_poisoned['target'] = y_train_poisoned
    df_poisoned.to_csv(os.path.join(poisoned_dir, filename), index=False)

    # Clean model mse and timing
    fit_time_clean = pred_time_clean = None
    if epsilon == 0.0:
        start = time.time()
        model.fit(X_train, y_train)
        fit_time_clean = time.time() - start

        start = time.time()
        y_pred_clean = model.predict(X_test)
        pred_time_clean = time.time() - start

        mse_clean = mean_squared_error(y_test, y_pred_clean)
    else:
        mse_clean = None

    # Poisoned model mse and timing
    model_poisoned = clone(model_proto)
    start = time.time()
    model_poisoned.fit(X_train_poisoned, y_train_poisoned)
    fit_time_poisoned = time.time() - start

    start = time.time()
    y_pred_poisoned = model_poisoned.predict(X_test)
    pred_time_poisoned = time.time() - start

    mse_poisoned = mean_squared_error(y_test, y_pred_poisoned)

    # Trim regression timing
    start = time.time()
    defended_model_trim = trim_regression(X_train_poisoned, y_train_poisoned, model_proto, keep_ratio=0.9)
    fit_time_trim = time.time() - start

    start = time.time()
    y_pred_trim = defended_model_trim.predict(X_test)
    pred_time_trim = time.time() - start

    mse_trim = mean_squared_error(y_test, y_pred_trim)

    # RANSAC regression timing
    start = time.time()
    defended_model_ransac = ransac_regression(X_train_poisoned, y_train_poisoned, base_model=model_proto)
    fit_time_ransac = time.time() - start

    start = time.time()
    y_pred_ransac = defended_model_ransac.predict(X_test)
    pred_time_ransac = time.time() - start

    mse_ransac = mean_squared_error(y_test, y_pred_ransac)

    # Huber Regression
    start = time.time()
    defended_model_huber = huber_regression(X_train_poisoned, y_train_poisoned)
    fit_time_huber = time.time() - start

    start = time.time()
    y_pred_huber = defended_model_huber.predict(X_test)
    pred_time_huber = time.time() - start

    mse_huber = mean_squared_error(y_test, y_pred_huber)


    # Get cpu info
    cpu_core, cpu_model = get_cpu_info()

    process_end_time = time.time()

    process_start_time_string = str(timedelta(seconds=process_start_time - begin_time))
    process_end_time_string = str(timedelta(seconds=process_end_time - begin_time))

    log_string = (f"process start time: {process_start_time_string}, "
                  f"process end time: {process_end_time_string}, "
                  f"attack time: {attack_time}, "
                  f"model: {model_proto}, "
                  f"dataset: {dataset_path}, "
                  f"epsilon: {epsilon}, "
                  f"attack name: {attack_name}, "
                  f"CPU core: {cpu_core}, "
                  f"CPU model: {cpu_model}")

    return {
        'dataset': dataset_path,
        'target': target_column,
        'attack': attack_name,
        'attack_time_sec': attack_time,
        'epsilon': epsilon,
        'model': type(model_proto).__name__,
        'mse_clean': mse_clean,
        'mse_poisoned': mse_poisoned,
        'mse_trim': mse_trim,
        'mse_ransac': mse_ransac,
        'mse_huber': mse_huber,
        'fit_time_clean': fit_time_clean,
        'pred_time_clean': pred_time_clean,
        'fit_time_poisoned': fit_time_poisoned,
        'pred_time_poisoned': pred_time_poisoned,
        'fit_time_trim': fit_time_trim,
        'pred_time_trim': pred_time_trim,
        'fit_time_ransac': fit_time_ransac,
        'pred_time_ransac': pred_time_ransac,
        'fit_time_huber': fit_time_huber,
        'pred_time_huber': pred_time_huber,
        'proc_start_time': process_start_time_string,
        'proc_end_time': process_end_time_string,
        'cpu_core': cpu_core,
        'cpu_model': cpu_model
    }, log_string

def evaluate_wrapper(task_args, counter, total_count, lock):
    task_id, attack_name, attack_func, model_proto, epsilon, dataset_path, target_column, min_len, begin_time = task_args
    with lock:
        counter.value += 1
        current_count = counter.value
    result, log_string = evaluate_poisoning_rate(
        (attack_name, attack_func, model_proto, epsilon, dataset_path, target_column, min_len, begin_time),
        current_count,
        total_count
    )
    return_log_string = f"end of task_id: {task_id} out of {total_count} | " + log_string
    print(return_log_string)
    return result, return_log_string



if __name__ == '__main__':
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

    min_len = float('inf')
    for path, target in datasets.items():
        df = pd.read_csv(path)
        min_len = min(min_len, len(df))
    print(f"Minimum dataset length: {min_len}")

    poisoning_rates = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]

    attacks = {
        'StatP': statp_attack,
        'OPTP': optp_attack
    }

    # Create a list of all combinations to evaluate
    all_args = []
    begin_time = time.time()
    for idx, (dataset_path, target_column) in enumerate(datasets.items()):
        for model_name, model_proto in models.items():
            for attack_name, attack_func in attacks.items():
                for epsilon in poisoning_rates:
                    task_id = len(all_args) + 1
                    all_args.append(
                        (task_id, attack_name, attack_func, model_proto, epsilon, dataset_path, target_column, min_len,
                         begin_time))

    print(f"Total experiments to run: {len(all_args)}")

    # Use multiprocessing to evaluate all experiments in parallel
    print(mp.cpu_count())

    with mp.Manager() as manager:
        counter = manager.Value('i', 0)
        lock = manager.Lock()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            outputs = pool.starmap(evaluate_wrapper, [(arg, counter, len(all_args), lock) for arg in all_args])

    results = []
    log_rows = []
    for result, log_row in outputs:
        results.append(result)
        log_rows.append(log_row)

    # Save logs
    with open("poisoning_experiment_log.txt", "w") as f:
        for log in log_rows:
            f.write(log + "\n")

    # Convert to DataFrame for easy analysis and saving
    df_results = pd.DataFrame(results)

    # Fill missing mse_clean values (None) with the value at epsilon=0 for that dataset/model/attack combo
    # (Because only epsilon=0 calculates mse_clean)
    for (dataset, model, attack), group in df_results.groupby(['dataset', 'model', 'attack']):
        clean_val = group.loc[group['epsilon'] == 0.0, 'mse_clean'].values
        if len(clean_val) > 0:
            fill_val = clean_val[0]
            df_results.loc[(df_results['dataset'] == dataset) &
                           (df_results['model'] == model) &
                           (df_results['attack'] == attack) &
                           (df_results['mse_clean'].isna()), 'mse_clean'] = fill_val

    # Save results
    df_results.to_csv('poisoning_experiment_results.csv', index=False)
    print("Results saved to poisoning_experiment_results.csv")


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
