import os
import time
from datetime import timedelta
import copy
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from scipy.optimize import minimize
from defense import trim_regression, ransac_regression, huber_regression
from attack import optp_attack, statp_attack
from utils import get_cpu_info, load_data, load_data_and_preprocess


# main testing function
def evaluate_poisoning_rate(args):
    attack_name, attack_func, model_proto, epsilon, dataset_path, target_column, min_len, begin_time, ifprocessed = args
    process_start_time =  time.time()

    # Load data
    if ifprocessed:
        X, y = load_data(dataset_path, target_column, n_samples=min_len)
    else:
        X, y = load_data_and_preprocess(dataset_path, target_column, n_samples=min_len)
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

    # Clean model
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

    # Poisoned model
    model_poisoned = clone(model_proto)
    start = time.time()
    model_poisoned.fit(X_train_poisoned, y_train_poisoned)
    fit_time_poisoned = time.time() - start

    start = time.time()
    y_pred_poisoned = model_poisoned.predict(X_test)
    pred_time_poisoned = time.time() - start

    mse_poisoned = mean_squared_error(y_test, y_pred_poisoned)

    # Trim regression
    start = time.time()
    defended_model_trim = trim_regression(X_train_poisoned, y_train_poisoned, model_proto)
    fit_time_trim = time.time() - start

    start = time.time()
    y_pred_trim = defended_model_trim.predict(X_test)
    pred_time_trim = time.time() - start

    mse_trim = mean_squared_error(y_test, y_pred_trim)

    # RANSAC regression
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

    # string for logging progress
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

# wrapper
def evaluate_wrapper(task_args, total_count):
    task_id, attack_name, attack_func, model_proto, epsilon, dataset_path, target_column, min_len, begin_time, ifprocessed = task_args
    result, log_string = evaluate_poisoning_rate(
        (attack_name, attack_func, model_proto, epsilon, dataset_path, target_column, min_len, begin_time, ifprocessed)
    )
    return_log_string = f"end of task_id: {task_id} out of {total_count} | " + log_string
    print(return_log_string)
    return result, return_log_string



if __name__ == '__main__':
    # models for experiments
    models = {
        'OLS': LinearRegression(),
        'LASSO': Lasso(alpha=0.1),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }

    # datasets for experiments (path: [targeted value, if already processed])
    datasets = {
        'datasets/house-processed.csv': ['SalePrice', True],
        'datasets/loan-processed.csv': ['int_rate', True],
        'datasets/pharm-preproc.csv': ['TherapeuticDoseofWarfarin', True],
        'datasets/housing.csv': ['housing_median_age', False],
        'datasets/creditcard.csv': ['Amount', False],
        'datasets/Taxi_Trip_Data_preprocessed.csv': ['fare_amount', False]
    }

    # find the minimum dataset length
    min_len = float('inf')
    for path, _ in datasets.items():
        df = pd.read_csv(path)
        min_len = min(min_len, len(df))
    print(f"Minimum dataset length: {min_len}")

    # poisoning rates
    poisoning_rates = [0.03 * i for i in range(8)]

    # attacks
    attacks = {
        'StatP': statp_attack,
        'OPTP': optp_attack
    }

    # Create a list of all combinations to evaluate
    all_args = []
    begin_time = time.time()
    for idx, (dataset_path, values) in enumerate(datasets.items()):
        for model_name, model_proto in models.items():
            for attack_name, attack_func in attacks.items():
                for epsilon in poisoning_rates:
                    task_id = len(all_args) + 1
                    all_args.append(
                        (task_id, attack_name, attack_func, model_proto, epsilon, dataset_path, values[0], min_len,
                         begin_time, values[1]))

    print(f"Total experiments to run: {len(all_args)}")

    # Use multiprocessing to evaluate all experiments in parallel
    print(f"Total number of cpu's: {mp.cpu_count()}")

    with mp.Manager() as manager:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            outputs = pool.starmap(evaluate_wrapper, [(arg, len(all_args)) for arg in all_args])

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
