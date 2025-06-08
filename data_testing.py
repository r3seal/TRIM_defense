import os
import time
import pandas as pd
from glob import glob
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from main import trim_regression, ransac_regression

def evaluate_all_poisoned_data(poisoned_data_dir, dataset_dir, model_dict, target_lookup, output_csv="all_poisoned_eval.csv"):
    result_rows = []
    poisoned_files = sorted(glob(os.path.join(poisoned_data_dir, "*.csv")))

    if not poisoned_files:
        print("No poisoned CSVs found.")
        return

    for file in poisoned_files:
        basename = os.path.basename(file)
        try:
            parts = basename.replace(".csv", "").split("-")
            dataset_name = parts[0]
            dataset_name2 = parts[1]
            attack = parts[2]
            model_name = parts[3]
            eps_str = parts[4]
            epsilon = float(eps_str.replace("eps", ""))
        except Exception as e:
            print(f"Skipping file with bad name: {basename} — error: {e}")
            continue

        dataset_path = dataset_dir + f"/{dataset_name}-{dataset_name2}.csv"
        if not os.path.exists(dataset_path):
            print(f"Missing dataset: {dataset_path}")
            continue

        if model_name not in model_dict:
            print(f"Model {model_name} not in provided model_dict.")
            continue

        if dataset_path not in target_lookup:
            print(dataset_path)
            print(target_lookup)
            print(f"No target column specified for dataset: {dataset_path}")
            continue

        target_column = target_lookup[dataset_path]
        model_proto = model_dict[model_name]

        # Load and prepare clean/original dataset
        df_orig = pd.read_csv(dataset_path)
        if target_column not in df_orig.columns:
            print(f"Missing target column '{target_column}' in dataset: {dataset_path}")
            continue

        y_full = df_orig[target_column]
        X_full = df_orig.drop(columns=[target_column])
        X_train_full, X_test, y_train_full, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

        if epsilon == 0:
            model_clean = clone(model_proto)
            start = time.time()
            model_clean.fit(X_train, y_train)
            fit_time_clean = time.time() - start

            start = time.time()
            y_pred_clean = model_clean.predict(X_test)
            pred_time_clean = time.time() - start

            mse_clean = mean_squared_error(y_test, y_pred_clean)
        else:
            mse_clean = fit_time_clean = pred_time_clean = ""

        # Load poisoned data (always using column 'target')
        df_poisoned = pd.read_csv(file)
        if 'target' not in df_poisoned.columns:
            print(f"Missing 'target' column in poisoned file: {file}")
            continue

        y_poisoned = df_poisoned['target'].values
        X_poisoned = df_poisoned.drop(columns=['target']).values

        # Poisoned model
        model_poisoned = clone(model_proto)
        start = time.time()
        model_poisoned.fit(X_poisoned, y_poisoned)
        fit_time_poisoned = time.time() - start

        start = time.time()
        y_pred_poisoned = model_poisoned.predict(X_test)
        pred_time_poisoned = time.time() - start

        mse_poisoned = mean_squared_error(y_test, y_pred_poisoned)

        # Trim defense
        try:
            start = time.time()
            model_trim = trim_regression(X_poisoned, y_poisoned, model_proto)
            fit_time_trim = time.time() - start

            start = time.time()
            y_pred_trim = model_trim.predict(X_test)
            pred_time_trim = time.time() - start

            mse_trim = mean_squared_error(y_test, y_pred_trim)
        except Exception:
            fit_time_trim = pred_time_trim = mse_trim = None

        # RANSAC defense
        try:
            start = time.time()
            model_ransac = ransac_regression(X_poisoned, y_poisoned, model_proto)
            fit_time_ransac = time.time() - start

            start = time.time()
            y_pred_ransac = model_ransac.predict(X_test)
            pred_time_ransac = time.time() - start

            mse_ransac = mean_squared_error(y_test, y_pred_ransac)
        except Exception:
            fit_time_ransac = pred_time_ransac = mse_ransac = None

        result_rows.append({
            "dataset": dataset_path,
            "target": target_column,
            "attack": attack,
            "model": model_name,
            "epsilon": epsilon,
            "mse_clean": mse_clean,
            "mse_poisoned": mse_poisoned,
            "mse_trim": mse_trim,
            "mse_ransac": mse_ransac,
            "attack_time_sec": 0.0,
            "fit_time_clean": fit_time_clean,
            "pred_time_clean": pred_time_clean,
            "fit_time_poisoned": fit_time_poisoned,
            "pred_time_poisoned": pred_time_poisoned,
            "fit_time_trim": fit_time_trim,
            "pred_time_trim": pred_time_trim,
            "fit_time_ransac": fit_time_ransac,
            "pred_time_ransac": pred_time_ransac,
        })

    df_all = pd.DataFrame(result_rows)
    df_all.to_csv(output_csv, index=False)
    print(f"All results saved to {output_csv}")

# Define models
model_dict = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=0.1),
    "Ridge": Ridge(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Map dataset filenames to their true target column names
target_lookup = {
    'datasets/house-processed.csv': 'SalePrice',
    'datasets/loan-processed.csv': 'int_rate',
    'datasets/pharm-preproc.csv': 'TherapeuticDoseofWarfarin'
}


# Run evaluation
evaluate_all_poisoned_data(
    poisoned_data_dir="poisoned_data",
    dataset_dir="datasets",
    model_dict=model_dict,
    target_lookup=target_lookup,
    output_csv="results/all_poisoned_eval.csv"
)
