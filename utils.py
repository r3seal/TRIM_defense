import os
import time
import pandas as pd
from functools import wraps
import ctypes
import platform
from sklearn.preprocessing import StandardScaler
import numpy as np

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if isinstance(result, tuple):
            return (*result, elapsed)
        else:
            return result, elapsed
    return wrapper

# load data
def load_data(file_path, target_column, n_samples=None):
    df = pd.read_csv(file_path)
    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=42)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def load_data_and_preprocess(file_path, target_column, sep=',', n_samples=None):
    df = pd.read_csv(file_path, sep=sep)
    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=42)

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

# get number of cpu core
def get_cpu_core():
    if platform.system() == 'Windows':
        try:
            # Windows API to get processor number
            GetCurrentProcessorNumber = ctypes.windll.kernel32.GetCurrentProcessorNumber
            GetCurrentProcessorNumber.restype = ctypes.c_uint
            return GetCurrentProcessorNumber()
        except Exception:
            return "Unknown"
    else:
        # Fallback for Unix/Linux
        try:
            return os.sched_getcpu()
        except AttributeError:
            return "Unknown"

# get number of cpu core and cpu model
def get_cpu_info():
    cpu_core = get_cpu_core()

    cpu_model = platform.processor()
    if not cpu_model:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line or "Hardware" in line:
                        cpu_model = line.strip().split(":")[-1].strip()
                        break
        except Exception:
            cpu_model = "Unknown"

    return cpu_core, cpu_model
