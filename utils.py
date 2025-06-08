import os
import time
import pandas as pd
from functools import wraps
import ctypes
import platform

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

def load_data(file_path, target_column, n_samples=None):
    df = pd.read_csv(file_path)
    if n_samples is not None:
        df = df.sample(n=n_samples, random_state=42)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

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
