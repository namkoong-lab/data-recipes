import json
import logging
import os
from pathlib import Path

import numpy as np
import psutil
import torch

PROJ_NAME = "data_mixing"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def get_logger(level=logging.INFO, filename=None, add_console=True):
    fmt_str = "%(asctime)s, [%(levelname)s, %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt_str)
    logger = logging.getLogger(PROJ_NAME)

    if add_console:
        logger.handlers.clear()
        console_handler = logging.StreamHandler()
        log_formatter = logging.Formatter(fmt_str)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode="a")
        log_formatter = logging.Formatter(fmt_str)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    if level is not None:
        logger.setLevel(level)
    logger.propagate = False
    return logger


def make_folder(folder):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)


def boolean(input):
    """Transform input to boolean, mostly used for ArgumentParser"""
    true_conditions = [
        lambda x: x.lower() == "true",
        lambda x: x.lower() == "yes",
        lambda x: x == "1",
    ]
    for cond in true_conditions:
        if cond(input) is True:
            return True
    return False


def type_or_none(input, type):
    if input.lower() == "none":
        return None
    else:
        return type(input)


def get_label_marginals(labels, num_labels=None):
    seen_labels, seen_counts = np.unique(labels, return_counts=True)
    seen_labels = seen_labels.astype(int)

    num_labels = np.max(labels) + 1 if num_labels is None else num_labels
    all_counts = np.zeros(num_labels)
    for idx, label in enumerate(seen_labels):
        all_counts[label] = seen_counts[idx]

    return all_counts / np.sum(all_counts)


def torch2np(array):
    if type(array) != np.ndarray:
        return array.cpu().numpy()
    return array


def is_completed(folder):
    return os.path.exists(f"{folder}/done.txt")


def note_completed(folder):
    with open(f"{folder}/done.txt", "w") as f:
        print(f"done", file=f)


def sort_both(a, b, reverse=True):
    sorted_items = list(zip(*sorted(zip(a, b), reverse=reverse)))
    return sorted_items[0], sorted_items[1]


def idx2onehot(label_idc, num_labels):
    onehot = torch.zeros((label_idc.shape[0], num_labels), device=label_idc.device)
    onehot[np.arange(label_idc.shape[0]), label_idc] = 1.0
    return onehot


def display_args(args, logger=None):
    print_f = print if logger is None else logger.info
    print_f(f"Arguments: {json.dumps(vars(args), indent=2)}")


def set_hf_cache(cache_dir):
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir


def pretty_number(num):
    # Write number in M, B, T format
    num = float(num)
    if abs(num) < 1e6:
        return f"{num:.2f}"
    elif abs(num) < 1e9:
        return f"{num/1e6:.2f}M"
    elif abs(num) < 1e12:
        return f"{num/1e9:.2f}B"
    else:
        return f"{num/1e12:.2f}T"


def json_dump(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert bytes to GB
    return memory_gb


def log_memory(message=""):
    """Print memory usage with optional message"""
    memory_gb = get_memory_usage()
    print(f"Memory Usage {message}: {memory_gb:.2f} GB")


def round_with_same_sum(values):
    target = round(sum(values))
    rounded = np.round(values).astype(int)

    if sum(rounded) == target:
        return rounded

    sum_diff = sum(rounded) - target
    diffs = rounded - values

    if sum_diff > 0:
        # Round down
        cands = np.where(diffs > 0)[0]
        cands = cands[np.argsort(diffs[cands])][::-1]
        rounded[cands[:sum_diff]] -= 1
    else:
        cands = np.where(diffs < 0)[0]
        cands = cands[np.argsort(diffs[cands])]
        rounded[cands[: abs(sum_diff)]] += 1
    return rounded


def lprint(message, logger, level=logging.INFO):
    if type(level) == str:
        level = level.upper()
        level = logging.getLevelName(level)

    if logger is None:
        print(message)
    else:
        logger.log(level, message)


def print_numpy_memory(arr, prefix="", logger=None):
    """Print memory usage of a numpy array"""
    lprint(f"{prefix} Memory usage: {arr.nbytes / (1024 ** 3)} GB", logger=logger)


def distribute_jobs(total_jobs, num_workers, return_indices=True):
    """Distribute total_jobs to num_workers"""
    jobs_per_worker = total_jobs // num_workers
    remaining_jobs = total_jobs % num_workers

    worker_jobs = [jobs_per_worker] * num_workers
    for i in range(remaining_jobs):
        worker_jobs[i] += 1

    if not return_indices:
        return worker_jobs

    worker_indices = []
    start_idx = 0
    for wj in worker_jobs:
        worker_indices.append((start_idx, start_idx + wj))
        start_idx += wj
    return worker_indices


def get_time():
    """Get current time in yyyy/mm/dd hh:mm:ss format"""
    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
