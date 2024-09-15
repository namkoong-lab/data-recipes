import json
import logging
import os
from pathlib import Path

import numpy as np
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
        file_handler = logging.FileHandler(filename, mode="w")
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
