"""This scripts count the number of documents, characters, and tokens in the dataset."""

import ctypes
import logging
from collections import defaultdict
from multiprocessing import Process, RawArray
from typing import List

import init_path
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import utils.misc_utils as mscu
from utils.global_variables import ANON_GB, GB


def process_chunk(
    ds, start, end, doc_subset_indices, doc_chars, process_index, SLIMPAJAMA_SUBSETS
):
    pbar = tqdm(
        total=end - start,
        position=process_index,
        desc=f"Process {process_index}: {start}-{end}",
        leave=False,
    )
    for i in range(start, end):
        cur_chars = len(ds[i]["text"])
        cur_subset = ds[i]["meta"]["redpajama_set_name"]
        cur_subset_idx = SLIMPAJAMA_SUBSETS.index(cur_subset)

        doc_subset_indices[i] = cur_subset_idx
        doc_chars[i] = cur_chars
        pbar.update(1)
    pbar.close()


def process_split(
    split: str,
    ds,
    SLIMPAJAMA_SUBSETS: List[str],
    logger: logging.Logger,
    num_workers: int,
):
    logger.info("*" * 84)
    logger.info(f"Split: {split}")
    num_docs = len(ds[split])

    doc_subset_indices = RawArray(ctypes.c_int, [-1] * num_docs)
    doc_chars = RawArray(ctypes.c_int, [-1] * num_docs)
    batch_size = num_docs // num_workers

    ranges = [(i * batch_size, (i + 1) * batch_size) for i in range(num_workers)]
    ranges[-1] = ((num_workers - 1) * batch_size, num_docs)

    processes = []
    for process_index, (start, end) in enumerate(ranges):
        p = Process(
            target=process_chunk,
            args=(
                ds[split],
                start,
                end,
                doc_subset_indices,
                doc_chars,
                process_index,
                SLIMPAJAMA_SUBSETS,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    doc_subset_indices = np.array(list(doc_subset_indices))
    doc_chars = np.array(list(doc_chars))

    logger.info(f"Indices collected. ")

    subset_indices = defaultdict(list)
    subset_ndocs = defaultdict(int)
    subset_chars = defaultdict(int)
    subset_tokens = defaultdict(int)
    for idx, subset in tqdm(
        enumerate(SLIMPAJAMA_SUBSETS), desc="Turning indices into list and counting. "
    ):
        subset_indices[subset] = np.where(doc_subset_indices == idx)[0]
        subset_ndocs[subset] = len(subset_indices[subset])
        subset_chars[subset] = np.sum(doc_chars[subset_indices[subset]]).item()
        subset_tokens[subset] = 4 * subset_chars[subset]  # est. 4 tokens per character

    for subset in SLIMPAJAMA_SUBSETS:
        logger.info(f"Subset: {subset}")
        logger.info(f"Number of documents: {mscu.pretty_number(subset_ndocs[subset])}")
        logger.info(f"Number of characters: {mscu.pretty_number(subset_chars[subset])}")
        logger.info(f"Number of tokens: {mscu.pretty_number(subset_tokens[subset])}")

    return {
        "subset_indices": subset_indices,
        "subset_ndocs": subset_ndocs,
        "subset_chars": subset_chars,
        "subset_tokens": subset_tokens,
    }


def process_dataset(
    ds, SLIMPAJAMA_SUBSETS: List[str], num_workers: int, logger: logging.Logger
):
    results = {}
    for split in ds.keys():
        results[split] = process_split(
            split, ds, SLIMPAJAMA_SUBSETS, logger=logger, num_workers=num_workers
        )
    return results


if __name__ == "__main__":
    log_folder = f"{GB['log_folder']}/index_slimpajama"
    mscu.make_folder(log_folder)
    logger = mscu.get_logger(
        add_console=True, filename=f"{log_folder}/index_slimpajama.log"
    )

    SLIMPAJAMA_SUBSETS = [
        "RedPajamaCommonCrawl",
        "RedPajamaC4",
        "RedPajamaWikipedia",
        "RedPajamaStackExchange",
        "RedPajamaGithub",
        "RedPajamaArXiv",
        "RedPajamaBook",
    ]

    num_workers = 100
    logger.info(f"Number of workers: {num_workers}")

    ds = load_dataset("cerebras/SlimPajama-627B", cache_dir=GB["dataset_path"])
    results = process_dataset(
        ds, SLIMPAJAMA_SUBSETS, num_workers=num_workers, logger=logger
    )

    for split in ["train", "validation", "test"]:
        folder = f"{GB['scratch_folder']}/data-mixing/slimpajama/{split}"
        mscu.make_folder(folder)

        fname = f"{folder}/index.npy"
        logger.info(f"Saving indices to {fname}")
        np.save(fname, results[split]["subset_indices"])

        details_fname = f"{folder}/details.json"
        logger.info(f"Saving details to {details_fname}")
        mscu.json_dump(
            {
                "subset_ndocs": results[split]["subset_ndocs"],
                "subset_chars": results[split]["subset_chars"],
                "subset_tokens": results[split]["subset_tokens"],
            },
            details_fname,
        )
