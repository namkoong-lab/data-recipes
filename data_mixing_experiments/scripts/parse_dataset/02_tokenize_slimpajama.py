"""This scripts tokenize part of the slimpajama dataset and save the tokenized dataset.
Parameters:
    - num_cpus: Number of cpus to use for tokenization.

"""

import argparse
import ctypes
import json
import os
from multiprocessing import Process, RawArray

import init_path
import numpy as np
import torch
import utils.data_utils as datu
import utils.misc_utils as mscu
from datasets import load_dataset
from tqdm import tqdm
from utils.global_variables import GB

os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # to avoid warning parallelized tokenizer warning
)
TOKENIZE_RUN_NAME = "tokenize_all"
SCRATCH_FOLDER = f"{GB['scratch_folder']}/data-mixing/slimpajama/{TOKENIZE_RUN_NAME}"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=96,
        help="Number of cpus to use for tokenization.",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=GB["context_length"],
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride for tokenization.",
    )

    return parser.parse_args()


def tokenize_chunks(
    split_ds,
    chosen_idc,
    start,
    end,
    chunk_idx,
    chunk_seq_counts,
    chunk_folder,
    process_index,
    context_length,
    stride,
    tokenizer,
):
    pbar = tqdm(
        total=end - start,
        position=process_index + 1,
        desc=f"Process {process_index}: {start}-{end}",
        leave=False,
    )

    result = {
        "input_ids": [],
        "attention_mask": [],
        "overflow_to_sample_mapping": [],
    }
    for i in range(start, end):
        cur_index = int(chosen_idc[i])
        doc = split_ds[cur_index]
        tokens = tokenizer(
            doc["text"],
            return_tensors="pt",
            padding="max_length",
            stride=stride,
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
        )
        input_ids = tokens["input_ids"].to(torch.uint32)
        attention_mask = tokens["attention_mask"].to(torch.bool)
        overflow_to_sample_mapping = tokens["overflow_to_sample_mapping"]
        overflow_to_sample_mapping[:] = cur_index  # set correct document indices

        result["input_ids"].append(input_ids)
        result["attention_mask"].append(attention_mask)
        result["overflow_to_sample_mapping"].append(overflow_to_sample_mapping)
        pbar.update(1)
    pbar.close()

    result["input_ids"] = torch.cat(result["input_ids"], dim=0)
    result["attention_mask"] = torch.cat(result["attention_mask"], dim=0)
    result["overflow_to_sample_mapping"] = torch.cat(
        result["overflow_to_sample_mapping"], dim=0
    )
    chunk_seq_counts[chunk_idx] = result["input_ids"].shape[0]

    i_name = f"{chunk_idx:06d}"
    input_ids_fname = f"{chunk_folder}/chunk={i_name}_input_ids.npy"
    attention_mask_fname = f"{chunk_folder}/chunk={i_name}_attention_mask.npy"
    overflow_to_sample_mapping_fname = (
        f"{chunk_folder}/chunk={i_name}_overflow_to_sample_mapping.npy"
    )
    result["input_ids"] = result["input_ids"].numpy().astype(np.uint16)
    result["attention_mask"] = result["attention_mask"].numpy().astype(np.bool_)
    result["overflow_to_sample_mapping"] = (
        result["overflow_to_sample_mapping"].numpy().astype(np.uint32)
    )

    np.save(input_ids_fname, result["input_ids"])
    np.save(attention_mask_fname, result["attention_mask"])
    np.save(overflow_to_sample_mapping_fname, result["overflow_to_sample_mapping"])


def tokenize_subset(split_ds, split, subset_name, subset_idc, num_cpus, args, logger):
    logger.info(f"Tokenizing {subset_name} for {split} split.")
    scratch_folder = f"{SCRATCH_FOLDER}/{split}/{subset_name}"
    mscu.make_folder(scratch_folder)

    doc_count = GB["slimpajama_trim"][split]["doc_count"][subset_name]
    if doc_count is None:
        doc_count = len(subset_idc)
    if doc_count > len(subset_idc):
        doc_count = len(subset_idc)

    logger.info(
        f"Subsampling {doc_count} documents "
        f"from a total of {len(subset_idc)} documents for {subset_name}."
        f" {doc_count / len(subset_idc) * 100:.2f}% of the data."
    )
    chosen_idc = subset_idc[:doc_count]

    chosen_idc_fname = f"{scratch_folder}/tokenize_chosen_index.npy"
    logger.info(f"Saving subsampled index to scratch: {chosen_idc_fname}")
    np.save(chosen_idc_fname, chosen_idc)

    tokenizer = datu.get_tokenizer()

    # Prepare doc batches for tokenization
    if subset_name in (
        "RedPajamaCommonCrawl",
        "RedPajamaC4",
        "RedPajamaWikipedia",
        "RedPajamaStackExchange",
    ):
        ndoc_per_cpu = 50000  # 50k
    elif subset_name == "RedPajamaGithub":
        ndoc_per_cpu = 25000
    elif subset_name == "RedPajamaArXiv":
        ndoc_per_cpu = 600
    elif subset_name == "RedPajamaBook":
        ndoc_per_cpu = 100
    else:
        raise ValueError(f"Unknown subset: {subset_name}")

    logger.info(
        f"Tokenizing {subset_name} for {split} split with chunk size {ndoc_per_cpu}."
    )
    ranges = [
        (i, min(i + ndoc_per_cpu, doc_count)) for i in range(0, doc_count, ndoc_per_cpu)
    ]
    chunk_seq_counts = RawArray(ctypes.c_int, [-1] * len(ranges))

    # Separate batches of documents for each cpus to tokenize
    pbar = tqdm(total=len(ranges), position=0, desc="Tokenizing chunks", leave=True)
    chunk_processed = 0
    while chunk_processed < len(ranges):
        processes = []
        for i in range(num_cpus):
            if chunk_processed >= len(ranges):
                break

            p = Process(
                target=tokenize_chunks,
                kwargs={
                    "split_ds": split_ds,
                    "chosen_idc": chosen_idc,
                    "start": ranges[chunk_processed][0],
                    "end": ranges[chunk_processed][1],
                    "chunk_idx": chunk_processed,
                    "chunk_seq_counts": chunk_seq_counts,
                    "chunk_folder": scratch_folder,
                    "process_index": i,
                    "context_length": args.context_length,
                    "stride": args.stride,
                    "tokenizer": tokenizer,
                },
            )
            p.start()
            processes.append(p)
            chunk_processed += 1

        for p in processes:
            p.join()
        pbar.update(len(processes))
    pbar.close()

    total_seqs = sum(chunk_seq_counts)
    logger.info(f"Total number of sequences: {total_seqs:,d}")

    # Save the number of sequences in each chunk
    chunk_seq_counts_fname = f"{scratch_folder}/_chunk_seq_counts.npy"
    np.save(chunk_seq_counts_fname, np.array(chunk_seq_counts))

    return doc_count, total_seqs


def main():
    log_folder = f"{GB['log_folder']}/tokenize_slimpajama"
    mscu.make_folder(log_folder)
    logger = mscu.get_logger(
        add_console=True, filename=f"{log_folder}/{TOKENIZE_RUN_NAME}.log"
    )

    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")

    # Script parameters
    num_cpus = args.num_cpus
    splits = ["train", "validation"]
    # splits = ["validation"]

    logger.info(
        f"Tokenizing based on the parameters:"
        f" {json.dumps(GB['slimpajama_trim'], indent=2)}"
        f"\nNumber of cpus: {num_cpus}"
        f"\nSplits: {splits}"
    )

    # Loading subset indices
    try:
        train_idc = np.load(
            f"{GB['scratch_folder']}/data-mixing/slimpajama/train/index.npy",
            allow_pickle=True,
        ).item()
        validation_idc = np.load(
            f"{GB['scratch_folder']}/data-mixing/slimpajama/validation/index.npy",
            allow_pickle=True,
        ).item()
    except Exception as e:
        logger.error(
            f"Failed to load index file: {e}. Have you ran 01_index_slimpajama.py?"
        )
        raise

    all_split_idc = {
        "train": train_idc,
        "validation": validation_idc,
    }

    ds = load_dataset("cerebras/SlimPajama-627B", cache_dir=GB["dataset_path"])

    # Save info on the number of sequences and number of documents
    counts = {}  # split
    counts_fname = f"{SCRATCH_FOLDER}/tokenize_counts.json"
    if os.path.exists(counts_fname):
        with open(counts_fname, "r") as f:
            counts = json.load(f)

    for split in splits:
        counts[split] = {}
        for subset in GB["slimpajama_subsets"]:
            doc_count, total_seqs = tokenize_subset(
                ds[split],
                split,
                subset,
                all_split_idc[split][subset],
                num_cpus=num_cpus,
                logger=logger,
                args=args,
            )
            counts[split][subset] = {"doc_count": doc_count, "total_seqs": total_seqs}

            # Save the counts
            with open(counts_fname, "w") as f:
                json.dump(counts, f, indent=2)

    # Save the counts
    with open(counts_fname, "w") as f:
        json.dump(counts, f, indent=2)

    logger.info("Tokenization is done.")


if __name__ == "__main__":
    main()
