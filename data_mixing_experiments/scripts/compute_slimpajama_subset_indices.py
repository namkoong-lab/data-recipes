"""This scripts computes the indices for different subsets of Slimpajama and save it to a file.
The save files then allow fast access to subsets of the dataset.

The pathing is
    {scratch_folder} / slimpajama_subset_indices
    / {dataset_name} / {split} / {subset_name}.npy
"""

import argparse

import init_path
import numpy as np
import utils.data_utils as datu
import utils.misc_utils as mscu
from datasets import load_dataset
from utils.global_variables import GB


def get_args():
    parser = argparse.ArgumentParser(description="Compute SlimPajama subset indices")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="DKYoon/SlimPajama-6B",  # smaller 6B dataset for testing
        choices=["cerebras/SlimPajama-627B", "DKYoon/SlimPajama-6B"],
    )

    return parser.parse_args()


def main(args, logger):
    ds = load_dataset(args.dataset_name, cache_dir=GB["dataset_path"])

    for split in ds.keys():
        logger.info(f"Processing {split}")
        folder = datu.get_slimpajama_subset_indices_folder(args.dataset_name, split)
        mscu.make_folder(folder)
        if mscu.is_completed(folder):
            logger.info(
                f"The split {split} of dataset {args.dataset_name} has been processed. Please double check if you want to recompute."
            )
            continue

        subsets = datu.compute_slimpajama_subdatasets(ds[split])
        for subset_name, subset_indices in subsets.items():
            mscu.make_folder(folder)
            filename = f"{folder}/{subset_name}.npy"

            logger.info(
                f"Saving {subset_name} of size {len(subset_indices)} to {filename}"
            )
            np.save(filename, arr=subset_indices)
        mscu.note_completed(folder)


if __name__ == "__main__":
    args = get_args()
    log_folder = (
        f"{GB['scratch_folder']}/slimpajama_subset_indices/{args.dataset_name}/"
    )
    mscu.make_folder(log_folder)
    logger = mscu.get_logger(
        add_console=True, filename=f"{log_folder}/compute_slimpajama_subset_indices.log"
    )
    mscu.display_args(args, logger)

    main(args, logger)
