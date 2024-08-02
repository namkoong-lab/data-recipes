"""Save slimpajama's dataset in jsonl format for DCLM.
"""

import argparse
import json

import init_path

import utils.data_utils as datu
import utils.misc_utils as mscu
from utils.global_variables import GB


def get_args():
    parser = argparse.ArgumentParser(
        description="Save SlimPajama dataset in jsonl format"
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        choices=GB["slimpajama_subsets"],
        help="Subset of the SlimPajama dataset to save",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        help="Number of samples to save",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="DKYoon/SlimPajama-6B",  # smaller 6B dataset for testing
        choices=["cerebras/SlimPajama-627B", "DKYoon/SlimPajama-6B"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for subsampling",
    )

    return parser.parse_args()


def main(args, logger):
    folder = f"{GB['scratch_folder']}/slimpajama_subsampled_jsonl/{args.dataset_name}/{args.split}/{args.subset_name}/{args.n_samples}/{args.seed}"
    mscu.make_folder(folder)

    if mscu.is_completed(folder):
        logger.warning(
            f"The subsampled dataset {args.dataset_name} {args.split} {args.subset_name} {args.n_samples} {args.seed} has been saved. Please double check if you want to recompute."
        )
        return

    ds = datu.load_dataset(args.dataset_name, cache_dir=GB["dataset_path"])[args.split]

    subdataset = datu.get_slimpajama_subdataset(
        subset_name=args.subset_name,
        dataset=ds,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    logger.info(
        f"Dataset {args.dataset_name} Subdataset {args.subset_name}"
        f" with split {args.split} has {len(subdataset)} samples."
        f"\n Subsampling {args.n_samples} samples with seed {args.seed}"
    )

    subsampled_subdataset = datu.subsample_dataset(
        subdataset, n_samples=args.n_samples, seed=args.seed
    )

    # Save subsampled dataset
    filename = f"{folder}/data.jsonl"
    logger.info(f"Saving subsampled dataset to {filename}")
    with open(filename, "w") as f:
        for idx in range(len(subsampled_subdataset)):
            # Keys requested in https://github.com/mlfoundations/dclm/blob/main/baselines/core/constants.py. Keeping chunk 1 and url empty for now.
            item = {"text": subsampled_subdataset[idx]["text"], "chunk": 1, "url": ""}
            json.dump(item, f)
            f.write("\n")

    mscu.note_completed(folder)


if __name__ == "__main__":
    args = get_args()
    log_folder = f"{init_path.PROJ_ROOT}/logs/save_slimpajama_jsonl/{args.dataset_name}/{args.split}"
    mscu.make_folder(log_folder)

    logger = mscu.get_logger(
        add_console=True, filename=f"{log_folder}/save_slimpajama_jsonl.log"
    )
    mscu.display_args(args, logger)

    main(args, logger)
