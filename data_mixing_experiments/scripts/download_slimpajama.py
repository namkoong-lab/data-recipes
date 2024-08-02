import argparse
import time
import traceback

import init_path
from datasets import load_dataset
from huggingface_hub import login

import utils.misc_utils as mscu
from utils.global_variables import ANON_GB, GB


def get_args():
    parser = argparse.ArgumentParser(description="Download SlimPajama dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="DKYoon/SlimPajama-6B",
        choices=["cerebras/SlimPajama-627B", "DKYoon/SlimPajama-6B"],
    )
    parser.add_argument("--simple_load", type=mscu.boolean, default=False)

    return parser.parse_args()


def main(args, logger):
    login(token=ANON_GB["hf_token"])  # To avoid rate limiting during download

    if args.simple_load:
        ds = load_dataset(args.dataset_name, cache_dir=GB["dataset_path"])
        import pdb; pdb.set_trace()  # fmt: skip
    else:
        WAIT_TIME = 30
        COUNT_DOWN_INTERVAL = 2

        ds = None
        while ds is None:
            try:
                ds = load_dataset(args.dataset_name, cache_dir=GB["dataset_path"])
            except Exception as e:
                logger.info(f"Traceback: {traceback.format_exc()}")
                logger.info(f"Waiting for {WAIT_TIME} seconds before retrying...")
                for i in range(WAIT_TIME, 0, -COUNT_DOWN_INTERVAL):
                    logger.info(f"Retrying in {i} seconds...", end="\r")
                    time.sleep(COUNT_DOWN_INTERVAL)
                logger.info(f"Retrying now...")
        import pdb; pdb.set_trace()  # fmt: skip


if __name__ == "__main__":
    args = get_args()
    logger = mscu.get_logger(add_console=True)
    logger.info("Hi")
    import pdb; pdb.set_trace()  # fmt: skip
    mscu.display_args(args, logger)

    main(args, logger)
