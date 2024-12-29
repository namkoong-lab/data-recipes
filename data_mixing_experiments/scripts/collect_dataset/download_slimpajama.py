from datasets import load_dataset
import argparse
import init_path
import utils.misc_utils as mscu
from huggingface_hub import login
from utils.global_variables import ANON_GB, GB
import time
import traceback


def get_args():
    parser = argparse.ArgumentParser(description="Download RedPajama dataset")
    parser.add_argument("--simple_load", type=mscu.boolean, default=False)
    return parser.parse_args()


def main(args, logger):
    login(token=ANON_GB["hf_token"])  # To avoid rate limiting during download

    if args.simple_load:
        ds = load_dataset(
            "cerebras/SlimPajama-627B",
            cache_dir=GB["dataset_path"],
        )
        import pdb; pdb.set_trace()  # fmt: skip
    else:
        WAIT_TIME = 30
        COUNT_DOWN_INTERVAL = 2

        ds = None
        while ds is None:
            try:
                ds = load_dataset(
                    "cerebras/SlimPajama-627B", "default", cache_dir=GB["dataset_path"]
                )
                # ds = load_dataset(
                #     "togethercomputer/RedPajama-Data-V2",
                #     name="sample",
                #     cache_dir=GB["dataset_path"],
                # )
            except Exception as e:
                logger.info(f"Traceback: {traceback.format_exc()}")
                logger.info(f"Waiting for {WAIT_TIME} seconds before retrying...")
                for i in range(WAIT_TIME, 0, -COUNT_DOWN_INTERVAL):
                    logger.info(f"Retrying in {i} seconds...")
                    time.sleep(COUNT_DOWN_INTERVAL)
                logger.info(f"Retrying now...")
        import pdb; pdb.set_trace()  # fmt: skip


if __name__ == "__main__":
    args = get_args()
    log_folder = f"{init_path.PROJ_ROOT}/scratch/redpajama_download_logs"
    mscu.make_folder(log_folder)
    log_filename = f"{log_folder}/download_redpajama.log"
    logger = mscu.get_logger(add_console=True, filename=log_filename)

    mscu.display_args(args, logger)

    main(args, logger)
