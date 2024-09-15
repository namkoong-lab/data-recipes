import argparse
import time
import traceback

import init_path
import utils.misc_utils as mscu
from datasets import load_dataset
from huggingface_hub import login
from utils.global_variables import ANON_GB, GB

SNAPSHOTS = [
    "2014-15",
    "2014-23",
    "2014-35",
    "2014-41",
    "2014-42",
    "2014-49",
    "2014-52",
    "2015-14",
    "2015-22",
    "2015-27",
    "2015-32",
    "2015-35",
    "2015-40",
    "2015-48",
    "2016-07",
    "2016-18",
    "2016-22",
    "2016-26",
    "2016-30",
    "2016-36",
    "2016-40",
    "2016-44",
    "2016-50",
    "2017-04",
    "2017-09",
    "2017-17",
    "2017-22",
    "2017-26",
    "2017-30",
    "2017-34",
    "2017-39",
    "2017-43",
    "2017-47",
    "2017-51",
    "2018-05",
    "2018-09",
    "2018-13",
    "2018-17",
    "2018-22",
    "2018-26",
    "2018-30",
    "2018-34",
    "2018-39",
    "2018-43",
    "2018-47",
    "2018-51",
    "2019-04",
    "2019-09",
    "2019-13",
    "2019-18",
    "2019-22",
    "2019-26",
    "2019-30",
    "2019-35",
    "2019-39",
    "2019-43",
    "2019-47",
    "2019-51",
    "2020-05",
    "2020-10",
    "2020-16",
    "2020-24",
    "2020-29",
    "2020-34",
    "2020-40",
    "2020-45",
    "2020-50",
    "2021-04",
    "2021-10",
    "2021-17",
    "2021-21",
    "2021-25",
    "2021-31",
    "2021-39",
    "2021-43",
    "2021-49",
    "2022-05",
    "2022-21",
    "2022-27",
    "2022-33",
    "2022-40",
    "2022-49",
    "2023-06",
    "2023-14",
]


def get_args():
    parser = argparse.ArgumentParser(description="Download RedPajama dataset")
    parser.add_argument("--simple_load", type=mscu.boolean, default=False)
    return parser.parse_args()


def main(args, logger):
    login(token=ANON_GB["hf_token"])  # To avoid rate limiting during download

    if args.simple_load:
        ds = load_dataset(
            "togethercomputer/RedPajama-Data-1T",
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
                    "togethercomputer/RedPajama-Data-1T",
                    "default",
                    cache_dir=GB["dataset_path"],
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
