import init_path
import numpy as np
import utils.misc_utils as mscu
from tqdm import tqdm
from utils.global_variables import GB

if __name__ == "__main__":
    log_folder = f"{GB['log_folder']}/count_tokens"
    mscu.make_folder(log_folder)
    logger = mscu.get_logger(
        add_console=True, filename=f"{log_folder}/count_tokens.log"
    )

    token_folder = f"{GB['scratch_folder']}/data-mixing/slimpajama/collected_10k"
    logger.info(f"Token folder: {token_folder}")

    for split in ["train", "validation"]:
        for subset in GB["slimpajama_subsets"]:
            chunk_sizes = np.load(
                f"{token_folder}/{split}/{subset}/_new_chunk_sizes.npy"
            )
            token_counts = np.ones(len(chunk_sizes)) * -1
            token_counts_sum = 0
            token_counts_with_padding = 0

            report_every = np.round((chunk_sizes.shape[0] / 10000)) * 1000

            for i, chunk_size in tqdm(
                enumerate(chunk_sizes),
                desc=f"Processing {subset} {split}",
                total=len(chunk_sizes),
            ):
                iname = f"{i:06d}"
                mask_f = (
                    f"{token_folder}/{split}/{subset}/chunk={iname}_attention_mask.npy"
                )
                mask = np.load(mask_f)
                token_counts[i] = np.sum(mask)
                token_counts_sum += token_counts[i]
                token_counts_with_padding += len(mask)

                if i % report_every == 0 and i > 0:
                    logger.info(
                        f"{split} {subset}: "
                        f"Processed {i} chunks. Token percentage: {token_counts_sum / token_counts_with_padding * 100:.4f}%"
                    )

            token_counts_fname = f"{token_folder}/{split}/{subset}/_token_counts.npy"
            np.save(token_counts_fname, token_counts)
            logger.info(f"Saved token counts to {token_counts_fname}")
            logger.info(
                f"{split} {subset}: "
                f"Final Token percentage: {token_counts_sum / token_counts_with_padding * 100:.4f}%"
            )
