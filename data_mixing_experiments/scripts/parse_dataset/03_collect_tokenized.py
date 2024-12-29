"""This scripts collect the tokenized dataset and repartition the data into equal-sized chunks.
The chunks are in the following format
    - input_ids: np.ndarray of type uint16
    - attention_mask: np.ndarray of type bool
    - overflow_to_sample_mapping: np.ndarray of type uint32
"""

import argparse
import ctypes
import json
from multiprocessing import Process, RawArray

import init_path
import numpy as np
import utils.misc_utils as mscu
from tqdm import tqdm
from utils.global_variables import GB


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence_per_chunk",
        "-s",
        type=int,
        default=10000,  # 10K sequences per chunk
        help="Number of sequences per chunk.",
    )
    parser.add_argument(
        "--do_train",
        type=mscu.boolean,
        default=True,
        help="Whether to process the train dataset.",
    )
    parser.add_argument(
        "--do_validation",
        type=mscu.boolean,
        default=True,
        help="Whether to process the validation dataset.",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=GB["context_length"],
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=5,
        help="Number of CPUs to use.",
    )

    return parser.parse_args()


def init_chunk(chunk_size, context_length):
    """Initialize a chunk with the given sequence size."""
    new_id = np.ones((chunk_size, context_length), dtype=np.uint16) * -1
    new_attention_mask = np.zeros((chunk_size, context_length), dtype=np.bool_)
    new_overflow_to_sample_mapping = np.ones(chunk_size, dtype=np.uint32) * -1

    return new_id, new_attention_mask, new_overflow_to_sample_mapping


def save_new_chunks(
    buffer_ids,
    buffer_masks,
    buffer_maps,
    left_chunk_idx,
    right_chunk_idx,
    chunk_seq_counts,
    new_token_folder,
    sequence_per_chunk,
):
    for i, new_chunk_idx in enumerate(range(left_chunk_idx, right_chunk_idx)):
        left_seq_idx = i * sequence_per_chunk
        right_seq_idx = min((i + 1) * sequence_per_chunk, buffer_ids.shape[0])
        chunk_seq_counts[new_chunk_idx] = right_seq_idx - left_seq_idx

        # Save the new chunk
        new_chunk_iname = f"{new_chunk_idx:06d}"
        new_ids_fname = f"{new_token_folder}/chunk={new_chunk_iname}_input_ids.npy"
        new_mask_fname = (
            f"{new_token_folder}/chunk={new_chunk_iname}_attention_mask.npy"
        )
        new_map_fname = (
            f"{new_token_folder}/chunk={new_chunk_iname}_overflow_to_sample_mapping.npy"
        )

        np.save(new_ids_fname, buffer_ids[left_seq_idx:right_seq_idx, :].reshape(-1))
        np.save(new_mask_fname, buffer_masks[left_seq_idx:right_seq_idx, :].reshape(-1))
        np.save(new_map_fname, buffer_maps[left_seq_idx:right_seq_idx])


def read_buffer(start_idx, end_idx, token_folder):
    buffer_ids = np.zeros((0, GB["context_length"]), dtype=np.uint16)
    buffer_masks = np.zeros((0, GB["context_length"]), dtype=np.bool_)
    buffer_maps = np.zeros((0,), dtype=np.uint32)

    pbar = tqdm(
        total=end_idx - start_idx,
        position=3,
        desc=f"Reading buffer {start_idx}-{end_idx}",
        leave=False,
    )
    for i in range(start_idx, end_idx):
        old_chunk_iname = f"{i:06d}"
        id_fname = f"{token_folder}/chunk={old_chunk_iname}_input_ids.npy"
        mask_fname = f"{token_folder}/chunk={old_chunk_iname}_attention_mask.npy"
        map_fname = (
            f"{token_folder}/chunk={old_chunk_iname}_overflow_to_sample_mapping.npy"
        )

        id_chunk = np.load(id_fname)
        mask_chunk = np.load(mask_fname)
        map_chunk = np.load(map_fname)

        buffer_ids = np.concatenate((buffer_ids, id_chunk))
        buffer_masks = np.concatenate((buffer_masks, mask_chunk))
        buffer_maps = np.concatenate((buffer_maps, map_chunk))
        pbar.update(1)
    pbar.close()

    return buffer_ids, buffer_masks, buffer_maps


def process_subset(
    token_folder, new_token_folder, sequence_per_chunk, num_cpus, logger
):
    """Load chunks of varying sequence sizes and repartition them into equal-sized chunks."""
    chunk_sequence_counts_fname = f"{token_folder}/_chunk_seq_counts.npy"
    chunk_sequence_counts = np.load(chunk_sequence_counts_fname)

    n_chunks = len(chunk_sequence_counts)
    n_sequences = np.sum(chunk_sequence_counts)

    logger.info(f"Number of chunks: {n_chunks}")
    logger.info(f"Number of sequences: {n_sequences}")

    n_new_chunks = n_sequences // sequence_per_chunk
    if n_sequences % sequence_per_chunk != 0:
        n_new_chunks += 1
    logger.info(f"Number of new chunks: {n_new_chunks}")

    buffer_ids = np.zeros((0, GB["context_length"]), dtype=np.uint16)
    buffer_masks = np.zeros((0, GB["context_length"]), dtype=np.bool_)
    buffer_maps = np.zeros((0,), dtype=np.uint32)

    # Read X chunks at a time into buffer. Spin up multiple process to save new chunks.
    chunk_per_buffer = num_cpus
    num_buffer_iteration = n_chunks // chunk_per_buffer
    if n_chunks % chunk_per_buffer != 0:
        num_buffer_iteration += 1

    new_chunks_processed = 0
    new_chunks_seq_counts = RawArray(ctypes.c_int, [-1] * n_new_chunks)
    new_chunks_pbar = tqdm(
        total=n_new_chunks,
        desc="Processing new chunks",
        position=2,
        leave=True,
    )
    old_chunks_pbar = tqdm(
        total=n_chunks,
        desc="Processing old chunks",
        position=1,
        leave=True,
    )
    for buffer_idx in tqdm(
        range(num_buffer_iteration),
        desc=f"Processing buffer size {chunk_per_buffer}",
        position=0,
    ):
        # Read chunks into buffer
        left_idx = buffer_idx * chunk_per_buffer
        right_idx = min((buffer_idx + 1) * chunk_per_buffer, n_chunks)
        new_buffer_ids, new_buffer_masks, new_buffer_maps = read_buffer(
            left_idx, right_idx, token_folder
        )
        old_chunks_pbar.update(right_idx - left_idx)

        buffer_ids = np.concatenate((buffer_ids, new_buffer_ids))
        buffer_masks = np.concatenate((buffer_masks, new_buffer_masks))
        buffer_maps = np.concatenate((buffer_maps, new_buffer_maps))

        # Spin up multiple processes to save new chunks.
        cur_num_new_chunks = buffer_ids.shape[0] // sequence_per_chunk
        if (
            buffer_idx == num_buffer_iteration - 1
            and buffer_ids.shape[0] % sequence_per_chunk != 0
        ):
            cur_num_new_chunks += 1

        new_chunks_per_cpu = cur_num_new_chunks // num_cpus
        if cur_num_new_chunks % num_cpus != 0:
            new_chunks_per_cpu += 1

        # Schedule new chunks indices to be processed by each CPU
        new_chunk_indices = mscu.distribute_jobs(cur_num_new_chunks, num_cpus)
        processes = []
        for process_idx in range(num_cpus):
            left_chunk_idx, right_chunk_idx = new_chunk_indices[process_idx]
            left_seq_idx = left_chunk_idx * sequence_per_chunk
            right_seq_idx = min(
                right_chunk_idx * sequence_per_chunk, buffer_ids.shape[0]
            )
            p = Process(
                target=save_new_chunks,
                kwargs={
                    "buffer_ids": buffer_ids[left_seq_idx:right_seq_idx, :],
                    "buffer_masks": buffer_masks[left_seq_idx:right_seq_idx, :],
                    "buffer_maps": buffer_maps[left_seq_idx:right_seq_idx],
                    "left_chunk_idx": new_chunks_processed + left_chunk_idx,
                    "right_chunk_idx": new_chunks_processed + right_chunk_idx,
                    "chunk_seq_counts": new_chunks_seq_counts,
                    "new_token_folder": new_token_folder,
                    "sequence_per_chunk": sequence_per_chunk,
                },
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Update buffer
        buffer_ids = buffer_ids[right_seq_idx:, :]
        buffer_masks = buffer_masks[right_seq_idx:, :]
        buffer_maps = buffer_maps[right_seq_idx:]

        new_chunks_processed += cur_num_new_chunks
        new_chunks_pbar.update(cur_num_new_chunks)

        # Save the new chunk sizes.
        np.save(
            f"{new_token_folder}/_new_chunk_sizes.npy", np.array(new_chunks_seq_counts)
        )

    new_chunks_pbar.close()
    old_chunks_pbar.close()

    seq_count = np.sum(new_chunks_seq_counts)
    return seq_count


def main(args, logger):
    splits = []
    if args.do_train:
        splits.append("train")
    if args.do_validation:
        splits.append("validation")

    old_token_folder = f"{GB['scratch_folder']}/data-mixing/slimpajama/tokenize_all"
    new_token_folder = f"{GB['scratch_folder']}/data-mixing/slimpajama/collected_10k"

    seq_counts_fname = f"{old_token_folder}/tokenize_counts.json"
    seq_counts = json.load(open(seq_counts_fname, "r"))
    new_seq_count_fname = f"{new_token_folder}/tokenize_counts.json"
    mscu.make_folder(new_token_folder)

    with open(f"{new_token_folder}/_script_parameters.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    for split in splits:
        for subset in GB["slimpajama_subsets"]:
            logger.info(f"Processing {split} {subset}")
            ss_token_folder = f"{old_token_folder}/{split}/{subset}"
            ss_new_token_folder = f"{new_token_folder}/{split}/{subset}"
            mscu.make_folder(ss_new_token_folder)
            seq_count = process_subset(
                ss_token_folder,
                ss_new_token_folder,
                args.sequence_per_chunk,
                args.num_cpus,
                logger,
            )

            assert seq_count == seq_counts[split][subset]["total_seqs"]
            seq_counts[split][subset]["new_chunk_size"] = args.sequence_per_chunk

            with open(new_seq_count_fname, "w") as f:
                json.dump(seq_counts, f, indent=2)
            logger.info(f"Processing {split} {subset} done.")


if __name__ == "__main__":
    log_folder = f"{GB['log_folder']}/collect_tokenized"
    mscu.make_folder(log_folder)
    logger = mscu.get_logger(
        add_console=True, filename=f"{log_folder}/collect_tokenized_10k.log"
    )

    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")
    main(args, logger)
