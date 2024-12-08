from typing import Dict, List

import numpy as np
import utils.misc_utils as mscu
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.global_variables import GB


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        x = self.dataset[self.indices[idx]]

        if self.transform is not None:
            transformed_item = self.transform(x)
            return transformed_item
        else:
            return x

    def __len__(self):
        return len(self.indices)


def get_slimpajama_subset_indices_folder(dataset_name: str, split: str) -> str:
    return f"{GB['scratch_folder']}/slimpajama_subset_indices/{dataset_name}/{split}"


def compute_slimpajama_subdatasets(dataset: Dataset) -> Dict[str, List[int]]:
    """Compute the indices of subdatasets of SlimPajama dataset.

    Assuming the data in datasets has the following data structure:
    {
        "text": [str],
        "meta": {"redpajama_set_name": [str], *:*}
    }
    Return a dictionary of the subsets as indicated by redpajama_set_name: {
        redpajama_set_name: [Subsets]
    }
    """
    # Get the map of redpajama_set_name to corresponding data's indices
    subsets = {}
    for idx in tqdm(range(len(dataset)), desc="Getting Subsets"):
        meta = dataset[idx]["meta"]
        redpajama_set_name = meta["redpajama_set_name"]
        if redpajama_set_name not in subsets:
            subsets[redpajama_set_name] = []
        subsets[redpajama_set_name].append(idx)

    return {set_name: indices for set_name, indices in subsets.items()}


def get_slimpajama_subdataset(
    subset_name,
    dataset=None,
    dataset_name="cerebras/SlimPajama-627B",
    split="train",
) -> Subset:
    """Get the subset of the dataset with the specified subset_name.
    If dataset is provided, then dataset and dataset_name must match.
    """
    if dataset is None:
        dataset = load_dataset(dataset_name, cache_dir=GB["dataset_path"])[split]
    folder = get_slimpajama_subset_indices_folder(dataset_name, split)
    subset_indices = np.load(f"{folder}/{subset_name}.npy")
    return Subset(dataset, subset_indices.tolist())


def subsample_dataset(dataset, n_samples, seed=42):
    """Subsample the dataset to n_samples."""
    mscu.set_seed(seed)
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    return Subset(dataset, indices.tolist())


def uniform_simplex_sample(n_points, n_categories, from_dirichlet=True):
    """Sample a point from the uniform simplex in n_categories dimensions.
    Following: https://math.stackexchange.com/questions/502583/uniform-sampling-of-points-on-a-simplex
    """
    if from_dirichlet:
        ps = np.random.dirichlet(np.ones(n_categories), size=n_points)
        return ps
    else:
        us = np.random.uniform(size=(n_points, n_categories))
        es = -np.log(us)
        return es / es.sum(axis=1, keepdims=True)


def get_tokenizer(tokenizer_name="GPT2TokenizerFast", model_name="gpt2"):
    """Get the tokenizer for the model."""
    if tokenizer_name == "GPT2TokenizerFast" and model_name == "gpt2":
        from transformers import GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:
        raise ValueError(
            f"Tokenizer {tokenizer_name} with model {model_name} not recognized."
        )


def get_slimpajama_sequences(subset, split, num_sequences, logger=None):
    """Get a SlimPajama dataset with the given subset and split.
    Get the first [num_sequences] sequences.
    """
    with open(
        f"{GB['scratch_folder']}/data-mixing/slimpajama/collected_tokenize/tokenize_counts.json",
        "r",
    ) as f:
        total_seq_counts = json.load(f)

    if num_sequences > total_seq_counts[split][subset]["total_seqs"]:
        mscu.lprint(
            f"Warning: Requested {num_sequences} sequences from subset {subset} split {split},"
            f" but only {total_seq_counts[split][subset]['total_seqs']} available.",
            logger,
            level="warning",
        )
        num_sequences = total_seq_counts[split][subset]["total_seqs"]
    chunk_size = total_seq_counts[split][subset]["new_chunk_size"]

    seq_folder = f"{GB['scratch_folder']}/data-mixing/slimpajama/collected_tokenize/{split}/{subset}"

    # Get first
    batch_num = num_sequences // chunk_size
    if num_sequences % chunk_size != 0:
        batch_num += 1

    all_ids = []
    all_masks = []
    all_maps = []
    for i in range(batch_num):
        ids = np.load(f"{seq_folder}/chunk={i}_input_ids.npy")
        masks = np.load(f"{seq_folder}/chunk={i}_attention_mask.npy")
        maps = np.load(f"{seq_folder}/chunk={i}_overflow_to_sample_mapping.npy")
        if i == batch_num - 1 and num_sequences % chunk_size != 0:
            ids = ids[: num_sequences % chunk_size]
            masks = masks[: num_sequences % chunk_size]
            maps = maps[: num_sequences % chunk_size]
        all_ids.append(ids)
        all_masks.append(masks)
        all_maps.append(maps)
    all_ids = np.concatenate(all_ids, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    all_maps = np.concatenate(all_maps)

    assert all_ids.shape[0] == num_sequences, "Shape mismatch"
    assert all_masks.shape[0] == num_sequences, "Shape mismatch"
    assert all_maps.shape == num_sequences, "Shape mismatch"

    return all_ids, all_masks, all_maps


class MixedSlimPajama(Dataset):
    def __init__(
        self, probs, split="train", num_sequences=100_000, verbose=True, logger=None
    ):
        self.probs = probs
        self.num_sequences = num_sequences
        self.subsets = [
            "RedPajamaCommonCrawl",
            "RedPajamaC4",
            "RedPajamaWikipedia",
            "RedPajamaStackExchange",
            "RedPajamaGithub",
            "RedPajamaArXiv",
            "RedPajamaBook",
        ]
        self.subset_sequence_counts = np.ones(len(self.subsets)) * -1
        self.input_ids = np.zeros(
            (num_sequences, GB["context_length"]), dtype=np.uint16
        )
        self.masks = np.zeros((num_sequences, GB["context_length"]), dtype=np.bool_)

        # Assign probability to each subset
        for idx in range(len(self.subsets)):
            self.subset_sequence_counts[idx] = self.num_sequences * self.probs[idx]

        self.subset_sequence_counts = mscu.round_with_same_sum(
            self.subset_sequence_counts
        )

        assert len(self.probs) == len(
            self.subsets
        ), "Probs must be the same length as subsets"
        assert np.sum(self.subset_sequence_counts) == num_sequences, "Sum mismatch"

        # Get the sequences
        enumerates = (
            enumerate(self.subsets)
            if not verbose
            else tqdm(
                enumerate(self.subsets),
                desc="Getting Sequences",
                total=len(self.subsets),
            )
        )
        self.input_ids = []
        self.masks = []
        self.maps = []
        for subset_idx, subset in enumerates:
            subset_seq_count = self.subset_sequence_counts[subset_idx]
            ids, masks, maps = get_slimpajama_sequences(
                subset, split, subset_seq_count, logger
            )
            self.input_ids.append(ids)
            self.masks.append(masks)
            self.maps.append(maps)
        self.input_ids = np.concatenate(self.input_ids, axis=0)
        self.masks = np.concatenate(self.masks, axis=0)
        self.maps = np.concatenate(self.maps)

        if num_sequences != self.input_ids.shape[0]:
            mscu.lprint(
                f"Warning: Requested {num_sequences} sequences for the entire dataset,"
                f" but only {self.input_ids.shape[0]} available.",
                logger,
                level="warning",
            )
            self.num_sequences = self.input_ids.shape[0]

        # Shuffle the sequences
        idxs = np.arange(self.num_sequences)
        np.random.shuffle(idxs)
        self.input_ids = self.input_ids[idxs]
        self.masks = self.masks[idxs]

        if verbose:
            mscu.print_numpy_memory(self.input_ids, "Dataset Input IDs", logger)
            mscu.print_numpy_memory(self.masks, "Dataset Masks", logger)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.masks[idx],
            "labels": self.input_ids[idx],
        }

    def __len__(self):
        return self.num_sequences
