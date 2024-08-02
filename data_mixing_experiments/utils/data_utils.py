from typing import Dict, List

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

import utils.misc_utils as mscu
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
    return f"{GB["scratch_folder"]}/slimpajama_subset_indices/{dataset_name}/{split}"


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
