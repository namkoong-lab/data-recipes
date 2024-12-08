import json

GB = {
    "dataset_path": "/mnt/home/tyen/datasets",
    "scratch_folder": "/mnt/home/tyen/scratch",
    "log_folder": "/mnt/home/tyen/data-recipes/data_mixing_experiments/logs",
    "ray_tmp_folder": "/mnt/home/tyen/ray_scratch",
    # "dataset_path": "/shared/share_mala/tyen/huggingface_datasets/downloads",
    # "dataset_path": "/scratch/ty2531/huggingface_datasets/downloads",
    ## Experiment Specific
    "context_length": 1024,
    "slimpajama_subsets": [
        "RedPajamaCommonCrawl",
        "RedPajamaC4",
        "RedPajamaWikipedia",
        "RedPajamaStackExchange",
        "RedPajamaGithub",
        "RedPajamaArXiv",
        "RedPajamaBook",
    ],
    "slimpajama_trim": {
        "batch_size": 100_000,
        # Set number of documents chosen (aiming to have 20B tokens per category)
        "train": {
            "doc_count": {
                "RedPajamaCommonCrawl": None,
                "RedPajamaC4": None,
                "RedPajamaWikipedia": None,
                "RedPajamaStackExchange": None,
                "RedPajamaGithub": None,
                "RedPajamaArXiv": None,
                "RedPajamaBook": None,
            },
        },
        # Taking 1% of the training set for validation
        "validation": {
            "doc_count": {
                "RedPajamaCommonCrawl": None,
                "RedPajamaC4": None,
                "RedPajamaWikipedia": None,
                "RedPajamaStackExchange": None,
                "RedPajamaGithub": None,
                "RedPajamaArXiv": None,
                "RedPajamaBook": None,
            },
        },
    },
    "model_params": {
        "1B": {
            "gpt_n_layer": 16,
            "gpt_n_head": 16,
            "gpt_n_embd": 2048,
        },  # Actually 0.9B
        "100M": {
            "gpt_n_layer": 8,
            "gpt_n_head": 8,
            "gpt_n_embd": 1024,
        },  # Actually 153M
        "10M": {
            "gpt_n_layer": 4,
            "gpt_n_head": 4,
            "gpt_n_embd": 512,
        },  # Actually 39M
    },
}

# Global Constants that should not be printed or shared
ANON_GB = {
    "hf_token": "your_huggingface_token_here",
}

print(f"*" * 84)
print(f"Accessing Global Variables:\n{json.dumps(GB, indent=2)}")
print(f"*" * 84)
