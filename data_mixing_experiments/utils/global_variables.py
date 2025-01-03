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
    "SlimPajamaPaddingPercentage": {
        "RedPajamaCommonCrawl": 0.1775,
        "RedPajamaC4": 0.5579,
        "RedPajamaWikipedia": 0.3074,
        "RedPajamaStackExchange": 0.3579,
        "RedPajamaGithub": 0.1763,
        "RedPajamaArXiv": 0.0139,
        "RedPajamaBook": 0.0023,
    },
    "training_params": {
        "1B": {
            "d_model": 2048,
            "n_heads": 16,
            "n_layers": 16,
            "mlp_ratio": 8,
            "num_documents": 20_000,
            "global_train_batch_size": 1024,
            "device_train_microbatch_size": 18,
            "device_eval_batch_size": 60,
        },
        "700M": {
            "d_model": 1536,
            "n_heads": 16,
            "n_layers": 16,
            "mlp_ratio": 8,
            "num_documents": 10_000,
            "global_train_batch_size": 512,
            "device_train_microbatch_size": 28,
            "device_eval_batch_size": 72,
        },
        "500M": {
            "d_model": 1280,
            "n_heads": 16,
            "n_layers": 16,
            "mlp_ratio": 8,
            "num_documents": 10_000,
            "global_train_batch_size": 512,
            "device_train_microbatch_size": 32,
            "device_eval_batch_size": 80,
        },
        "300M": {
            "d_model": 1024,
            "n_heads": 16,
            "n_layers": 16,
            "mlp_ratio": 8,
            "num_documents": 10_000,
            "global_train_batch_size": 512,
            "device_train_microbatch_size": 32,
            "device_eval_batch_size": 80,
        },
        "150M": {
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "mlp_ratio": 8,
            "num_documents": 10_000,
        },
        "60M": {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 8,
            "mlp_ratio": 8,
            "num_documents": 10_000,
        },
        "20M": {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 8,
            "mlp_ratio": 8,
            "num_documents": 10_000,
            "global_train_batch_size": 512,
            "device_train_microbatch_size": 64,
            "device_eval_batch_size": 128,
        },
    },
}

# Global Constants that should not be printed or shared
ANON_GB = {
    "hf_token": "your_huggingface_token_here",
}

print(f"*" * 84)
print(f"Accessing Global Variables:\n{json.dumps(GB, indent=2)}")
print(f"*" * 84)
print(f"*" * 84)
print(f"*" * 84)
print(f"*" * 84)
