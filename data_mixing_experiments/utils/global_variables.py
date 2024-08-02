import json

GB = {
    "dataset_path": "/shared/share_mala/data/huggingface_datasets/downloads",
    "scratch_folder": "/shared/share_mala/tyen",
    # "dataset_path": "/shared/share_mala/tyen/huggingface_datasets/downloads",
    # "dataset_path": "/scratch/ty2531/huggingface_datasets/downloads",
    "slimpajama_subsets": [
        "RedPajamaCommonCrawl",
        "RedPajamaC4",
        "RedPajamaWikipedia",
        "RedPajamaStackExchange",
        "RedPajamaGithub",
        "RedPajamaArXiv",
        "RedPajamaBook",
    ],
}

# Global Constants that should not be printed or shared
ANON_GB = {
    "hf_token": "your_huggingface_token_here",
}

print(f"*" * 84)
print(f"Accessing Global Variables:\n{json.dumps(GB, indent=2)}")
print(f"*" * 84)
