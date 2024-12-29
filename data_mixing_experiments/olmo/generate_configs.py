"""Generate configs for OLMO to run
"""

import argparse
import json
import os
import string

import init_path
import numpy as np
import utils.data_utils as datu
import utils.misc_utils as mscu
import yaml
from tqdm import tqdm
from utils.global_variables import ANON_GB, GB


def get_args():
    parser = argparse.ArgumentParser(description="Generate configs for OLMO to run")
    parser.add_argument(
        "--categories",
        nargs="+",
        type=str,
        default=[
            "RedPajamaWikipedia",
            "RedPajamaStackExchange",
            "RedPajamaGithub",
            "RedPajamaArXiv",
            "RedPajamaBook",
        ],  # Exclude CommonCrawl and C4 from training files
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=str,
        choices=GB["training_params"].keys(),
        default=["20M"],
    )
    parser.add_argument("--num_configs", type=int, default=1)
    parser.add_argument(
        "--output_dir", type=str, default=init_path.PROJ_ROOT + "/olmo/configs"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=GB["scratch_folder"] + "/data-mixing/slimpajama/collected_arr",
    )
    parser.add_argument("--limit_validation", type=mscu.boolean, default=True)
    parser.add_argument("--drop_last", type=mscu.boolean, default=True)
    parser.add_argument("--overwrite", type=mscu.boolean, default=False)
    return parser.parse_args()


def generate_wandb_id(prefix):
    timestamp = mscu.get_time()
    chosen_chars = np.random.choice(
        list(string.ascii_letters) + list(string.digits), size=6, replace=True
    )
    random_id = "".join(chosen_chars)
    unique_id = f"{prefix}-{random_id}-{timestamp}"
    return unique_id


def generate_file_allocation(num_configs, categories, num_documents):
    # Generate percentage of numpy files for each category
    token_probs = datu.uniform_simplex_sample(
        n_points=num_configs, n_categories=len(categories)
    )

    ## Fix proportion of files for each category
    #   b/c for each category, the percentage of padding tokens is different
    padding_pct = [GB["SlimPajamaPaddingPercentage"][cat] for cat in categories]
    token_pct = 1 - np.array(padding_pct)

    file_probs = token_probs.copy()
    file_probs = (file_probs / token_pct) / np.sum(
        file_probs / token_pct, axis=1
    ).reshape(-1, 1)

    # Checking sum to 1
    assert np.allclose(file_probs.sum(axis=1), 1), file_probs.sum(axis=1)
    raw_allocation = file_probs * num_documents
    allocation = np.floor(raw_allocation).astype(int)

    for i in range(num_configs):
        remaining = num_documents - allocation[i].sum()
        if remaining > 0:
            remainders = raw_allocation[i] - allocation[i]
            indices = np.argsort(remainders)[::-1]
            allocation[i, indices[:remaining]] += 1

    # Checking sum to num_documents
    assert np.allclose(allocation.sum(axis=1), num_documents)
    return token_probs, file_probs, allocation


def main(scale, token_files, args):
    training_params = GB["training_params"][scale]
    num_documents = training_params.get("num_documents", 20_000)

    # 1M tokens per document
    token_probs, file_probs, train_file_allocation = generate_file_allocation(
        num_configs=args.num_configs,
        categories=args.categories,
        num_documents=num_documents,
    )

    # Check there are enough files for each category
    for i in range(args.num_configs):
        for category in args.categories:
            key = f"{category}_train"
            if train_file_allocation[i, args.categories.index(category)] > len(
                token_files[key]
            ):
                raise ValueError(
                    f"Config {i} has more documents than available for category {category}",
                    f"The sampled proportion is {token_probs[i, args.categories.index(category)]} at index {args.categories.index(category)}",
                )

    # Generate configs
    config_folder = f"{args.output_dir}/{scale}"
    mscu.make_folder(config_folder)

    # Find latest config number if not overwriting
    start_config_idx = 0
    if not args.overwrite:
        config_files = os.listdir(config_folder)
        config_files = [
            f for f in config_files if f.startswith("config_") and f.endswith(".yaml")
        ]
        if config_files:
            latest_config = max(
                int(f.split("_")[1].split(".")[0]) for f in config_files
            )
            start_config_idx = latest_config + 1
            print(
                f"Found existing configs. Starting from config idx {start_config_idx}"
            )
        else:
            start_config_idx = 0

    # Load token counts
    token_counts = {}
    for category in GB["slimpajama_subsets"]:
        token_counts[category] = np.load(
            f"{args.data_dir}/train/{category}/_token_counts.npy"
        )

    for i, config_idx in tqdm(
        enumerate(range(start_config_idx, start_config_idx + args.num_configs)),
        desc="Generating configs",
        total=args.num_configs,
    ):
        # Sample files
        selected_files = []
        actual_tokens = np.zeros(len(args.categories))
        for category_idx, category in enumerate(args.categories):
            key = f"{category}_train"
            num_files = train_file_allocation[i, category_idx]

            selected_idc = np.random.choice(
                len(token_files[key]), num_files, replace=False
            )
            actual_tokens[category_idx] = np.sum(token_counts[category][selected_idc])
            selected_files.extend(token_files[key][selected_idc].tolist())
        actual_token_probs = actual_tokens / np.sum(actual_tokens)

        # Write config
        generated_date = mscu.get_time()
        run_name = (
            f"{scale}_"
            + f"config_{config_idx:04d}_"
            + "-".join(
                f"{args.categories[cat_idx]}{token_probs[i, cat_idx]:.2f}"
                for cat_idx in range(len(args.categories))
            )
            + "_"
            + generated_date
        )

        config = {
            "compile": None,
            "run_name": run_name,
            "wandb": {
                "run_id": generate_wandb_id(str(scale)),
                "entity": ANON_GB["wandb_username"],
                "name": run_name,
                "project": "olmo-slimpajama",
                "group": scale,
            },
            # Data Mixing configs
            "data_mixing": {
                "categories": args.categories,
                "target_token_probs": token_probs[i].tolist(),
                "actual_token_probs": actual_token_probs.tolist(),
                "target_file_probs": file_probs[i].tolist(),
                "actual_file_probs": [
                    float(
                        train_file_allocation[i, args.categories.index(category)]
                        / num_documents
                    )
                    for category in args.categories
                ],
            },
            # Train/Eval configs
            "global_train_batch_size": training_params.get(
                "global_train_batch_size", 1024
            ),
            "device_train_microbatch_size": training_params.get(
                "device_train_microbatch_size", 8
            ),
            "device_eval_batch_size": training_params.get("device_eval_batch_size", 64),
            "eval_interval": 100,
            "eval_subset_num_batches": -1,
            "max_duration": "1ep",  # 1 epoch
            "max_grad_norm": 1.0,
            "max_grad_norm_ratio": None,
            # Save configs
            "save_folder": GB["scratch_folder"] + f"/olmo/{scale}/{run_name}",
            "save_interval_unsharded": 1000,
            "save_num_unsharded_checkpoints_to_keep": -1,
            "save_overwrite": False,
            # Model configs
            "model": {
                "activation_type": "swiglu",
                "alibi": False,
                "attention_dropout": 0.0,
                "attention_layer_norm": False,
                "block_type": "sequential",
                "clip_qkv": None,
                "d_model": training_params["d_model"],
                "embedding_dropout": 0.0,
                "embedding_size": 50257,
                "eos_token_id": 50256,
                "flash_attention": True,
                "include_bias": False,
                "init_cutoff_factor": 3,
                "init_device": "cuda",
                "init_fn": "mitchell",
                "init_std": 0.02,
                "layer_norm_type": "default",
                "layer_norm_with_affine": False,
                "bias_for_layer_norm": False,
                "attention_layer_norm_with_affine": False,
                "max_sequence_length": GB["context_length"],
                "mlp_ratio": training_params["mlp_ratio"],
                "n_heads": training_params["n_heads"],
                "n_layers": training_params["n_layers"],
                "pad_token_id": 50256,
                "residual_dropout": 0.0,
                "rope": True,
                "vocab_size": 50257,
                "weight_tying": False,
            },
            # Optimizer configs
            "optimizer": {
                "betas": [0.9, 0.95],
                "decay_embeddings": True,
                "decay_norm_and_bias": True,
                "eps": 1e-8,
                "learning_rate": 0.0006,
                "metrics_log_interval": 10,
                "name": "adamw",
                "weight_decay": 0.1,
            },
            # Scheduler configs
            "scheduler": {
                "alpha_f": 0.1,
                "name": "cosine_with_warmup",
                "t_warmup": 2000,
                "warmup_min_lr": 0,
            },
            # Tokenizer
            "tokenizer": {
                "identifier": f"{init_path.PROJ_ROOT}/olmo/GPT2TokenizerFast-gpt2/tokenizer.json",
                "truncate_direction": "right",
            },
            # DDP
            "distributed_strategy": "ddp",
            "ddp": {
                "find_unused_params": False,
                "grad_sync_mode": "batch",
            },
            # Others
            "seed": np.random.randint(0, 100000),
            "precision": "amp_bf16",
            "speed_monitor": {"window_size": 20},
            "dry_run": False,
            "gen1_gc_interval": 1,
            "load_path": None,
            # Evaluation
            "evaluators": [
                {
                    "label": "slimpajama",
                    "data": {
                        "drop_last": True,
                        "num_workers": 0,
                        "pad_direction": "right",
                        "persistent_workers": True,
                        "pin_memory": True,
                        "prefetch_factor": 8,
                        "datasets": {
                            category: [
                                str(f) for f in token_files[f"{category}_validation"]
                            ]
                            for category in GB["slimpajama_subsets"]
                        },
                    },
                },
                {
                    "label": "piqa",
                    "type": "downstream",
                },
                {
                    "label": "hellaswag",
                    "type": "downstream",
                },
                {
                    "label": "winogrande",
                    "type": "downstream",
                },
                {
                    "label": "arc_easy",
                    "type": "downstream",
                },
            ],
            # Training
            "data": {
                "drop_last": True,
                "num_workers": 0,
                "pad_direction": "right",
                "paths": [str(f) for f in selected_files],
                "persistent_workers": True,
                "pin_memory": True,
                "prefetch_factor": 8,
                "timeout": 0,
            },
        }
        iname = f"{config_idx:04d}"
        with open(f"{config_folder}/config_{iname}.yaml", "w") as f:
            yaml.dump(config, f, sort_keys=False)
        with open(f"{config_folder}/config_{iname}_details.json", "w") as f:
            details = {
                "data_mixing": {
                    "categories": args.categories,
                    "target_token_probs": token_probs[
                        i
                    ].tolist(),  # This is the intended proportion of tokens for each category
                    "actual_token_probs": actual_token_probs.tolist(),
                    "target_file_probs": file_probs[
                        i
                    ].tolist(),  # This is the intended proportion of files for each category
                    "actual_file_probs": [
                        float(
                            train_file_allocation[i, args.categories.index(category)]
                            / num_documents
                        )
                        for category in args.categories
                    ],
                },
                "generated_date": generated_date,
                "run_date": None,
                "has_run": False,
            }
            json.dump(details, f, indent=2)


if __name__ == "__main__":
    args = get_args()
    print(args)

    # Get list of documents for each category for training
    token_files = {}
    for category in args.categories:
        key = f"{category}_train"
        folder = f"{args.data_dir}/train/{category}"
        chunk_sizes = np.load(f"{folder}/_new_chunk_sizes.npy")
        cur_files = [
            f"{folder}/chunk={i:06d}_input_ids.npy" for i in range(len(chunk_sizes))
        ]
        if args.drop_last:
            cur_files = cur_files[:-1]
            print(f"Drop last file for {key} with {chunk_sizes[-1]} sequences")
        token_files[key] = np.array(cur_files)
        print(f"{key}: Number of Input ids files: {len(token_files[key])}")
        print(f"*" * 84)

    # Get list of documents for each category for validation
    ## Save and load the same files for validation for consistent evaluation
    valid_choice_fname = f"{init_path.PROJ_ROOT}/olmo/scratch/validation.json"
    if os.path.exists(valid_choice_fname):
        with open(valid_choice_fname, "r") as f:
            valid_choice = json.load(f)
        for category in GB["slimpajama_subsets"]:
            key = f"{category}_validation"
            folder = f"{args.data_dir}/validation/{category}"

            token_files[key] = np.array(
                [f"{folder}/{fname}" for fname in valid_choice[category]]
            )
            print(f"{key}: Number of Input ids files: {len(token_files[key])}")
            print(f"*" * 84)
    else:
        # Generate and save the chosen files for validation
        valid_choice = {}
        for category in GB["slimpajama_subsets"]:
            key = f"{category}_validation"
            folder = f"{args.data_dir}/validation/{category}"
            chunk_sizes = np.load(f"{folder}/_new_chunk_sizes.npy")
            cur_files = [
                f"chunk={i:06d}_input_ids.npy" for i in range(len(chunk_sizes))
            ]
            if args.drop_last:
                cur_files = cur_files[:-1]
                print(f"Drop last file for {key} with {chunk_sizes[-1]} sequences")
            if args.limit_validation:
                if category in ["RedPajamaCommonCrawl", "RedPajamaC4"]:
                    count = 4
                else:
                    count = 2
                print(
                    f"Limiting validation files for {key} from {len(cur_files)} to {count}"
                )
                cur_files = np.random.choice(
                    cur_files, min(count, len(cur_files)), replace=False
                )
            valid_choice[category] = cur_files.tolist()
            token_files[key] = np.array([f"{folder}/{fname}" for fname in cur_files])
            print(f"{key}: Number of Input ids files: {len(token_files[key])}")
            print(f"*" * 84)

        mscu.make_folder(os.path.dirname(valid_choice_fname))
        with open(valid_choice_fname, "w") as f:
            json.dump(valid_choice, f, indent=2)

    for scale in args.scales:
        print(f"Generating configs for scale: {scale}")
        main(scale, token_files, args)
