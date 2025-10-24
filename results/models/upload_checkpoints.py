#!/usr/bin/env python3
"""
Script to upload latest model checkpoints to S3.

This script:
1. Loops through all folders under a specified path
2. Finds the latest step checkpoint (step0-unsharded, step10000-unsharded, step19531, etc.)
3. Uploads model.pt and config.yaml to an S3 bucket

Usage:
    python upload_checkpoints.py --models-path /path/to/models --bucket your-s3-bucket
    python upload_checkpoints.py --models-path /path/to/models --bucket your-s3-bucket --dry-run
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    logger.error("boto3 is required. Install it with: pip install boto3")
    sys.exit(1)


def extract_step_number(folder_name: str) -> int:
    """Extract step number from folder name like 'step10000-unsharded' or 'step19531'."""
    match = re.match(r"step(\d+)", folder_name)
    if match:
        return int(match.group(1))
    return -1


def find_latest_checkpoint(model_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint directory based on step number."""
    if not model_dir.exists() or not model_dir.is_dir():
        return None

    checkpoint_dirs = []
    for item in model_dir.iterdir():
        if item.is_dir() and item.name.startswith("step"):
            step_num = extract_step_number(item.name)
            if step_num >= 0:
                checkpoint_dirs.append((step_num, item))

    if not checkpoint_dirs:
        logger.warning(f"No valid checkpoint directories found in {model_dir}")
        return None

    # Sort by step number and return the latest
    checkpoint_dirs.sort(key=lambda x: x[0])
    latest_step, latest_dir = checkpoint_dirs[-1]
    logger.info(
        f"Found latest checkpoint: {latest_dir.name} (step {latest_step}) in {model_dir.name}"
    )
    return latest_dir


def upload_file_to_s3(
    file_path: Path, bucket: str, s3_key: str, s3_client, dry_run: bool = False
) -> bool:
    """Upload a file to S3."""
    if dry_run:
        logger.info(f"[DRY RUN] Would upload {file_path} to s3://{bucket}/{s3_key}")
        return True

    try:
        logger.info(f"Uploading {file_path} to s3://{bucket}/{s3_key}")
        s3_client.upload_file(str(file_path), bucket, s3_key)
        logger.info(f"Successfully uploaded {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {file_path}: {e}")
        return False


def upload_checkpoint_files(
    checkpoint_dir: Path, bucket: str, s3_prefix: str, s3_client, dry_run: bool = False
) -> Tuple[bool, bool]:
    """Upload model.pt and config.yaml from checkpoint directory to S3."""
    model_file = checkpoint_dir / "model.pt"
    config_file = checkpoint_dir / "config.yaml"

    model_uploaded = False
    config_uploaded = False

    # Upload model.pt
    if model_file.exists():
        s3_key = f"{s3_prefix}/model.pt"
        model_uploaded = upload_file_to_s3(
            model_file, bucket, s3_key, s3_client, dry_run
        )
    else:
        logger.warning(f"model.pt not found in {checkpoint_dir}")

    # Upload config.yaml
    if config_file.exists():
        s3_key = f"{s3_prefix}/config.yaml"
        config_uploaded = upload_file_to_s3(
            config_file, bucket, s3_key, s3_client, dry_run
        )
    else:
        logger.warning(f"config.yaml not found in {checkpoint_dir}")

    return model_uploaded, config_uploaded


def process_model_directory(
    model_dir: Path,
    bucket: str,
    s3_client,
    base_s3_prefix: str = "",
    dry_run: bool = False,
) -> bool:
    """Process a single model directory and upload its latest checkpoint."""
    logger.info(f"Processing model directory: {model_dir}")

    # Find the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(model_dir)
    if not latest_checkpoint:
        logger.warning(f"No valid checkpoints found in {model_dir}")
        return False

    # Create S3 prefix based on model directory name and checkpoint
    model_name = model_dir.name
    checkpoint_name = latest_checkpoint.name
    s3_prefix = f"{base_s3_prefix}/{model_name}/{checkpoint_name}".strip("/")

    # Upload the files
    model_uploaded, config_uploaded = upload_checkpoint_files(
        latest_checkpoint, bucket, s3_prefix, s3_client, dry_run
    )

    if model_uploaded or config_uploaded or dry_run:
        logger.info(f"Successfully processed {model_dir.name}")
        return True
    else:
        logger.error(f"Failed to upload any files from {model_dir.name}")
        return False


def validate_aws_access(bucket: str, s3_client) -> bool:
    """Validate AWS access and bucket permissions."""
    try:
        # Test S3 connection by attempting to head the bucket
        s3_client.head_bucket(Bucket=bucket)
        logger.info(f"Successfully connected to S3 bucket: {bucket}")
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure AWS credentials.")
        return False
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logger.error(
                f"S3 bucket '{bucket}' does not exist or you don't have access to it."
            )
        elif error_code == "403":
            logger.error(
                f"Access denied to S3 bucket '{bucket}'. Check your permissions."
            )
        else:
            logger.error(f"Error accessing S3 bucket '{bucket}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error connecting to S3: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload latest model checkpoints to S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --models-path /path/to/models --bucket my-s3-bucket
  %(prog)s --models-path /path/to/models --bucket my-s3-bucket --s3-prefix checkpoints
  %(prog)s --models-path /path/to/models --bucket my-s3-bucket --dry-run
  %(prog)s --models-path /path/to/models --bucket my-s3-bucket --aws-profile my-profile

The script will:
1. Scan all subdirectories in the models path
2. For each directory, find checkpoint folders like step0-unsharded, step10000-unsharded, step19531
3. Select the checkpoint with the highest step number
4. Upload model.pt and config.yaml to S3 with the structure:
   s3://bucket/prefix/model-name/checkpoint-name/model.pt
   s3://bucket/prefix/model-name/checkpoint-name/config.yaml
        """,
    )

    parser.add_argument(
        "--models-path",
        type=str,
        required=True,
        help="Path to directory containing model folders",
    )
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="models",
        help="S3 prefix/folder for uploaded models (default: 'models')",
    )
    parser.add_argument("--aws-profile", type=str, help="AWS profile to use (optional)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate models path
    models_path = Path(args.models_path)
    if not models_path.exists():
        logger.error(f"Models path does not exist: {models_path}")
        return 1

    if not models_path.is_dir():
        logger.error(f"Models path is not a directory: {models_path}")
        return 1

    # Initialize S3 client
    try:
        if args.aws_profile:
            logger.info(f"Using AWS profile: {args.aws_profile}")
            session = boto3.Session(profile_name=args.aws_profile)
            s3_client = session.client("s3")
        else:
            s3_client = boto3.client("s3")

    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        return 1

    # Validate AWS access (skip for dry run to avoid unnecessary API calls)
    if not args.dry_run:
        if not validate_aws_access(args.bucket, s3_client):
            return 1

    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be uploaded")

    # Process each model directory
    success_count = 0
    total_count = 0

    model_dirs = [item for item in models_path.iterdir() if item.is_dir()]
    if not model_dirs:
        logger.warning(f"No subdirectories found in {models_path}")
        return 1

    logger.info(f"Found {len(model_dirs)} model directories to process")

    for model_dir in model_dirs:
        total_count += 1
        logger.info(f"Processing {total_count}/{len(model_dirs)}: {model_dir.name}")

        if process_model_directory(
            model_dir, args.bucket, s3_client, args.s3_prefix, args.dry_run
        ):
            success_count += 1

        logger.info("-" * 50)  # Separator between models

    # Summary
    logger.info(f"Processing complete!")
    logger.info(
        f"Successfully processed: {success_count}/{total_count} model directories"
    )

    if success_count == total_count:
        logger.info("All model directories processed successfully!")
        return 0
    else:
        failed_count = total_count - success_count
        logger.warning(f"{failed_count} model directories failed to process")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
