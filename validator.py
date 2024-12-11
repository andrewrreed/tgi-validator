#!/usr/bin/env python3

import os
import subprocess
import time
import logging
import signal
import psutil
import argparse
from typing import Optional
import json
from datetime import datetime
from pathlib import Path
from huggingface_hub import get_safetensors_metadata
from transformers import AutoConfig


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GPU_TYPES = [
    {"gpu_type": "nvidia-t4", "per_gpu_memory_in_gb": 16, "compute_capability": 7.5},
    {"gpu_type": "nvidia-a2", "per_gpu_memory_in_gb": 16, "compute_capability": 8.6},
    {"gpu_type": "nvidia-l4", "per_gpu_memory_in_gb": 24, "compute_capability": 8.9},
    {"gpu_type": "nvidia-a30", "per_gpu_memory_in_gb": 24, "compute_capability": 8.0},
    {"gpu_type": "nvidia-a10g", "per_gpu_memory_in_gb": 24, "compute_capability": 8.6},
    {"gpu_type": "nvidia-a40", "per_gpu_memory_in_gb": 48, "compute_capability": 8.6},
    {"gpu_type": "nvidia-l40s", "per_gpu_memory_in_gb": 48, "compute_capability": 8.9},
    {"gpu_type": "nvidia-l40", "per_gpu_memory_in_gb": 48, "compute_capability": 8.9},
    {"gpu_type": "nvidia-a16", "per_gpu_memory_in_gb": 64, "compute_capability": 8.6},
    {"gpu_type": "nvidia-h100", "per_gpu_memory_in_gb": 80, "compute_capability": 9.0},
    {"gpu_type": "nvidia-a100", "per_gpu_memory_in_gb": 80, "compute_capability": 8.0},
]


def bytes_to_gb(bytes: int):
    return round((bytes) / 10**9, 5)


def get_model_overhead(model_id: str, dtype: str = "float16") -> float:
    """Get the model size in GB for a given model ID and data type."""
    metadata = get_safetensors_metadata(model_id)

    if hasattr(metadata, "metadata") and metadata.metadata is not None:
        if metadata.metadata.get("total_size", False):
            return bytes_to_gb(metadata.metadata["total_size"])
    elif hasattr(metadata, "parameter_count"):
        if metadata.parameter_count:
            # specific to Llama-3.2 models
            return bytes_to_gb(metadata.parameter_count["BF16"] * 2)

    raise ValueError(f"Could not determine model size for {model_id}")


def get_gpu_vram(gpu_type: str) -> float:
    """Get the VRAM in GB for a given GPU type."""
    for gpu in GPU_TYPES:
        if gpu["gpu_type"] == gpu_type:
            return gpu["per_gpu_memory_in_gb"]
    raise ValueError(f"Unknown GPU type: {gpu_type}")


def get_config_values(max_total_tokens: int) -> dict:
    max_input_tokens = max_total_tokens - 1
    max_batch_prefill_tokens = max_input_tokens + 50

    return {
        "max_input_tokens": max_input_tokens,
        "max_total_tokens": max_total_tokens,
        "max_batch_prefill_tokens": max_batch_prefill_tokens,
    }


def create_docker_run_command(
    model_id: str, max_total_tokens: int, num_shards: int, image_uri: str
) -> str:
    """Create the docker run command with appropriate parameters."""
    max_input_tokens = max_total_tokens - 1
    max_batch_prefill_tokens = max_input_tokens + 50

    return f"""docker run --gpus all --shm-size 1g \
    -e HF_TOKEN={os.getenv("HF_TOKEN")} \
    -e NUM_SHARD={num_shards} \
    -e MAX_INPUT_TOKENS={max_input_tokens} \
    -e MAX_TOTAL_TOKENS={max_total_tokens} \
    -e MAX_BATCH_PREFILL_TOKENS={max_batch_prefill_tokens} \
    -p 8080:80 \
    -v $PWD/data:/data \
    {image_uri} \
    --model-id {model_id}
    """


def save_test_results(model_id: str, gpu_type: str, all_results: list, image_uri: str):
    """Save all test results to a single JSON file."""
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = (
        results_dir
        / f"model_test_{model_id.replace('/', '_')}_{gpu_type}_{timestamp}.json"
    )

    result_data = {
        "model_id": model_id,
        "gpu_type": gpu_type,
        "image_uri": image_uri,
        "tests": all_results,
        "timestamp": timestamp,
    }

    with open(result_file, "w") as f:
        json.dump(result_data, f, indent=2)

    logger.info(f"All test results saved to {result_file}")


def run_model_container(
    model_id: str, max_total_tokens: int, num_shards: int, image_uri: str
) -> tuple[bool, dict]:
    """Modified to return both success status and test configuration"""
    config = get_config_values(max_total_tokens)
    cmd = create_docker_run_command(model_id, max_total_tokens, num_shards, image_uri)
    logger.info(f"Starting container with command:\n{cmd}")

    process: Optional[subprocess.Popen] = None

    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Monitor the output
        success = False
        start_time = time.time()
        timeout = 60 * 90  # 1.5 hour timeout

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("Container startup timed out after 1.5 hours")

            output = process.stdout.readline()
            if output:
                logger.info(output.strip())
                if "Connected" in output:
                    success = True
                    logger.info("Container successfully started and connected!")
                    break
                elif "error" in output.lower() or "failed" in output.lower():
                    raise Exception(f"Container failed to start: {output}")

            # Check if process has ended
            if process.poll() is not None:
                break

        save_test_results_data = {
            "max_input_tokens": config["max_input_tokens"],
            "max_total_tokens": config["max_total_tokens"],
            "max_batch_prefill_tokens": config["max_batch_prefill_tokens"],
            "num_shard": num_shards,
            "success": success,
            "skipped": False,
            "error": None,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        return success, save_test_results_data

    except Exception as e:
        logger.error(f"Error running container: {str(e)}")
        save_test_results_data = {
            **config,
            "num_shard": num_shards,
            "success": False,
            "skipped": False,
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        return False, save_test_results_data

    finally:
        if process and process.pid:
            logger.info("Cleaning up Docker container...")
            try:
                # Get container ID of the running text-generation-inference container
                get_container_cmd = "docker ps -q --filter ancestor=ghcr.io/huggingface/text-generation-inference:2.4.1"
                container_id = (
                    subprocess.check_output(get_container_cmd, shell=True)
                    .decode()
                    .strip()
                )

                if container_id:
                    # Stop the container gracefully with timeout
                    logger.debug(f"Stopping container {container_id}")
                    stop_cmd = f"docker stop --time 30 {container_id}"
                    subprocess.run(stop_cmd, shell=True, check=True)

                    # Verify container is stopped
                    check_cmd = f"docker ps -q --filter id={container_id}"
                    if (
                        not subprocess.check_output(check_cmd, shell=True)
                        .decode()
                        .strip()
                    ):
                        logger.info("Docker container stopped successfully")
                    else:
                        # Force remove if stop didn't work
                        logger.warning("Container still running, forcing removal")
                        subprocess.run(
                            f"docker rm -f {container_id}", shell=True, check=True
                        )
                else:
                    logger.warning("No running container found")

                # Kill the parent process if it's somehow still running
                try:
                    process.kill()
                except:
                    pass

            except subprocess.CalledProcessError as e:
                logger.error(f"Error stopping Docker container: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error during cleanup: {str(e)}")


def try_run_model_with_shards(
    model_id: str,
    max_total_tokens: int,
    initial_shards: int,
    min_shards_by_tokens: dict,
    gpu_type: str,
    image_uri: str,
) -> tuple[bool, dict]:
    """Try running model with different shard counts, checking VRAM requirements first."""
    shard_options = [1, 2, 4, 8]
    model_vram = get_model_overhead(model_id)
    gpu_vram = get_gpu_vram(gpu_type)

    logger.info(f"Model {model_id} requires {model_vram}GB VRAM")
    logger.info(f"GPU type {gpu_type} has {gpu_vram}GB VRAM per GPU")

    # Determine minimum shards needed based on VRAM requirements
    min_shards_for_vram = max(1, int(model_vram / gpu_vram))
    logger.info(
        f"Model requires minimum {min_shards_for_vram} shard(s) based on VRAM requirements (model size {model_vram}GB / GPU VRAM {gpu_vram}GB)"
    )

    # If even maximum shards isn't enough, return formatted error
    if min_shards_for_vram > max(shard_options):
        logger.error(
            f"Model requires {min_shards_for_vram} shards, but maximum available is {max(shard_options)}"
        )
        config = get_config_values(max_total_tokens)
        return False, {
            "max_input_tokens": config["max_input_tokens"],
            "max_total_tokens": config["max_total_tokens"],
            "max_batch_prefill_tokens": config["max_batch_prefill_tokens"],
            "num_shard": max(shard_options),
            "success": False,
            "skipped": False,
            "error": "Insufficient GPU memory",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

    # Determine minimum shards based on previous results
    min_required_shards = 1
    for tokens, shards in min_shards_by_tokens.items():
        if tokens < max_total_tokens and shards > min_required_shards:
            min_required_shards = shards

    # Use the maximum of all minimum shard requirements
    start_shards = max(min_required_shards, initial_shards, min_shards_for_vram)

    # Find starting index in shard options
    try:
        start_index = shard_options.index(start_shards)
    except ValueError:
        # If exact shard count not found, find next highest value
        start_index = 0
        for i, shards in enumerate(shard_options):
            if shards >= start_shards:
                start_index = i
                break
        if start_index == 0 and shard_options[0] < start_shards:
            logger.error(
                f"Model requires {start_shards} shards, but maximum available is {shard_options[-1]}"
            )
            return False, {"error": "Insufficient GPU memory"}

    logger.info(
        f"Starting with {shard_options[start_index]} shard(s) based on VRAM and previous results"
    )

    for num_shards in shard_options[start_index:]:
        total_vram = num_shards * gpu_vram
        if total_vram < model_vram:
            logger.warning(
                f"Skipping {num_shards} shard(s) as total VRAM ({total_vram}GB) < model size ({model_vram}GB)"
            )
            continue

        logger.info(f"Attempting to start model with {num_shards} shard(s)...")
        success, test_results = run_model_container(
            model_id, max_total_tokens, num_shards, image_uri
        )
        if success:
            return True, test_results
        logger.warning(f"Failed with {num_shards} shard(s), trying next shard count...")

    return False, test_results


def main():
    parser = argparse.ArgumentParser(
        description="Run text-generation-inference container"
    )
    parser.add_argument(
        "--model-ids",
        type=str,
        nargs="+",
        required=True,
        help="List of model IDs to run",
    )
    parser.add_argument(
        "--token-values",
        type=int,
        nargs="+",
        default=[2048],
        help="List of max token values to test (default: [2048])",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Initial number of shards (default: 1)",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        required=True,
        help="GPU type (e.g., nvidia-t4, nvidia-a100)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--image-uri",
        type=str,
        default="ghcr.io/huggingface/text-generation-inference:2.4.1",
        help="Docker image URI for text-generation-inference (default: ghcr.io/huggingface/text-generation-inference:2.4.1)",
    )

    args = parser.parse_args()

    # Validate GPU type
    if args.gpu_type not in [gpu["gpu_type"] for gpu in GPU_TYPES]:
        parser.error(
            f"Invalid GPU type. Must be one of: {', '.join([gpu['gpu_type'] for gpu in GPU_TYPES])}"
        )

    if args.debug:
        logger.setLevel(logging.DEBUG)

    overall_success = True

    for model_id in args.model_ids:
        logger.info(f"Running tests for model: {model_id}")
        all_results = []
        min_shards_by_tokens = {}

        # Sort token values to ensure we test from smallest to largest
        token_values = sorted(args.token_values)

        for max_tokens in token_values:
            logger.info(f"Testing with max_tokens = {max_tokens}")
            success, test_results = try_run_model_with_shards(
                model_id,
                max_tokens,
                args.num_shards,
                min_shards_by_tokens,
                args.gpu_type,
                args.image_uri,
            )
            all_results.append(test_results)

            if success:
                # Store the successful shard count for future reference
                min_shards_by_tokens[max_tokens] = test_results["num_shard"]
            else:
                overall_success = False
                # Check if we failed with maximum shards
                if (
                    test_results.get("num_shard", 0) == 8
                    or test_results.get("error") == "Insufficient GPU memory"
                ):
                    remaining_tokens = token_values[
                        token_values.index(max_tokens) + 1 :
                    ]
                    if remaining_tokens:
                        logger.warning(
                            f"Failed with maximum shards ({test_results.get('num_shard', 8)}) for {max_tokens} tokens. "
                            f"Skipping remaining token values {remaining_tokens} as they would also fail."
                        )
                        # Add failure entries for remaining token values
                        for skip_tokens in remaining_tokens:
                            skip_results = {
                                "max_input_tokens": skip_tokens - 1,
                                "max_total_tokens": skip_tokens,
                                "max_batch_prefill_tokens": skip_tokens - 1 + 50,
                                "num_shard": test_results.get("num_shard", 8),
                                "success": False,
                                "skipped": True,
                                "error": "Skipped due to previous failure with maximum shards",
                                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                            }
                            all_results.append(skip_results)
                        break  # Exit the token values loop

        # Save all results to a single file
        save_test_results(model_id, args.gpu_type, all_results, args.image_uri)

    if overall_success:
        logger.info("All model configurations tested successfully!")
        exit(0)
    else:
        logger.error("Some model configurations failed")
        exit(1)


if __name__ == "__main__":
    main()
