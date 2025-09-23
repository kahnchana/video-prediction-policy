#!/usr/bin/env python3
"""
Simple parallel execution script for Calvin evaluation.
This script splits the evaluation sequences across multiple processes
and runs them in parallel at the top level.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from collections import namedtuple

import numpy as np
from hydra import compose, initialize

import wandb
from policy_evaluation.calvin_evaluate import main, print_and_save
from policy_evaluation.multistep_sequences import get_sequences


def cleanup_gpu_resources():
    """Clean up GPU resources to prevent hanging processes."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to cleanup GPU resources: {e}")


def worker_process(
    process_id, cfg_dict, eval_sequences_subset, device_id, result_queue
):
    """Worker process that runs evaluation on a subset of sequences."""
    print(f"Process {process_id}: Starting evaluation on device {device_id}")
    print(f"Process {process_id}: Evaluating {len(eval_sequences_subset)} sequences")

    # Set the device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    try:
        # Initialize hydra and compose config - FIX: Use correct path
        with initialize(
            config_path="../policy_conf", job_name="calvin_evaluate_all.yaml"
        ):
            cfg = compose(config_name="calvin_evaluate_all.yaml")

        # Override config with provided values
        for key, value in cfg_dict.items():
            if "." in key:
                parts = key.split(".")
                obj = cfg
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        break
                else:
                    setattr(obj, parts[-1], value)
            else:
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

        # Set device and disable wandb for worker processes
        cfg.device = 0  # Always use device 0 since we set CUDA_VISIBLE_DEVICES
        cfg.log_wandb = False

        # Run the main evaluation function with the subset of sequences
        results_dict, plans_dict = main(cfg, eval_sequences=eval_sequences_subset)

        # Extract results and plans from the dictionary (should only have one checkpoint)
        if results_dict and plans_dict:
            checkpoint = list(results_dict.keys())[0]
            results = results_dict[checkpoint]
            plans = plans_dict[checkpoint]
        else:
            results = []
            plans = {}

        # Return results through queue
        result_queue.put(
            {
                "process_id": process_id,
                "results": results,
                "plans": plans,
                "sequences": eval_sequences_subset,
            }
        )

        print(f"Process {process_id}: Completed successfully")

    except Exception as e:
        print(f"Process {process_id}: Failed with error: {e}")
        result_queue.put({"process_id": process_id, "error": str(e)})
    finally:
        cleanup_gpu_resources()


def split_sequences(sequences, num_processes):
    """Split sequences into roughly equal chunks for parallel processing."""
    chunk_size = len(sequences) // num_processes
    remainder = len(sequences) % num_processes

    chunks = []
    start_idx = 0

    for i in range(num_processes):
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunks.append(sequences[start_idx:end_idx])
        start_idx = end_idx

    return chunks


def aggregate_results(all_results):
    """Aggregate results from all processes."""
    combined_results = []
    combined_plans = defaultdict(list)

    # Sort results by process_id to maintain order
    all_results.sort(key=lambda x: x["process_id"])

    for result_data in all_results:
        if "error" in result_data:
            print(
                f"Process {result_data['process_id']} had an error: {result_data['error']}"
            )
            continue

        combined_results.extend(result_data["results"])

        # Combine plans if they exist
        if result_data["plans"]:
            for checkpoint, plans in result_data["plans"].items():
                combined_plans[checkpoint].extend(plans)

    return combined_results, combined_plans


def main_parallel():
    """Main function for parallel execution."""
    parser = argparse.ArgumentParser(description="Parallel Calvin evaluation")
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--action_model_folder", type=str, default="")
    parser.add_argument("--clip_model_path", type=str, default="")
    parser.add_argument("--calvin_abc_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument(
        "--num_processes", type=int, default=4, help="Number of parallel processes"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=1000,
        help="Total number of sequences to evaluate",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0,1,2,3",
        help="Comma-separated list of CUDA device IDs (e.g., '0,1' for 2 GPUs or '0,1,2,3' for 4 GPUs)",
    )

    args = parser.parse_args()

    # Parse devices to get all available GPUs
    device_list = [int(d.strip()) for d in args.devices.split(",")]
    gpu_devices = device_list  # Use all available GPUs

    if len(gpu_devices) == 0:
        print("Error: No GPUs specified in --devices argument")
        sys.exit(1)

    print(f"Starting parallel evaluation with {args.num_processes} processes")
    print(f"Using GPUs: {gpu_devices}")
    print(f"Total sequences: {args.num_sequences}")

    # Show distribution info
    processes_per_gpu = args.num_processes // len(gpu_devices)
    remainder = args.num_processes % len(gpu_devices)
    print(
        f"Process distribution: {processes_per_gpu} processes per GPU"
        + (
            f" (with {remainder} extra processes on first {remainder} GPU(s))"
            if remainder > 0
            else ""
        )
    )

    # Generate all evaluation sequences once
    print("Generating evaluation sequences...")
    eval_sequences = get_sequences(args.num_sequences)
    print(f"Generated {len(eval_sequences)} sequences")

    # Split sequences across processes
    sequence_chunks = split_sequences(eval_sequences, args.num_processes)

    # Prepare configuration for all processes
    cfg_dict = {
        "model.pretrained_model_path": args.video_model_path,
        "train_folder": args.action_model_folder,
        "model.text_encoder_path": args.clip_model_path,
        "root_data_dir": args.calvin_abc_dir,
        "num_sequences": args.num_sequences,
    }

    with initialize(config_path="../policy_conf", job_name="calvin_evaluate_all.yaml"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    cfg.num_sequences = args.num_sequences

    # Initialize wandb for logging (only in main process).
    try:
        wandb.init(name=args.run_name, config=dict(cfg))
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")

    # Create result queue for collecting results
    result_queue = mp.Queue()

    # Create and start processes
    processes = []
    for i in range(args.num_processes):
        chunk = sequence_chunks[i]
        # Assign GPU based on process index (round-robin)
        assigned_gpu = gpu_devices[i % len(gpu_devices)]

        process = mp.Process(
            target=worker_process, args=(i, cfg_dict, chunk, assigned_gpu, result_queue)
        )
        processes.append(process)
        process.start()
        print(f"Started process {i} with {len(chunk)} sequences on GPU {assigned_gpu}")

    # Wait for all processes to complete
    print("Waiting for all processes to complete...")
    for i, process in enumerate(processes):
        process.join()
        print(f"Process {i} completed")

    # Collect all results from the queue
    print("Collecting results from all processes...")
    all_results = []
    while not result_queue.empty():
        try:
            result = result_queue.get(timeout=1)
            all_results.append(result)
        except Exception as e:
            print(f"Error collecting results: {e}")
            break

    print(f"Collected results from {len(all_results)} processes")

    # Aggregate results
    if all_results:
        print("Aggregating results...")
        combined_results, combined_plans = aggregate_results(all_results)

        # Create log directory
        log_dir = Path("ckpt") / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging results to {log_dir}")

        # Get config for logging
        eval_config_class = namedtuple("eval_config", ["train_folder", "num_sequences"])
        eval_config = eval_config_class(log_dir, cfg_dict["num_sequences"])

        try:
            if combined_results:
                checkpoint = Path("last")
                total_results = {checkpoint: combined_results}
                print_and_save(
                    total_results, combined_plans, eval_config, log_dir=log_dir
                )
            else:
                print(f"Total sequences evaluated: {len(combined_results)}")
                if len(combined_results) > 0:
                    avg_seq_len = np.mean(combined_results)
                    print(f"Average successful sequence length: {avg_seq_len}")

                    data = {
                        "avg_seq_len": avg_seq_len,
                        "total_sequences": len(combined_results),
                        "results": combined_results,
                    }

                    with open(log_dir / "results.json", "w") as file:
                        json.dump(data, file, indent=2)
                else:
                    print("No successful evaluations!")

        except Exception as e:
            print(f"Error during result logging: {e}")

        # Finish wandb
        try:
            wandb.finish()
        except Exception as e:
            print(f"Error finishing wandb: {e}")
            pass
    else:
        print("No results collected from any processes!")

    print("All processes completed and results aggregated!")


if __name__ == "__main__":
    # Set multiprocessing start method.
    mp.set_start_method("spawn", force=True)

    # Run the main parallel function.
    main_parallel()
