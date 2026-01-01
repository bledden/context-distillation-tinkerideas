#!/usr/bin/env python3
"""
Run Distribution Cliff Mitigation Experiment.

This experiment tests 6 different strategies to mitigate the "distribution cliff"
problem observed in hybrid context distillation, where transitioning from off-policy
to on-policy training causes catastrophic collapse (0% downstream accuracy).

Methods Tested:
1. extended_on_policy - Pure on-policy with 200 steps (baseline)
2. teacher_seeded - On-policy with teacher-initialized sampling
3. mixture - Mixed objectives every step (no phases)
4. replay_buffer - On-policy with teacher replay buffer
5. kl_anchored - Off-policy with KL anchor to initial student
6. reverse_curriculum - On-policy first, then off-policy refinement

Plus baselines:
- on_policy_gkd (100 steps) - Original on-policy that achieved 4.2%
- hybrid (100 steps) - Original hybrid that achieved 0%

Usage:
    # Quick test (3 seeds, 50 steps)
    python run_cliff_mitigation_experiment.py --quick-test

    # Full experiment with Qwen (default)
    python run_cliff_mitigation_experiment.py --n-seeds 10 --model-family qwen

    # Full experiment with Llama
    python run_cliff_mitigation_experiment.py --n-seeds 10 --model-family llama

    # Run specific methods only
    python run_cliff_mitigation_experiment.py --methods extended_on_policy mixture replay_buffer

    # Run in background (Qwen)
    nohup python run_cliff_mitigation_experiment.py --n-seeds 10 --model-family qwen > cliff_qwen.log 2>&1 &

    # Run in background (Llama)
    nohup python run_cliff_mitigation_experiment.py --n-seeds 10 --model-family llama > cliff_llama.log 2>&1 &
"""

import argparse
import json
import logging
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from env_loader import load_env
load_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# All mitigation methods to test
CLIFF_MITIGATION_METHODS = [
    "on_policy_gkd",        # Baseline: original on-policy (4.2% accuracy)
    "hybrid",               # Baseline: original hybrid (0% accuracy)
    "extended_on_policy",   # Method 1: Pure on-policy with 200 steps
    "teacher_seeded",       # Method 2: Teacher-initialized sampling
    "mixture",              # Method 3: Mixed objectives every step
    "replay_buffer",        # Method 4: Replay buffer with teacher injection
    "kl_anchored",          # Method 5: KL-anchored off-policy
    "reverse_curriculum",   # Method 6: On-policy first, then off-policy
]

# Model family configurations
MODEL_FAMILIES = {
    "qwen": {
        "student_model": "Qwen/Qwen3-4B-Instruct-2507",
        "teacher_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "description": "Qwen family: 4B student ← 30B MoE teacher",
    },
    "llama": {
        "student_model": "meta-llama/Llama-3.1-8B-Instruct",
        "teacher_model": "meta-llama/Llama-3.3-70B-Instruct",
        "description": "Llama family: 8B student ← 70B teacher",
    },
}


def load_gsm8k_prompts(num_prompts: int = 100) -> List[Dict]:
    """Load GSM8K benchmark prompts."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("openai/gsm8k", "main", split="train")

        prompts = []
        for i, example in enumerate(dataset):
            if i >= num_prompts:
                break
            prompts.append({
                "question": example["question"],
                "answer": example["answer"],
                "type": "math_reasoning",
            })

        logger.info(f"Loaded {len(prompts)} GSM8K prompts")
        return prompts
    except Exception as e:
        logger.warning(f"Failed to load GSM8K: {e}")
        return None


def format_prompt(example: Dict) -> str:
    """Format benchmark example as prompt."""
    return f"Question: {example['question']}\nSolve this step by step and provide the final numerical answer."


def extract_answer(response: str) -> Optional[str]:
    """Extract numerical answer from response."""
    # GSM8K format: #### <number>
    match = re.search(r"####\s*\$?([0-9,]+(?:\.[0-9]+)?)", response)
    if match:
        return match.group(1).replace(",", "").replace("$", "")

    # "The answer is X"
    match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        response, re.IGNORECASE
    )
    if match:
        return match.group(1).replace(",", "").replace("$", "")

    # Fallback: last number
    numbers = re.findall(r"\$?([0-9,]+(?:\.[0-9]+)?)", response)
    if numbers:
        return numbers[-1].replace(",", "").replace("$", "")

    return None


def evaluate_downstream(
    sampling_client,
    tokenizer,
    renderer,
    benchmark_examples: List[Dict],
    num_samples: int = 50,
) -> Dict:
    """Evaluate model on GSM8K downstream benchmark."""
    import tinker

    correct = 0
    total = 0

    for example in benchmark_examples[:num_samples]:
        prompt = format_prompt(example)
        messages = [{"role": "user", "content": prompt}]
        model_input = renderer.build_generation_prompt(messages)

        try:
            result = sampling_client.sample(
                prompt=model_input,
                sampling_params=tinker.SamplingParams(max_tokens=256, temperature=0.01),
                num_samples=1,
            ).result()

            if result.sequences:
                response = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
                predicted = extract_answer(response)

                # Extract ground truth
                gt_match = re.search(r"####\s*([0-9,]+)", example.get("answer", ""))
                ground_truth = gt_match.group(1).replace(",", "") if gt_match else ""

                if predicted == ground_truth:
                    correct += 1
                total += 1
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            continue

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistics with 95% CI."""
    import math

    if not values:
        return {}

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0
    std = math.sqrt(variance)
    stderr = std / math.sqrt(n) if n > 0 else 0

    t_values = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228}
    t = t_values.get(n, 1.96)

    return {
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "ci_lower": mean - t * stderr,
        "ci_upper": mean + t * stderr,
        "n": n,
    }


def run_cliff_mitigation_experiment(
    n_seeds: int = 10,
    max_steps: int = 100,
    num_prompts: int = 50,
    output_dir: str = "./outputs/cliff_mitigation",
    model_family: str = "qwen",
    methods: List[str] = None,
    run_downstream_eval: bool = True,
) -> Dict:
    """
    Run the full cliff mitigation experiment.

    Args:
        n_seeds: Number of random seeds per method
        max_steps: Training steps per run (extended_on_policy uses 2x)
        num_prompts: Number of training prompts
        output_dir: Output directory
        model_family: Which model family to use ("qwen" or "llama")
        methods: Which methods to run (default: all)
        run_downstream_eval: Whether to evaluate downstream accuracy

    Returns:
        Full results with statistics
    """
    from tinker_trainer import TinkerDistillationTrainer, TinkerTrainingConfig
    from config import ExperimentConfig

    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {model_family}. Choose from: {list(MODEL_FAMILIES.keys())}")

    model_config = MODEL_FAMILIES[model_family]

    if methods is None:
        methods = CLIFF_MITIGATION_METHODS.copy()

    # Create output directory with model family suffix
    output_dir = f"{output_dir}_{model_family}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DISTRIBUTION CLIFF MITIGATION EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model Family: {model_family}")
    logger.info(f"  Student: {model_config['student_model']}")
    logger.info(f"  Teacher: {model_config['teacher_model']}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Seeds: {n_seeds}")
    logger.info(f"Steps per run: {max_steps}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)

    # Load benchmark
    benchmark_data = load_gsm8k_prompts(num_prompts)
    if benchmark_data:
        prompts = [format_prompt(ex) for ex in benchmark_data]
    else:
        logger.error("Failed to load GSM8K, cannot run experiment")
        return {}

    eval_prompts = prompts[:10]

    # Initialize results
    results = {
        "experiment": "cliff_mitigation_comparison",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_seeds": n_seeds,
            "max_steps": max_steps,
            "num_prompts": num_prompts,
            "model_family": model_family,
            "student_model": model_config["student_model"],
            "teacher_model": model_config["teacher_model"],
            "methods": methods,
            "benchmark": "gsm8k",
        },
        "methods": {},
    }

    for method in methods:
        results["methods"][method] = {
            "final_losses": [],
            "avg_scores": [],
            "avg_kl": [],
            "downstream_accuracy": [],
            "histories": [],
        }

    # Run experiments - configure for the selected model family
    config = TinkerTrainingConfig(
        student_model=model_config["student_model"],
        teacher_model=model_config["teacher_model"],
    )

    for seed in range(n_seeds):
        logger.info(f"\n{'='*70}")
        logger.info(f"SEED {seed + 1}/{n_seeds}")
        logger.info(f"{'='*70}")

        for method in methods:
            logger.info(f"\n--- Method: {method} ---")

            # Fresh trainer for each run
            exp_config = ExperimentConfig()
            trainer = TinkerDistillationTrainer(config, exp_config)
            trainer.initialize()

            # Run the appropriate method
            try:
                if method == "on_policy_gkd":
                    result = trainer.run_on_policy_distillation(
                        prompts, max_steps=max_steps, eval_prompts=eval_prompts
                    )
                elif method == "hybrid":
                    result = trainer.run_hybrid_distillation(
                        prompts, max_steps=max_steps, eval_prompts=eval_prompts
                    )
                elif method == "extended_on_policy":
                    result = trainer.run_extended_on_policy(
                        prompts, max_steps=max_steps * 2, eval_prompts=eval_prompts
                    )
                elif method == "teacher_seeded":
                    result = trainer.run_teacher_seeded_on_policy(
                        prompts, max_steps=max_steps, eval_prompts=eval_prompts
                    )
                elif method == "mixture":
                    result = trainer.run_mixture_distillation(
                        prompts, max_steps=max_steps, eval_prompts=eval_prompts
                    )
                elif method == "replay_buffer":
                    result = trainer.run_replay_buffer_distillation(
                        prompts, max_steps=max_steps, eval_prompts=eval_prompts
                    )
                elif method == "kl_anchored":
                    result = trainer.run_kl_anchored_off_policy(
                        prompts, max_steps=max_steps, eval_prompts=eval_prompts
                    )
                elif method == "reverse_curriculum":
                    result = trainer.run_reverse_curriculum(
                        prompts, max_steps=max_steps, eval_prompts=eval_prompts
                    )
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue

                # Store results
                results["methods"][method]["final_losses"].append(result.get("final_loss", 0.0))
                results["methods"][method]["avg_scores"].append(result.get("avg_score", 0.0))
                results["methods"][method]["avg_kl"].append(result.get("avg_kl", 0.0))
                results["methods"][method]["histories"].append({
                    "seed": seed,
                    "history": result.get("history", [])[-5:],  # Last 5 steps
                })

                # Downstream evaluation
                if run_downstream_eval:
                    logger.info("Running downstream GSM8K evaluation...")
                    eval_result = evaluate_downstream(
                        trainer._sampling_client,
                        trainer._tokenizer,
                        trainer._renderer,
                        benchmark_data,
                        num_samples=50,
                    )
                    accuracy = eval_result["accuracy"]
                    results["methods"][method]["downstream_accuracy"].append(accuracy)
                    logger.info(f"Downstream accuracy: {accuracy:.1%} ({eval_result['correct']}/{eval_result['total']})")

            except Exception as e:
                logger.error(f"Method {method} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save intermediate results
        intermediate_path = f"{output_dir}/intermediate_results_seed{seed}.json"
        with open(intermediate_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        logger.info(f"Intermediate results saved to {intermediate_path}")

    # Compute final statistics
    logger.info("\n" + "=" * 70)
    logger.info("COMPUTING FINAL STATISTICS")
    logger.info("=" * 70)

    for method in methods:
        method_data = results["methods"][method]

        if method_data["final_losses"]:
            method_data["loss_statistics"] = compute_statistics(method_data["final_losses"])

        if method_data["avg_scores"]:
            method_data["score_statistics"] = compute_statistics(method_data["avg_scores"])

        if method_data["avg_kl"]:
            method_data["kl_statistics"] = compute_statistics(method_data["avg_kl"])

        if method_data["downstream_accuracy"]:
            method_data["downstream_statistics"] = compute_statistics(method_data["downstream_accuracy"])

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Method':<25} {'Downstream Acc':>15} {'Avg Score':>12} {'Avg KL':>10}")
    logger.info("-" * 70)

    for method in methods:
        method_data = results["methods"][method]

        downstream = method_data.get("downstream_statistics", {})
        score = method_data.get("score_statistics", {})
        kl = method_data.get("kl_statistics", {})

        acc_str = f"{downstream.get('mean', 0)*100:.1f}%" if downstream else "N/A"
        score_str = f"{score.get('mean', 0):.3f}" if score else "N/A"
        kl_str = f"{kl.get('mean', 0):.3f}" if kl else "N/A"

        logger.info(f"{method:<25} {acc_str:>15} {score_str:>12} {kl_str:>10}")

    logger.info("=" * 70)

    # Identify best method
    best_method = None
    best_accuracy = -1
    for method in methods:
        downstream = results["methods"][method].get("downstream_statistics", {})
        acc = downstream.get("mean", 0)
        if acc > best_accuracy:
            best_accuracy = acc
            best_method = method

    if best_method:
        logger.info(f"\nBEST METHOD: {best_method} with {best_accuracy*100:.1f}% downstream accuracy")

    # Save final results
    final_path = f"{output_dir}/final_results.json"
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"\nFinal results saved to {final_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run distribution cliff mitigation experiment"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of random seeds per method"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Training steps per run"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=50,
        help="Number of training prompts"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/cliff_mitigation",
        help="Output directory (will be suffixed with model family)"
    )
    parser.add_argument(
        "--model-family", type=str, default="qwen",
        choices=list(MODEL_FAMILIES.keys()),
        help="Model family to use: 'qwen' (4B←30B) or 'llama' (8B←70B)"
    )
    parser.add_argument(
        "--methods", nargs="+", choices=CLIFF_MITIGATION_METHODS,
        help="Which methods to run (default: all)"
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run quick test with 3 seeds, 50 steps"
    )
    parser.add_argument(
        "--no-downstream-eval", action="store_true",
        help="Skip downstream evaluation"
    )

    args = parser.parse_args()

    if args.quick_test:
        args.n_seeds = 3
        args.max_steps = 50
        args.methods = ["on_policy_gkd", "mixture", "replay_buffer"]
        logger.info("Running QUICK TEST mode")

    run_cliff_mitigation_experiment(
        n_seeds=args.n_seeds,
        max_steps=args.max_steps,
        num_prompts=args.num_prompts,
        output_dir=args.output_dir,
        model_family=args.model_family,
        methods=args.methods,
        run_downstream_eval=not args.no_downstream_eval,
    )
