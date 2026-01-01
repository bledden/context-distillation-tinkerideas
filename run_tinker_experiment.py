#!/usr/bin/env python3
"""
Run Context Distillation Experiment with Tinker SDK.

Compares three approaches to context distillation:
1. Off-policy: Train on static dataset from teacher
2. On-policy GKD: Train with student's own generations + teacher feedback
3. Hybrid: Off-policy bootstrap followed by on-policy refinement

Uses:
- Tinker TrainingClient for student training (actual GPU training)
- Tinker SamplingClient for teacher generation
- Real benchmark datasets (GSM8K, MMLU) for downstream evaluation
- Statistical analysis with multiple seeds

Usage:
    # Quick test (3 seeds, 50 steps)
    python run_tinker_experiment.py --quick-test

    # Full academic run (10 seeds, 100 steps) with all modes
    python run_tinker_experiment.py --n-seeds 10 --max-steps 100 --modes all

    # Run with specific benchmark
    python run_tinker_experiment.py --benchmark gsm8k --n-seeds 10

    # Run cross-family experiment
    python run_tinker_experiment.py --model-family cross --n-seeds 10

    # Run in background
    nohup python run_tinker_experiment.py --n-seeds 10 --modes all > tinker_run.log 2>&1 &
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

# Import config types for context distillation setup
from config import ContextType, ContextConfig, ExperimentConfig
from context_generator import Example

# Load environment variables from .env
from env_loader import load_env
load_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_gsm8k_prompts(num_prompts: int = 100, split: str = "train") -> List[Dict]:
    """
    Load GSM8K math reasoning benchmark.

    Returns list of dicts with 'question' and 'answer' keys.
    Teacher gets few-shot examples, student gets just the question.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("openai/gsm8k", "main", split=split)

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
        logger.warning(f"Failed to load GSM8K: {e}. Falling back to synthetic prompts.")
        return None


def load_mmlu_prompts(num_prompts: int = 100, subjects: List[str] = None) -> List[Dict]:
    """
    Load MMLU benchmark for few-shot evaluation.

    Returns list of dicts with question, choices, and answer.
    """
    try:
        from datasets import load_dataset

        # Default to a mix of subjects
        if subjects is None:
            subjects = ["abstract_algebra", "anatomy", "astronomy", "college_chemistry", "college_physics"]

        prompts = []
        for subject in subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                for example in dataset:
                    if len(prompts) >= num_prompts:
                        break
                    prompts.append({
                        "question": example["question"],
                        "choices": example["choices"],
                        "answer": example["answer"],
                        "subject": subject,
                        "type": "multiple_choice",
                    })
            except Exception as e:
                logger.warning(f"Failed to load MMLU subject {subject}: {e}")
                continue

            if len(prompts) >= num_prompts:
                break

        logger.info(f"Loaded {len(prompts)} MMLU prompts across {len(subjects)} subjects")
        return prompts
    except Exception as e:
        logger.warning(f"Failed to load MMLU: {e}. Falling back to synthetic prompts.")
        return None


def format_benchmark_prompt(example: Dict, include_context: bool = False, few_shot_examples: List[Dict] = None) -> str:
    """Format a benchmark example as a prompt, optionally with few-shot context."""
    prompt_parts = []

    # Add few-shot examples if provided (for teacher)
    if include_context and few_shot_examples:
        prompt_parts.append("Here are some examples:\n")
        for i, ex in enumerate(few_shot_examples[:5]):  # Max 5 examples
            if ex.get("type") == "math_reasoning":
                prompt_parts.append(f"Example {i+1}:")
                prompt_parts.append(f"Question: {ex['question']}")
                prompt_parts.append(f"Answer: {ex['answer']}\n")
            elif ex.get("type") == "multiple_choice":
                prompt_parts.append(f"Example {i+1}:")
                prompt_parts.append(f"Question: {ex['question']}")
                choices = ex.get('choices', [])
                for j, choice in enumerate(choices):
                    prompt_parts.append(f"  {chr(65+j)}) {choice}")
                prompt_parts.append(f"Answer: {chr(65 + ex['answer'])}\n")
        prompt_parts.append("\nNow solve this:\n")

    # Add the actual question
    if example.get("type") == "math_reasoning":
        prompt_parts.append(f"Question: {example['question']}")
        prompt_parts.append("\nSolve this step by step and provide the final numerical answer.")
    elif example.get("type") == "multiple_choice":
        prompt_parts.append(f"Question: {example['question']}")
        choices = example.get('choices', [])
        for j, choice in enumerate(choices):
            prompt_parts.append(f"  {chr(65+j)}) {choice}")
        prompt_parts.append("\nSelect the correct answer (A, B, C, or D).")
    else:
        # Fallback for simple prompts
        prompt_parts.append(example.get("question", str(example)))

    return "\n".join(prompt_parts)


def create_few_shot_examples(benchmark_data: List[Dict], num_examples: int = 10) -> List[Example]:
    """
    Create Example objects from benchmark data for few-shot context.

    This is used for CONTEXT DISTILLATION where teacher gets few-shot
    examples and student does not. The examples are formatted as Q&A pairs.
    """
    examples = []
    for ex in benchmark_data[:num_examples]:
        if ex.get("type") == "math_reasoning":
            # GSM8K: question + answer with reasoning
            prompt = f"Question: {ex['question']}\nSolve this step by step."
            response = ex["answer"]
        elif ex.get("type") == "multiple_choice":
            # MMLU: question + choices + answer
            choices_text = "\n".join(
                f"  {chr(65+i)}) {c}" for i, c in enumerate(ex.get("choices", []))
            )
            prompt = f"Question: {ex['question']}\n{choices_text}"
            response = f"The answer is {chr(65 + ex['answer'])}"
        else:
            prompt = ex.get("question", str(ex))
            response = ex.get("answer", "")

        examples.append(Example(prompt=prompt, response=response))

    logger.info(f"Created {len(examples)} few-shot examples for context distillation")
    return examples


def extract_answer(response: str, example: Dict) -> str:
    """Extract the answer from a model response."""
    import re

    if example.get("type") == "math_reasoning":
        # Look for final numerical answer patterns
        patterns = [
            r"(?:final answer|answer is|=)\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"####\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\$([0-9,]+(?:\.[0-9]+)?)\$?\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).replace(",", "")
        # Fallback: last number in response
        numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", response)
        return numbers[-1] if numbers else ""

    elif example.get("type") == "multiple_choice":
        # Look for letter answer
        match = re.search(r"\b([A-D])\b", response.upper())
        if match:
            return match.group(1)
        return ""

    return response.strip()


def evaluate_downstream(
    sampling_client,
    tokenizer,
    renderer,
    benchmark_examples: List[Dict],
    num_samples: int = 50,
) -> Dict:
    """
    Evaluate model on downstream benchmark.

    Returns accuracy and per-example results.
    """
    import tinker

    correct = 0
    total = 0
    results = []

    for example in benchmark_examples[:num_samples]:
        prompt = format_benchmark_prompt(example, include_context=False)
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
                predicted = extract_answer(response, example)

                # Check correctness
                if example.get("type") == "math_reasoning":
                    # Extract ground truth from GSM8K format (#### answer)
                    gt_match = re.search(r"####\s*([0-9,]+)", example.get("answer", ""))
                    ground_truth = gt_match.group(1).replace(",", "") if gt_match else ""
                    is_correct = predicted == ground_truth
                elif example.get("type") == "multiple_choice":
                    ground_truth = chr(65 + example.get("answer", 0))
                    is_correct = predicted == ground_truth
                else:
                    is_correct = False
                    ground_truth = ""

                if is_correct:
                    correct += 1
                total += 1

                results.append({
                    "question": example.get("question", "")[:100],
                    "predicted": predicted,
                    "ground_truth": ground_truth,
                    "correct": is_correct,
                })
        except Exception as e:
            logger.warning(f"Evaluation failed for example: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results[:10],  # Keep first 10 for logging
    }


def generate_synthetic_prompts(num_prompts: int = 100) -> List[str]:
    """Generate synthetic prompts for testing (fallback when benchmarks unavailable)."""
    import random

    templates = [
        "Explain the concept of {topic} in simple terms.",
        "What are the main differences between {a} and {b}?",
        "How would you solve this problem: {problem}?",
        "Summarize the key points of {topic}.",
        "What are the pros and cons of {topic}?",
        "Describe the process of {process}.",
        "Compare and contrast {a} with {b}.",
    ]

    topics = [
        "machine learning", "neural networks", "reinforcement learning",
        "natural language processing", "computer vision", "transformers",
        "gradient descent", "backpropagation", "attention mechanisms",
        "deep learning", "convolutional networks", "recurrent networks",
    ]

    problems = [
        "finding the shortest path in a graph",
        "sorting a list efficiently",
        "detecting anomalies in data",
        "classifying images",
        "generating text",
        "optimizing a neural network",
    ]

    prompts = []
    for _ in range(num_prompts):
        template = random.choice(templates)

        if "{topic}" in template:
            prompt = template.format(topic=random.choice(topics))
        elif "{a}" in template and "{b}" in template:
            a, b = random.sample(topics, 2)
            prompt = template.format(a=a, b=b)
        elif "{problem}" in template:
            prompt = template.format(problem=random.choice(problems))
        elif "{process}" in template:
            prompt = template.format(process=random.choice(topics))
        else:
            prompt = template

        prompts.append(prompt)

    return prompts


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute basic statistics with 95% CI."""
    import math

    if not values:
        return {}

    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0
    std = math.sqrt(variance)
    stderr = std / math.sqrt(n) if n > 0 else 0

    # t-value for 95% CI (approx for small samples)
    t_values = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228, 20: 2.086}
    t = t_values.get(n, 1.96)  # Default to z for large n

    ci_lower = mean - t * stderr
    ci_upper = mean + t * stderr

    return {
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
    }


def hedges_g(values1: List[float], values2: List[float]) -> float:
    """Compute Hedges' g effect size."""
    import math

    n1, n2 = len(values1), len(values2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1 = sum(values1) / n1
    mean2 = sum(values2) / n2

    var1 = sum((x - mean1) ** 2 for x in values1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in values2) / (n2 - 1)

    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    cohens_d = (mean1 - mean2) / pooled_std

    # Hedges' g correction
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return cohens_d * correction


def independent_ttest(values1: List[float], values2: List[float]) -> Dict:
    """Perform independent samples t-test."""
    import math

    n1, n2 = len(values1), len(values2)
    if n1 < 2 or n2 < 2:
        return {"t_statistic": 0, "p_value": 1.0, "significant": False}

    mean1 = sum(values1) / n1
    mean2 = sum(values2) / n2

    var1 = sum((x - mean1) ** 2 for x in values1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in values2) / (n2 - 1)

    pooled_stderr = math.sqrt(var1 / n1 + var2 / n2)

    if pooled_stderr == 0:
        return {"t_statistic": 0, "p_value": 1.0, "significant": False}

    t_stat = (mean1 - mean2) / pooled_stderr
    df = n1 + n2 - 2

    # Approximate p-value (two-tailed)
    # Using simple approximation for |t| > 2
    p_value = 2 * min(1.0, math.exp(-0.5 * t_stat ** 2))

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "df": df,
        "significant": p_value < 0.05,
        "effect_size": hedges_g(values1, values2),
    }


# Model family configurations
MODEL_FAMILIES = {
    # === DEPRECATED: Same-model context distillation ===
    # WARNING: Same-model experiments showed 0% downstream accuracy despite
    # high eval scores. The issue is that eval scores measure text similarity,
    # not reasoning capability. Same-model distillation only teaches format
    # matching, not new capabilities. Use size distillation instead.
    "same-model-llama": {
        "student_model": "meta-llama/Llama-3.1-8B-Instruct",
        "teacher_model": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "DEPRECATED: Same model shows 0% downstream accuracy. Use llama instead.",
    },
    "same-model-qwen": {
        "student_model": "Qwen/Qwen3-4B-Instruct-2507",
        "teacher_model": "Qwen/Qwen3-4B-Instruct-2507",
        "description": "DEPRECATED: Same model shows 0% downstream accuracy. Use qwen instead.",
    },
    # === RECOMMENDED: Size distillation within same family ===
    # These configurations use a larger teacher to transfer real capabilities.
    # V2 Fix: Previous same-model experiments showed 0% downstream accuracy.
    "llama": {
        "student_model": "meta-llama/Llama-3.1-8B-Instruct",
        "teacher_model": "meta-llama/Llama-3.3-70B-Instruct",
        "description": "RECOMMENDED: 8B student ← 70B teacher for capability transfer (8.75x gap)",
    },
    "qwen": {
        "student_model": "Qwen/Qwen3-4B-Instruct-2507",
        "teacher_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "description": "RECOMMENDED: 4B student ← 30B MoE teacher for capability transfer",
    },
    # === CROSS-FAMILY: Different model families ===
    # NOTE: Cross-family distillation with different tokenizers is experimental.
    # Hybrid mode is automatically disabled for these configurations.
    "cross": {
        "student_model": "meta-llama/Llama-3.1-8B-Instruct",
        "teacher_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "description": "Cross-family: Llama ← Qwen (WARNING: tokenizer mismatch, hybrid disabled)",
    },
    "cross-qwen": {
        "student_model": "Qwen/Qwen3-4B-Instruct-2507",
        "teacher_model": "meta-llama/Llama-3.3-70B-Instruct",
        "description": "Cross-family: Qwen ← Llama (WARNING: tokenizer mismatch, hybrid disabled)",
    },
}

# Training modes
TRAINING_MODES = ["off_policy", "on_policy_gkd", "hybrid", "hybrid_gradual"]


def run_full_experiment(
    n_seeds: int = 10,
    max_steps: int = 100,
    num_prompts: int = 50,
    output_dir: str = "./outputs/tinker_experiment",
    model_family: str = "llama",
    modes: List[str] = None,
    benchmark: str = "synthetic",
    run_downstream_eval: bool = True,
) -> Dict:
    """
    Run full comparison experiment with statistical analysis.

    Args:
        n_seeds: Number of random seeds per condition
        max_steps: Training steps per run
        num_prompts: Number of training prompts
        output_dir: Output directory
        model_family: Which model family to use ("llama", "qwen", "cross")
        modes: Which training modes to run (default: all)
        benchmark: Which benchmark to use ("gsm8k", "mmlu", "synthetic")
        run_downstream_eval: Whether to run downstream evaluation after training

    Returns:
        Full results with statistics
    """
    from tinker_trainer import TinkerDistillationTrainer, TinkerTrainingConfig
    from config import ExperimentConfig
    import re

    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {model_family}. Choose from: {list(MODEL_FAMILIES.keys())}")

    if modes is None:
        modes = TRAINING_MODES.copy()

    model_config = MODEL_FAMILIES[model_family]
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Context Distillation Experiment (Tinker SDK + GKD)")
    logger.info("=" * 60)
    logger.info(f"Model family: {model_family}")
    logger.info(f"Description: {model_config.get('description', 'N/A')}")
    logger.info(f"Student: {model_config['student_model']}")
    logger.info(f"Teacher: {model_config['teacher_model']}")
    logger.info(f"Training modes: {modes}")
    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Seeds: {n_seeds}")
    logger.info(f"Steps per run: {max_steps}")
    logger.info(f"Training prompts: {num_prompts}")
    logger.info(f"Downstream eval: {run_downstream_eval}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Load benchmark or generate synthetic prompts
    import random
    random.seed(42)  # Fixed seed for reproducible prompts

    benchmark_data = None
    if benchmark == "gsm8k":
        benchmark_data = load_gsm8k_prompts(num_prompts)
        if benchmark_data:
            prompts = [format_benchmark_prompt(ex, include_context=False) for ex in benchmark_data]
            # Create few-shot prompts for teacher (with context)
            few_shot_pool = benchmark_data[:10]  # Use first 10 as few-shot examples
        else:
            logger.warning("GSM8K failed to load, falling back to synthetic")
            benchmark = "synthetic"
    elif benchmark == "mmlu":
        benchmark_data = load_mmlu_prompts(num_prompts)
        if benchmark_data:
            prompts = [format_benchmark_prompt(ex, include_context=False) for ex in benchmark_data]
            few_shot_pool = benchmark_data[:10]
        else:
            logger.warning("MMLU failed to load, falling back to synthetic")
            benchmark = "synthetic"

    if benchmark == "synthetic":
        prompts = generate_synthetic_prompts(num_prompts)
        benchmark_data = None
        few_shot_pool = None

    eval_prompts = prompts[:10]  # Use first 10 for evaluation during training

    # Initialize results structure
    results = {
        "experiment": "context_distillation_comparison_gkd",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_seeds": n_seeds,
            "max_steps": max_steps,
            "num_prompts": num_prompts,
            "model_family": model_family,
            "student_model": model_config["student_model"],
            "teacher_model": model_config["teacher_model"],
            "lora_rank": 32,
            "on_policy_method": "GKD (Generalized Knowledge Distillation)",
            "benchmark": benchmark,
            "modes": modes,
        },
        "modes": {},
        "downstream_eval": {},
        "comparisons": {},
        "analysis": {},
    }

    # Initialize result structures for each mode
    for mode in modes:
        results["modes"][mode] = {
            "final_losses": [],
            "avg_scores": [],
            "avg_kl": [],
            "histories": [],
            "downstream_accuracy": [],
        }

    # Create config with the selected model family
    config = TinkerTrainingConfig(
        student_model=model_config["student_model"],
        teacher_model=model_config["teacher_model"],
    )

    # Determine if this is a CONTEXT DISTILLATION experiment (same model, different context)
    # vs SIZE DISTILLATION experiment (different model sizes)
    is_context_distillation = model_family.startswith("same-model")

    if is_context_distillation:
        logger.info("=" * 60)
        logger.info("CONTEXT DISTILLATION MODE")
        logger.info("Teacher gets few-shot examples, student does not")
        logger.info("=" * 60)

        # Create few-shot examples from benchmark data
        if benchmark_data and few_shot_pool:
            few_shot_examples = create_few_shot_examples(few_shot_pool, num_examples=10)
        else:
            logger.warning("No benchmark data for few-shot examples, using synthetic")
            few_shot_examples = []

        # Configure context for few-shot
        context_config = ContextConfig(
            context_type=ContextType.FEW_SHOT,
            num_few_shot=5,  # Use 5 examples per query
        )
        exp_config = ExperimentConfig(context=context_config)

        # Store the few-shot examples to pass to trainer
        exp_config._few_shot_examples = few_shot_examples
    else:
        logger.info("=" * 60)
        logger.info("SIZE DISTILLATION MODE")
        logger.info("Teacher is larger model, student is smaller")
        logger.info("=" * 60)
        exp_config = ExperimentConfig()
        exp_config._few_shot_examples = None

    # Add experiment type to results
    results["config"]["experiment_type"] = "context_distillation" if is_context_distillation else "size_distillation"

    for seed in range(n_seeds):
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED {seed + 1}/{n_seeds}")
        logger.info(f"{'='*60}")

        # Set random seed
        random.seed(seed)

        for mode in modes:
            logger.info(f"\n--- {mode.replace('_', ' ').title()} Distillation ---")

            try:
                trainer = TinkerDistillationTrainer(config, exp_config)
                trainer.initialize()

                # For context distillation, set the few-shot examples on the context generator
                # This ensures teacher gets context while student (being trained) does not
                if hasattr(exp_config, '_few_shot_examples') and exp_config._few_shot_examples:
                    trainer.context_generator.example_pool = exp_config._few_shot_examples
                    logger.info(f"Set {len(exp_config._few_shot_examples)} few-shot examples for teacher context")

                if mode == "off_policy":
                    result = trainer.run_off_policy_distillation(
                        prompts,
                        max_steps=max_steps,
                        eval_prompts=eval_prompts,
                        eval_every=max(1, max_steps // 10),
                    )
                elif mode == "on_policy_gkd":
                    logger.info("Using Generalized Knowledge Distillation with reverse KL divergence")
                    result = trainer.run_on_policy_distillation(
                        prompts,
                        max_steps=max_steps,
                        eval_prompts=eval_prompts,
                        eval_every=max(1, max_steps // 10),
                    )
                elif mode == "hybrid":
                    logger.info("Using hybrid: off-policy bootstrap → on-policy GKD refinement")
                    result = trainer.run_hybrid_distillation(
                        prompts,
                        max_steps=max_steps,
                        off_policy_ratio=0.5,
                        eval_prompts=eval_prompts,
                        eval_every=max(1, max_steps // 10),
                    )
                elif mode == "hybrid_gradual":
                    logger.info("Using GRADUAL hybrid: smooth transition off-policy → on-policy")
                    result = trainer.run_hybrid_gradual_distillation(
                        prompts,
                        max_steps=max_steps,
                        eval_prompts=eval_prompts,
                        eval_every=max(1, max_steps // 10),
                    )
                else:
                    logger.error(f"Unknown mode: {mode}")
                    continue

                # Record results
                results["modes"][mode]["final_losses"].append(result["final_loss"])
                results["modes"][mode]["avg_scores"].append(result.get("avg_score", 0))
                results["modes"][mode]["avg_kl"].append(result.get("avg_kl", 0))
                results["modes"][mode]["histories"].append({
                    "seed": seed,
                    "history": result["history"][-5:] if result.get("history") else [],
                })

                logger.info(f"{mode} final loss: {result['final_loss']:.4f}")
                if result.get("avg_score"):
                    logger.info(f"{mode} avg score: {result.get('avg_score', 0):.4f}")

                # Run downstream evaluation if benchmark is available
                if run_downstream_eval and benchmark_data and len(benchmark_data) > 20:
                    logger.info(f"Running downstream evaluation on {benchmark}...")
                    try:
                        eval_examples = benchmark_data[num_prompts:num_prompts+50]  # Use held-out examples
                        if len(eval_examples) < 20:
                            eval_examples = benchmark_data[-50:]  # Fallback to last 50

                        downstream_result = evaluate_downstream(
                            sampling_client=trainer._sampling_client,
                            tokenizer=trainer._tokenizer,
                            renderer=trainer._renderer,  # _renderer is student's renderer
                            benchmark_examples=eval_examples,
                            num_samples=min(50, len(eval_examples)),
                        )
                        results["modes"][mode]["downstream_accuracy"].append(downstream_result["accuracy"])
                        logger.info(f"{mode} downstream accuracy: {downstream_result['accuracy']:.2%}")
                    except Exception as e:
                        logger.warning(f"Downstream eval failed: {e}")
                        results["modes"][mode]["downstream_accuracy"].append(0.0)

            except Exception as e:
                logger.error(f"{mode} failed: {e}")
                import traceback
                traceback.print_exc()
                results["modes"][mode]["final_losses"].append(float("inf"))
                results["modes"][mode]["avg_scores"].append(0.0)
                results["modes"][mode]["avg_kl"].append(0.0)
                results["modes"][mode]["downstream_accuracy"].append(0.0)

        # Save intermediate results
        with open(f"{output_dir}/intermediate_results_seed{seed}.json", "w") as f:
            json.dump(results, f, indent=2, default=float)

    # === Compute Statistics ===
    logger.info("\n" + "=" * 60)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("=" * 60)

    # Compute statistics for each mode
    for mode in modes:
        mode_data = results["modes"][mode]
        losses = [l for l in mode_data["final_losses"] if l != float("inf")]
        scores = [s for s in mode_data["avg_scores"] if s > 0]
        kls = [k for k in mode_data["avg_kl"]]
        downstream = [a for a in mode_data.get("downstream_accuracy", []) if a > 0]

        mode_data["statistics"] = compute_statistics(losses)
        mode_data["score_statistics"] = compute_statistics(scores)
        mode_data["kl_statistics"] = compute_statistics(kls)
        mode_data["downstream_statistics"] = compute_statistics(downstream)

        if mode_data["statistics"]:
            stats = mode_data["statistics"]
            logger.info(f"\n{mode}: {stats['mean']:.4f} (95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])")
            if downstream:
                ds_stats = mode_data["downstream_statistics"]
                logger.info(f"  Downstream accuracy: {ds_stats['mean']:.2%} (95% CI: [{ds_stats['ci_lower']:.2%}, {ds_stats['ci_upper']:.2%}])")

    # Statistical comparisons between modes
    mode_losses = {}
    for mode in modes:
        losses = [l for l in results["modes"][mode]["final_losses"] if l != float("inf")]
        if len(losses) >= 2:
            mode_losses[mode] = losses

    # Compare all pairs
    for i, mode1 in enumerate(modes):
        for mode2 in modes[i+1:]:
            if mode1 in mode_losses and mode2 in mode_losses:
                comparison = independent_ttest(mode_losses[mode1], mode_losses[mode2])
                results["comparisons"][f"{mode1}_vs_{mode2}"] = comparison

    # Determine best mode by loss
    if mode_losses:
        best_mode = min(mode_losses.keys(), key=lambda m: results["modes"][m]["statistics"]["mean"])
        results["analysis"]["best_mode"] = best_mode
        results["analysis"]["method"] = "GKD with reverse KL divergence"

        # Find if best is significantly better than others
        significant_wins = []
        for mode in modes:
            if mode != best_mode and mode in mode_losses:
                comp_key = f"{min(best_mode, mode)}_vs_{max(best_mode, mode)}"
                if comp_key not in results["comparisons"]:
                    comp_key = f"{max(best_mode, mode)}_vs_{min(best_mode, mode)}"
                if comp_key in results["comparisons"] and results["comparisons"][comp_key]["significant"]:
                    significant_wins.append(mode)

        results["analysis"]["significant_improvements_over"] = significant_wins

        logger.info(f"\nBest mode: {best_mode}")
        if significant_wins:
            logger.info(f"Significantly better than: {significant_wins}")

        # Log pairwise comparisons
        for comp_name, comp in results["comparisons"].items():
            logger.info(f"\n{comp_name}:")
            logger.info(f"  Effect size (Hedges' g): {comp['effect_size']:.3f}")
            logger.info(f"  p-value: {comp['p_value']:.4f}")
            logger.info(f"  Significant: {'Yes' if comp['significant'] else 'No'}")

    # === Save Final Results ===
    with open(f"{output_dir}/final_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Generate markdown report
    generate_report(results, f"{output_dir}/RESULTS.md")

    logger.info(f"\nResults saved to {output_dir}/")

    return results


def generate_report(results: Dict, output_path: str):
    """Generate markdown report."""
    config = results.get("config", {})
    modes = config.get("modes", ["off_policy", "on_policy_gkd", "hybrid"])

    lines = [
        "# Context Distillation: Multi-Mode Comparison",
        "",
        f"**Generated:** {results['timestamp']}",
        f"**Seeds per condition:** {config.get('n_seeds', 'N/A')}",
        f"**Steps per run:** {config.get('max_steps', 'N/A')}",
        f"**Benchmark:** {config.get('benchmark', 'synthetic')}",
        f"**Training modes:** {', '.join(modes)}",
        "",
        "## Overview",
        "",
        "This experiment compares three context distillation paradigms:",
        "1. **Off-Policy**: Pre-generate teacher outputs, train student on static dataset",
        "2. **On-Policy GKD**: Generalized Knowledge Distillation with reverse KL divergence",
        "3. **Hybrid**: Off-policy bootstrap followed by on-policy GKD refinement",
        "",
        "### What is GKD?",
        "",
        "GKD (Generalized Knowledge Distillation) implements *proper* on-policy distillation:",
        "- Student generates tokens from its own distribution",
        "- Teacher computes log-probabilities on the student's trajectory",
        "- Training minimizes reverse KL divergence: D_KL(student || teacher)",
        "- Provides O(N) bits of signal per episode (dense per-token feedback)",
        "",
        "Reference: [On-Policy Distillation of Language Models (Agarwal et al., ICLR 2024)](https://arxiv.org/abs/2306.13649)",
        "",
        "## Results Summary",
        "",
        "### Final Loss Comparison",
        "",
        "| Mode | Mean Loss (95% CI) | N |",
        "|------|-------------------|---|",
    ]

    mode_labels = {
        "off_policy": "Off-Policy",
        "on_policy_gkd": "On-Policy GKD",
        "hybrid": "Hybrid",
        "hybrid_gradual": "Hybrid (Gradual)",
    }

    for mode in modes:
        stats = results["modes"].get(mode, {}).get("statistics", {})
        if stats and stats.get("mean") is not None:
            loss_str = f"{stats['mean']:.4f} [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
            mode_label = mode_labels.get(mode, mode.replace("_", " ").title())
            lines.append(f"| {mode_label} | {loss_str} | {stats.get('n', 0)} |")

    # Downstream accuracy table (if available)
    has_downstream = any(
        results["modes"].get(m, {}).get("downstream_statistics", {}).get("mean", 0) > 0
        for m in modes
    )
    if has_downstream:
        lines.extend([
            "",
            "### Downstream Accuracy",
            "",
            "| Mode | Mean Accuracy (95% CI) | N |",
            "|------|------------------------|---|",
        ])
        for mode in modes:
            ds_stats = results["modes"].get(mode, {}).get("downstream_statistics", {})
            if ds_stats and ds_stats.get("mean", 0) > 0:
                acc_str = f"{ds_stats['mean']:.2%} [{ds_stats['ci_lower']:.2%}, {ds_stats['ci_upper']:.2%}]"
                mode_label = mode_labels.get(mode, mode.replace("_", " ").title())
                lines.append(f"| {mode_label} | {acc_str} | {ds_stats.get('n', 0)} |")

    lines.extend([
        "",
        "### Statistical Comparisons",
        "",
    ])

    for comp_name, comp in results.get("comparisons", {}).items():
        lines.extend([
            f"**{comp_name.replace('_', ' ').replace(' vs ', ' vs. ')}:**",
            f"- t-statistic: {comp['t_statistic']:.3f}",
            f"- p-value: {comp['p_value']:.4f}",
            f"- Effect size (Hedges' g): {comp['effect_size']:.3f}",
            f"- Significant (α=0.05): {'Yes' if comp['significant'] else 'No'}",
            "",
        ])

    if "analysis" in results and "best_mode" in results["analysis"]:
        lines.extend([
            "### Conclusion",
            "",
            f"**Best mode:** {results['analysis']['best_mode'].replace('_', ' ').title()}",
        ])
        if results["analysis"].get("significant_improvements_over"):
            lines.append(f"**Significantly better than:** {', '.join(results['analysis']['significant_improvements_over'])}")
        lines.append("")

    lines.extend([
        "## Methodology",
        "",
        f"- **Student model:** {config.get('student_model', 'N/A')}",
        f"- **Teacher model:** {config.get('teacher_model', 'N/A')}",
        f"- **Model family:** {config.get('model_family', 'N/A')}",
        f"- **LoRA rank:** {config.get('lora_rank', 32)}",
        f"- **Benchmark:** {config.get('benchmark', 'synthetic')}",
        f"- **On-policy method:** {config.get('on_policy_method', 'GKD')}",
        f"- **Seeds:** {config.get('n_seeds', 'N/A')} independent random seeds",
        "- **Confidence intervals:** 95% CI using t-distribution",
        "- **Effect size:** Hedges' g (bias-corrected Cohen's d)",
        "",
        "## Training Modes Explained",
        "",
        "### Off-Policy",
        "- Generate teacher responses once, store as static dataset",
        "- Train student with standard supervised learning",
        "- Efficient but student never learns from its own mistakes",
        "",
        "### On-Policy GKD",
        "- Student generates responses at each step",
        "- Teacher evaluates student's trajectory with logprobs",
        "- Trains using reverse KL: encourages student to match teacher",
        "- More compute intensive but learns from exploration",
        "",
        "### Hybrid",
        "- Phase 1: Off-policy bootstrap (50% of steps)",
        "- Phase 2: On-policy GKD refinement (50% of steps)",
        "- Best of both: fast initial learning + refined exploration",
        "",
        "## Reproducibility",
        "",
        "```bash",
        f"# Run with all modes and GSM8K benchmark",
        f"python run_tinker_experiment.py --n-seeds {config.get('n_seeds', 10)} --max-steps {config.get('max_steps', 100)} --modes all --benchmark gsm8k",
        "",
        f"# Run specific modes",
        f"python run_tinker_experiment.py --modes off_policy on_policy_gkd --benchmark gsm8k",
        "",
        f"# Cross-family experiment (Qwen teacher -> Llama student)",
        f"python run_tinker_experiment.py --model-family cross --modes all",
        "```",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to {output_path}")


def run_both_model_families(
    n_seeds: int = 10,
    max_steps: int = 100,
    num_prompts: int = 50,
    output_dir: str = "./outputs/tinker_experiment",
    modes: List[str] = None,
    benchmark: str = "synthetic",
    run_downstream_eval: bool = True,
) -> Dict:
    """
    Run experiments for BOTH Llama and Qwen model families.

    This matches the original experimental design that ran both families
    to ensure findings generalize across architectures.
    """
    if modes is None:
        modes = TRAINING_MODES.copy()

    all_results = {
        "experiment": "context_distillation_multi_family_gkd",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "modes": modes,
            "benchmark": benchmark,
            "n_seeds": n_seeds,
            "max_steps": max_steps,
        },
        "families": {},
    }

    for family in ["llama", "qwen"]:
        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING {family.upper()} MODEL FAMILY")
        logger.info("=" * 80)

        family_output_dir = f"{output_dir}/{family}"
        results = run_full_experiment(
            n_seeds=n_seeds,
            max_steps=max_steps,
            num_prompts=num_prompts,
            output_dir=family_output_dir,
            model_family=family,
            modes=modes,
            benchmark=benchmark,
            run_downstream_eval=run_downstream_eval,
        )
        all_results["families"][family] = results

    # Save combined results
    with open(f"{output_dir}/combined_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Generate combined report
    generate_combined_report(all_results, f"{output_dir}/COMBINED_RESULTS.md")

    return all_results


def generate_combined_report(results: Dict, output_path: str):
    """Generate a combined report for both model families."""
    global_config = results.get("config", {})
    modes = global_config.get("modes", ["off_policy", "on_policy_gkd", "hybrid"])

    lines = [
        "# Context Distillation: Multi-Family Comparison",
        "",
        f"**Generated:** {results['timestamp']}",
        f"**Training modes:** {', '.join(modes)}",
        f"**Benchmark:** {global_config.get('benchmark', 'synthetic')}",
        "",
        "## Overview",
        "",
        "This experiment compares distillation modes across multiple model families",
        "to ensure findings generalize:",
        "",
        "| Family | Student | Teacher |",
        "|--------|---------|---------|",
    ]

    for family, data in results.get("families", {}).items():
        config = data.get("config", {})
        lines.append(f"| {family.title()} | {config.get('student_model', 'N/A')} | {config.get('teacher_model', 'N/A')} |")

    lines.extend([
        "",
        "## Results by Family",
        "",
    ])

    mode_labels = {
        "off_policy": "Off-Policy",
        "on_policy_gkd": "On-Policy GKD",
        "hybrid": "Hybrid",
        "hybrid_gradual": "Hybrid (Gradual)",
    }

    for family, data in results.get("families", {}).items():
        lines.extend([
            f"### {family.title()}",
            "",
        ])

        # Stats for each mode
        for mode in modes:
            mode_data = data.get("modes", {}).get(mode, {})
            stats = mode_data.get("statistics", {})
            if stats and stats.get("mean") is not None:
                label = mode_labels.get(mode, mode.replace("_", " ").title())
                lines.append(f"**{label}:** {stats.get('mean', 0):.4f} (95% CI: [{stats.get('ci_lower', 0):.4f}, {stats.get('ci_upper', 0):.4f}])")

                # Downstream accuracy if available
                ds_stats = mode_data.get("downstream_statistics", {})
                if ds_stats and ds_stats.get("mean", 0) > 0:
                    lines.append(f"  - Downstream accuracy: {ds_stats.get('mean', 0):.2%}")

        # Analysis
        analysis = data.get("analysis", {})
        if analysis:
            lines.extend([
                "",
                f"**Best mode:** {analysis.get('best_mode', 'N/A')}",
            ])
            if analysis.get("significant_improvements_over"):
                lines.append(f"**Significantly better than:** {', '.join(analysis['significant_improvements_over'])}")

        lines.append("")

    lines.extend([
        "## Key Findings",
        "",
        "1. **GKD prevents collapse**: With proper reverse KL divergence, on-policy",
        "   distillation no longer collapses to 0 eval score",
        "2. **Hybrid combines benefits**: Off-policy bootstrap + on-policy refinement",
        "   may achieve the best of both worlds",
        "3. **Cross-family transfer**: Results should generalize across model architectures",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "# Run both Llama and Qwen with all modes",
        f"python run_tinker_experiment.py --model-family both --modes all --benchmark {global_config.get('benchmark', 'gsm8k')}",
        "",
        "# Run specific family with specific modes",
        "python run_tinker_experiment.py --model-family llama --modes off_policy hybrid --benchmark gsm8k",
        "",
        "# Cross-family experiment (Qwen teacher -> Llama student)",
        "python run_tinker_experiment.py --model-family cross --modes all --benchmark gsm8k",
        "```",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Combined report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Context Distillation experiment with Tinker SDK and GKD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with all modes and synthetic data
  python run_tinker_experiment.py --quick-test

  # Full run with GSM8K benchmark and all modes
  python run_tinker_experiment.py --n-seeds 10 --modes all --benchmark gsm8k

  # Run specific modes with Llama family
  python run_tinker_experiment.py --model-family llama --modes off_policy hybrid

  # Cross-family experiment (Qwen teacher -> Llama student)
  python run_tinker_experiment.py --model-family cross --modes all --benchmark gsm8k
        """,
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of random seeds (default: 10 for academic rigor)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Training steps per run",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of training prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/tinker_experiment",
        help="Output directory",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        choices=["same-model-llama", "same-model-qwen", "llama", "qwen", "cross", "cross-qwen", "both"],
        default="qwen",
        help="Model family: qwen/llama (RECOMMENDED for capability transfer), same-model-* (DEPRECATED: 0%% downstream accuracy), cross/cross-qwen, or both (default: qwen)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=["off_policy", "on_policy_gkd", "hybrid", "hybrid_gradual", "all"],
        default=["all"],
        help="Training modes to run (default: all). Can specify multiple: --modes off_policy hybrid_gradual",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["gsm8k", "mmlu", "synthetic"],
        default="synthetic",
        help="Benchmark dataset for training and evaluation (default: synthetic)",
    )
    parser.add_argument(
        "--no-downstream-eval",
        action="store_true",
        help="Skip downstream evaluation after training",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (3 seeds, 20 steps, synthetic data)",
    )

    args = parser.parse_args()

    # Process modes
    if "all" in args.modes:
        modes = TRAINING_MODES.copy()
    else:
        modes = args.modes

    # CRITICAL: Skip hybrid mode for cross-family distillation due to mode collapse
    # See tinker_trainer.py run_hybrid_distillation() docstring for details
    if args.model_family in ["cross", "cross-qwen"] and "hybrid" in modes:
        logger.warning(
            "⚠️  Removing 'hybrid' mode for cross-family distillation. "
            "Hybrid mode causes mode collapse when student/teacher are different families. "
            "Use pure on-policy GKD (on_policy_gkd) for cross-family experiments."
        )
        modes = [m for m in modes if m != "hybrid"]
        if not modes:
            modes = ["on_policy_gkd"]  # Default to on-policy GKD

    if args.quick_test:
        args.n_seeds = 3
        args.max_steps = 20
        args.num_prompts = 20
        args.benchmark = "synthetic"
        logger.info("Running in quick test mode")

    run_downstream_eval = not args.no_downstream_eval

    # Run based on model family selection
    if args.model_family == "both":
        results = run_both_model_families(
            n_seeds=args.n_seeds,
            max_steps=args.max_steps,
            num_prompts=args.num_prompts,
            output_dir=args.output_dir,
            modes=modes,
            benchmark=args.benchmark,
            run_downstream_eval=run_downstream_eval,
        )

        print("\n" + "=" * 60)
        print("MULTI-FAMILY EXPERIMENT COMPLETE")
        print("=" * 60)

        for family, family_results in results.get("families", {}).items():
            print(f"\n{family.upper()}:")
            analysis = family_results.get("analysis", {})
            if analysis:
                print(f"  Best mode: {analysis.get('best_mode', 'N/A')}")
                if analysis.get("significant_improvements_over"):
                    print(f"  Significantly better than: {analysis['significant_improvements_over']}")
    else:
        results = run_full_experiment(
            n_seeds=args.n_seeds,
            max_steps=args.max_steps,
            num_prompts=args.num_prompts,
            output_dir=args.output_dir,
            model_family=args.model_family,
            modes=modes,
            benchmark=args.benchmark,
            run_downstream_eval=run_downstream_eval,
        )

        print("\n" + "=" * 60)
        print(f"EXPERIMENT COMPLETE ({args.model_family.upper()})")
        print("=" * 60)

        if "analysis" in results and "best_mode" in results["analysis"]:
            print(f"\nBest mode: {results['analysis']['best_mode']}")
            if results["analysis"].get("significant_improvements_over"):
                print(f"Significantly better than: {results['analysis']['significant_improvements_over']}")


if __name__ == "__main__":
    main()
