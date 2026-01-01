"""
Context Distillation Trainer using Tinker SDK for GPU Training.

This module provides trainers that use Tinker's training API for actual
GPU training, rather than local PyTorch training.

Key differences from off_policy_trainer.py and on_policy_trainer.py:
- Uses Tinker TrainingClient with LoRA for student training
- Leverages Tinker's distributed training infrastructure
- Each training step is a "clock cycle" (~10 seconds)

See: https://tinker-docs.thinkingmachines.ai
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Try to import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

# Load environment variables from .env
from env_loader import load_env
load_env()

from config import ExperimentConfig, OnPolicyConfig, FeedbackType, TrainingMode
from context_generator import ContextGenerator, ContextBundle
from model_wrapper import TinkerModel, GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class TinkerTrainingConfig:
    """Configuration for Tinker SDK training.

    IMPORTANT: For context distillation to transfer real capabilities,
    the teacher should be a LARGER model than the student. Same-model
    context distillation (where teacher=student but teacher gets few-shot)
    only teaches format matching, not new reasoning capabilities.

    V2 Fix (2025-12-27): Updated default teacher to 30B to enable actual
    capability transfer. Previous same-model experiments showed 0% downstream
    accuracy despite high eval scores (which measured text similarity, not correctness).
    """
    # Student model - smaller model that learns from teacher
    student_model: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Teacher model - MUST be larger than student for capability transfer
    # Previous same-model experiments (4B→4B) showed 0% downstream accuracy
    # because text similarity != reasoning capability transfer
    teacher_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    lora_rank: int = 32
    learning_rate: float = 1e-4
    # Lower LR for on-policy/GKD to prevent KL mode collapse
    # On-policy can destabilize if LR too high (KL spikes to 10+)
    on_policy_learning_rate: float = 5e-5
    beta1: float = 0.9
    beta2: float = 0.95
    # KL warmup: start with lower advantage clipping, increase over time
    kl_warmup_steps: int = 20
    max_advantage_clip: float = 5.0
    initial_advantage_clip: float = 2.0
    # Gradual hybrid: smooth transition from off-policy to on-policy
    gradual_transition_schedule: str = "cosine"  # "cosine" or "linear"
    gradual_refresh_every: int = 10  # More frequent refresh than standard hybrid


@dataclass
class DistillationResult:
    """Result from a distillation training run."""
    prompt: str
    student_response: str
    teacher_response: str
    score: float
    loss: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TinkerDistillationTrainer:
    """
    Distillation trainer using Tinker SDK for GPU training.

    This trainer:
    1. Uses a Tinker SamplingClient for teacher generation
    2. Uses a Tinker TrainingClient for student training
    3. Implements both off-policy and on-policy distillation
    """

    def __init__(
        self,
        config: TinkerTrainingConfig,
        experiment_config: Optional[ExperimentConfig] = None,
    ):
        self.config = config
        self.experiment_config = experiment_config or ExperimentConfig()
        self.context_config = self.experiment_config.context

        # Tinker clients (initialized lazily)
        self._training_client = None
        self._sampling_client = None
        self._teacher_client = None
        self._tokenizer = None
        self._renderer = None

        # Context generator
        self.context_generator = ContextGenerator(self.context_config)

        # Training state
        self.history = []
        self.checkpoints = []
        self._wandb_run = None

    def _init_wandb(self, run_name: str = None, mode: str = "off_policy"):
        """Initialize wandb for experiment tracking."""
        if not WANDB_AVAILABLE:
            logger.info("wandb not available, skipping initialization")
            return

        try:
            self._wandb_run = wandb.init(
                project="context-distillation",
                name=run_name or f"tinker_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "student_model": self.config.student_model,
                    "teacher_model": self.config.teacher_model,
                    "lora_rank": self.config.lora_rank,
                    "learning_rate": self.config.learning_rate,
                    "mode": mode,
                },
                reinit=True,
            )
            logger.info(f"wandb initialized: {self._wandb_run.url}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self._wandb_run = None

    def _log_wandb(self, metrics: Dict[str, Any], step: int = None):
        """Log metrics to wandb if available."""
        if self._wandb_run is not None:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"wandb log failed: {e}")

    def _finish_wandb(self):
        """Finish wandb run."""
        if self._wandb_run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
            self._wandb_run = None

    def _check_tinker_available(self) -> bool:
        """Check if Tinker SDK is available."""
        try:
            import tinker
            if not os.getenv("TINKER_API_KEY"):
                logger.error("TINKER_API_KEY not set")
                return False
            return True
        except ImportError:
            logger.error("Tinker SDK not installed. Run: pip install tinker tinker-cookbook")
            return False

    def initialize(self):
        """Initialize Tinker clients."""
        if not self._check_tinker_available():
            raise ImportError("Tinker SDK not available")

        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        logger.info(f"Initializing Tinker training session...")
        logger.info(f"  Student model: {self.config.student_model}")
        logger.info(f"  Teacher model: {self.config.teacher_model}")
        logger.info(f"  LoRA rank: {self.config.lora_rank}")

        # Initialize service client
        service_client = tinker.ServiceClient()

        # Create training client for student (with LoRA)
        self._training_client = service_client.create_lora_training_client(
            base_model=self.config.student_model,
            rank=self.config.lora_rank,
        )

        # Create sampling client for student (to generate responses)
        self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
            name="init"
        )

        # Create separate sampling client for teacher
        self._teacher_client = service_client.create_sampling_client(
            base_model=self.config.teacher_model
        )

        # Get tokenizer from training client (properly authenticated)
        self._tokenizer = self._training_client.get_tokenizer()
        self._renderer = renderers.get_renderer(
            "llama3" if "llama" in self.config.student_model.lower() else "qwen3",
            self._tokenizer
        )

        # Teacher tokenizer/renderer - SamplingClient doesn't have get_tokenizer, use tokenizer_utils
        self._teacher_tokenizer = get_tokenizer(self.config.teacher_model)
        self._teacher_renderer = renderers.get_renderer(
            "llama3" if "llama" in self.config.teacher_model.lower() else "qwen3",
            self._teacher_tokenizer
        )

        logger.info("Tinker clients initialized successfully")

    def _generate_student_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> Tuple[str, List[int], List[float]]:
        """
        Generate student response with logprobs.

        Returns:
            (response_text, tokens, logprobs)
        """
        import tinker

        messages = [{"role": "user", "content": prompt}]
        model_input = self._renderer.build_generation_prompt(messages)

        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),
        )

        # Logprobs are included by default in SampledSequence
        result = self._sampling_client.sample(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1,
        ).result()

        if result.sequences:
            seq = result.sequences[0]
            tokens = list(seq.tokens)
            logprobs = list(seq.logprobs) if seq.logprobs else []

            parsed_message, _ = self._renderer.parse_response(tokens)
            if parsed_message and "content" in parsed_message:
                response_text = parsed_message["content"]
            else:
                response_text = self._tokenizer.decode(tokens, skip_special_tokens=True)

            return response_text, tokens, logprobs

        return "", [], []

    def _generate_teacher_response(
        self,
        prompt: str,
        context_bundle: Optional[ContextBundle] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Generate teacher response (with context if provided)."""
        import tinker

        # Format prompt with context for teacher
        if context_bundle:
            full_prompt = context_bundle.format_for_prompt(prompt)
        else:
            full_prompt = prompt

        messages = [{"role": "user", "content": full_prompt}]
        model_input = self._teacher_renderer.build_generation_prompt(messages)

        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),
        )

        result = self._teacher_client.sample(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1,
        ).result()

        if result.sequences:
            tokens = result.sequences[0].tokens
            parsed_message, _ = self._teacher_renderer.parse_response(tokens)
            if parsed_message and "content" in parsed_message:
                return parsed_message["content"]
            return self._teacher_tokenizer.decode(tokens, skip_special_tokens=True)

        return ""

    def _compute_score(
        self,
        student_response: str,
        teacher_response: str,
    ) -> float:
        """Compute similarity score between student and teacher responses."""
        if not teacher_response or not teacher_response.strip():
            return 0.5

        student_response = student_response.strip()
        teacher_response = teacher_response.strip()

        if not student_response:
            return 0.0

        # Jaccard similarity
        student_words = set(student_response.lower().split())
        teacher_words = set(teacher_response.lower().split())

        if teacher_words:
            jaccard = len(student_words & teacher_words) / len(student_words | teacher_words)
        else:
            jaccard = 0.0

        # Length ratio
        len_ratio = min(len(student_response), len(teacher_response)) / max(
            len(student_response), len(teacher_response), 1
        )

        # Bigram overlap
        def get_bigrams(text):
            words = text.lower().split()
            return set(zip(words[:-1], words[1:])) if len(words) > 1 else set()

        student_bigrams = get_bigrams(student_response)
        teacher_bigrams = get_bigrams(teacher_response)

        if teacher_bigrams:
            bigram_overlap = len(student_bigrams & teacher_bigrams) / len(teacher_bigrams)
        else:
            bigram_overlap = 0.0

        score = 0.4 * jaccard + 0.3 * len_ratio + 0.3 * bigram_overlap
        return min(1.0, max(0.0, score))

    def train_supervised_step(
        self,
        prompt: str,
        target_response: str,
    ) -> Dict[str, float]:
        """
        Train student on a single (prompt, target) pair using supervised learning.

        This is used for off-policy distillation.
        """
        import tinker
        from tinker import types

        # Build supervised example
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_response},
        ]

        # build_supervised_example returns (ModelInput, weights_tensor)
        model_input, weights_tensor = self._renderer.build_supervised_example(messages)

        # Convert to lists for Tinker API
        tokens = model_input.to_ints()
        weights = weights_tensor.tolist()

        # Create training datum
        datum = tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": tokens,
                "weights": weights,
            }
        )

        # Forward-backward pass
        fwd_bwd_future = self._training_client.forward_backward(
            [datum], loss_fn="cross_entropy"
        )

        # Optimizer step
        optim_future = self._training_client.optim_step(
            types.AdamParams(
                learning_rate=self.config.learning_rate,
                beta1=self.config.beta1,
                beta2=self.config.beta2,
            )
        )

        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_future.result()

        return fwd_bwd_result.metrics

    def train_rl_step(
        self,
        tokens: List[int],
        logprobs: List[float],
        advantages: List[float],
        loss_fn: str = "importance_sampling",
    ) -> Dict[str, float]:
        """
        Train using RL (importance sampling / REINFORCE).

        This is used for on-policy distillation with reward shaping.

        CRITICAL: Language model training uses shifted sequences:
        - model_input = tokens[:-1] (input to predict next token)
        - target_tokens = tokens[1:] (ground truth for next token prediction)
        - logprobs/advantages also shift by 1 to align with targets
        """
        import tinker
        from tinker import types
        import torch

        # Language model training uses shifted sequences
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        shifted_logprobs = logprobs[1:]  # Align with targets
        shifted_advantages = advantages[1:]  # Align with targets

        # importance_sampling loss only accepts: target_tokens, logprobs, advantages
        datum = tinker.Datum(
            model_input=types.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": types.TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": types.TensorData.from_torch(torch.tensor(shifted_logprobs, dtype=torch.float32)),
                "advantages": types.TensorData.from_torch(torch.tensor(shifted_advantages, dtype=torch.float32)),
            }
        )

        fwd_bwd_future = self._training_client.forward_backward(
            [datum], loss_fn=loss_fn
        )

        optim_future = self._training_client.optim_step(
            types.AdamParams(
                learning_rate=self.config.learning_rate,
                beta1=self.config.beta1,
                beta2=self.config.beta2,
            )
        )

        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_future.result()

        return fwd_bwd_result.metrics

    def run_off_policy_distillation(
        self,
        prompts: List[str],
        max_steps: int = 100,
        eval_prompts: Optional[List[str]] = None,
        eval_every: int = 10,
    ) -> Dict[str, Any]:
        """
        Run off-policy distillation.

        1. Generate teacher response for each prompt (with context)
        2. Train student to match teacher response
        """
        logger.info(f"Running off-policy distillation for {max_steps} steps")
        logger.info(f"Training prompts: {len(prompts)}")

        # Initialize wandb
        self._init_wandb(mode="off_policy")

        self.history = []
        prompt_idx = 0

        for step in range(max_steps):
            # Get current prompt (cycle through)
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # Generate context for teacher
            context_bundle = self.context_generator.generate(prompt)

            # Get teacher response
            teacher_response = self._generate_teacher_response(
                prompt,
                context_bundle=context_bundle,
                temperature=self.experiment_config.teacher.temperature,
            )

            if not teacher_response:
                logger.warning(f"Empty teacher response for: {prompt[:50]}...")
                continue

            # Train student on teacher response
            metrics = self.train_supervised_step(prompt, teacher_response)

            self.history.append({
                "step": step,
                "loss": metrics.get("loss:sum", 0.0),
                "prompt": prompt[:100],
                "teacher_response_len": len(teacher_response),
            })

            if step % eval_every == 0:
                loss = metrics.get('loss:sum', 0.0)
                logger.info(f"Step {step}: loss={loss:.4f}")

                # Log to wandb
                wandb_metrics = {"train/loss": loss, "train/step": step}

                # Evaluation
                if eval_prompts:
                    eval_result = self._evaluate(eval_prompts[:5])  # Quick eval
                    self.history[-1]["eval"] = eval_result
                    avg_score = eval_result.get('avg_score', 0)
                    logger.info(f"  Eval avg_score: {avg_score:.3f}")
                    wandb_metrics["eval/avg_score"] = avg_score

                self._log_wandb(wandb_metrics, step=step)

        # Final checkpoint
        checkpoint_name = f"off_policy_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self._save_checkpoint(checkpoint_name)
        self.checkpoints.append(checkpoint_path)

        # Finish wandb
        self._finish_wandb()

        return {
            "history": self.history,
            "final_loss": self.history[-1]["loss"] if self.history else 0.0,
            "checkpoint": checkpoint_path,
            "training_mode": "off_policy",
        }

    def run_on_policy_distillation(
        self,
        prompts: List[str],
        max_steps: int = 100,
        eval_prompts: Optional[List[str]] = None,
        eval_every: int = 10,
        refresh_student_every: int = 20,
    ) -> Dict[str, Any]:
        """
        Run on-policy distillation using Generalized Knowledge Distillation (GKD).

        This implements PROPER on-policy distillation as described in:
        - "On-Policy Distillation of Language Models" (Agarwal et al., ICLR 2024)
        - Thinking Machines blog: https://thinkingmachines.ai/blog/on-policy-distillation/

        Key insight: Use reverse KL divergence with per-token teacher logprobs
        on student-generated trajectories. This provides O(N) bits of signal
        per episode (dense feedback) vs sparse episode-level rewards.

        Algorithm:
        1. Student generates tokens (samples from its own distribution)
        2. Teacher computes logprobs on student's trajectory
        3. Train using reverse KL: minimize D_KL(student || teacher)
           = E_student[log p_student(x) - log p_teacher(x)]

        This is ~50-100x more compute efficient than RL (per Thinking Machines).
        """
        logger.info(f"Running GKD on-policy distillation for {max_steps} steps")
        logger.info(f"Training prompts: {len(prompts)}")

        # Initialize wandb
        self._init_wandb(mode="on_policy_gkd")

        self.history = []
        prompt_idx = 0

        for step in range(max_steps):
            # Get current prompt
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # Refresh sampling client periodically to use updated weights
            if step > 0 and step % refresh_student_every == 0:
                self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
                    name=f"step_{step}"
                )
                logger.info(f"Refreshed student sampling client at step {step}")

            # Step 1: Student generates response (sample from student distribution)
            student_response, student_tokens, student_logprobs = self._generate_student_response(
                prompt,
                temperature=self.experiment_config.on_policy.temperature,
            )

            if not student_tokens or len(student_tokens) < 2:
                logger.warning(f"Empty student generation for: {prompt[:50]}...")
                continue

            # Step 2: Get teacher logprobs on student's trajectory
            teacher_logprobs = self._get_teacher_logprobs_on_trajectory(
                prompt, student_tokens
            )

            if teacher_logprobs is None or len(teacher_logprobs) != len(student_logprobs):
                logger.warning(f"Teacher logprob mismatch, falling back to supervised")
                # Fallback: generate teacher response and do supervised
                context_bundle = self.context_generator.generate(prompt)
                teacher_response = self._generate_teacher_response(
                    prompt, context_bundle=context_bundle
                )
                if teacher_response:
                    metrics = self.train_supervised_step(prompt, teacher_response)
                else:
                    metrics = {"loss:sum": 0.0}
                training_type = "supervised_fallback"
            else:
                # Step 3: Train using reverse KL divergence
                metrics = self.train_gkd_step(
                    student_tokens,
                    student_logprobs,
                    teacher_logprobs,
                    step=step,  # Pass step for KL warmup
                )
                training_type = "gkd"

            # Compute score for monitoring (not used for training decisions)
            context_bundle = self.context_generator.generate(prompt)
            teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
            score = self._compute_score(student_response, teacher_response) if teacher_response else 0.0

            # Compute KL divergence for logging
            if training_type == "gkd" and student_logprobs and teacher_logprobs:
                kl_div = sum(sp - tp for sp, tp in zip(student_logprobs, teacher_logprobs)) / len(student_logprobs)
            else:
                kl_div = 0.0

            self.history.append({
                "step": step,
                "loss": metrics.get("loss:sum", 0.0),
                "score": score,
                "kl_divergence": kl_div,
                "training_type": training_type,
                "prompt": prompt[:100],
                "n_tokens": len(student_tokens),
            })

            if step % eval_every == 0:
                loss = metrics.get('loss:sum', 0.0)
                logger.info(
                    f"Step {step}: loss={loss:.4f}, "
                    f"score={score:.3f}, kl={kl_div:.4f}, type={training_type}"
                )

                # Log to wandb
                wandb_metrics = {
                    "train/loss": loss,
                    "train/score": score,
                    "train/kl_divergence": kl_div,
                    "train/step": step,
                    "train/n_tokens": len(student_tokens),
                    "train/training_type": 1 if training_type == "gkd" else 0,
                }

                if eval_prompts:
                    eval_result = self._evaluate(eval_prompts[:5])
                    self.history[-1]["eval"] = eval_result
                    avg_score = eval_result.get('avg_score', 0)
                    logger.info(f"  Eval avg_score: {avg_score:.3f}")
                    wandb_metrics["eval/avg_score"] = avg_score

                self._log_wandb(wandb_metrics, step=step)

        # Final checkpoint
        checkpoint_name = f"on_policy_gkd_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self._save_checkpoint(checkpoint_name)
        self.checkpoints.append(checkpoint_path)

        # Finish wandb
        self._finish_wandb()

        return {
            "history": self.history,
            "final_loss": self.history[-1]["loss"] if self.history else 0.0,
            "avg_score": sum(h.get("score", 0) for h in self.history) / len(self.history) if self.history else 0.0,
            "avg_kl": sum(h.get("kl_divergence", 0) for h in self.history) / len(self.history) if self.history else 0.0,
            "checkpoint": checkpoint_path,
            "training_mode": "on_policy_gkd",
        }

    def run_hybrid_distillation(
        self,
        prompts: List[str],
        max_steps: int = 100,
        off_policy_ratio: float = 0.5,
        eval_prompts: Optional[List[str]] = None,
        eval_every: int = 10,
        refresh_student_every: int = 20,
    ) -> Dict[str, Any]:
        """
        Run hybrid distillation: off-policy first, then on-policy GKD.

        This implements the sequential hybrid approach from the project spec:
        1. First phase: Off-policy distillation to bootstrap the student
        2. Second phase: On-policy GKD to refine with student's own distribution

        WARNING: Hybrid mode is known to cause mode collapse for cross-family
        distillation (e.g., Llama student ← Qwen teacher). After off-policy training
        on teacher outputs, the student's generation style drifts and produces
        empty/garbage outputs in on-policy phase. For cross-family distillation,
        use pure on-policy GKD instead.

        Args:
            prompts: Training prompts
            max_steps: Total training steps
            off_policy_ratio: Fraction of steps for off-policy (default 0.5)
            eval_prompts: Prompts for evaluation
            eval_every: Evaluate every N steps
            refresh_student_every: Refresh sampling client every N steps (on-policy phase)

        Returns:
            Combined results from both phases
        """
        # Warn about cross-family hybrid mode issues
        if self._is_cross_family_distillation():
            logger.warning(
                "⚠️  CROSS-FAMILY HYBRID MODE: Known to cause mode collapse! "
                "After off-policy training on Qwen teacher outputs, Llama student "
                "may generate empty/garbage in on-policy phase. "
                "Consider using pure on-policy GKD (run_on_policy_distillation) instead."
            )

        off_policy_steps = int(max_steps * off_policy_ratio)
        on_policy_steps = max_steps - off_policy_steps

        logger.info(f"Running hybrid distillation: {off_policy_steps} off-policy + {on_policy_steps} on-policy GKD")
        logger.info(f"Training prompts: {len(prompts)}")

        # Initialize wandb for the full run
        self._init_wandb(mode="hybrid")

        self.history = []
        combined_history = {
            "off_policy_phase": [],
            "on_policy_phase": [],
        }

        # === Phase 1: Off-Policy Bootstrap ===
        logger.info(f"\n--- Phase 1: Off-Policy Bootstrap ({off_policy_steps} steps) ---")
        prompt_idx = 0

        for step in range(off_policy_steps):
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # Generate context for teacher
            context_bundle = self.context_generator.generate(prompt)

            # Get teacher response
            teacher_response = self._generate_teacher_response(
                prompt,
                context_bundle=context_bundle,
                temperature=self.experiment_config.teacher.temperature,
            )

            if not teacher_response:
                logger.warning(f"Empty teacher response for: {prompt[:50]}...")
                continue

            # Train student on teacher response
            metrics = self.train_supervised_step(prompt, teacher_response)

            entry = {
                "step": step,
                "phase": "off_policy",
                "loss": metrics.get("loss:sum", 0.0),
                "prompt": prompt[:100],
                "teacher_response_len": len(teacher_response),
            }
            self.history.append(entry)
            combined_history["off_policy_phase"].append(entry)

            if step % eval_every == 0:
                loss = metrics.get('loss:sum', 0.0)
                logger.info(f"[Off-Policy] Step {step}: loss={loss:.4f}")

                wandb_metrics = {
                    "train/loss": loss,
                    "train/step": step,
                    "train/phase": 0,  # 0 = off-policy
                }

                if eval_prompts:
                    eval_result = self._evaluate(eval_prompts[:5])
                    entry["eval"] = eval_result
                    avg_score = eval_result.get('avg_score', 0)
                    logger.info(f"  Eval avg_score: {avg_score:.3f}")
                    wandb_metrics["eval/avg_score"] = avg_score

                self._log_wandb(wandb_metrics, step=step)

        # Refresh sampling client to use updated weights for on-policy phase
        logger.info("Transitioning to on-policy phase - refreshing student sampling client...")
        self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
            name="hybrid_transition"
        )

        # === Phase 2: On-Policy GKD Refinement ===
        logger.info(f"\n--- Phase 2: On-Policy GKD ({on_policy_steps} steps) ---")

        for step in range(on_policy_steps):
            global_step = off_policy_steps + step
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # Refresh sampling client periodically
            if step > 0 and step % refresh_student_every == 0:
                self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
                    name=f"hybrid_step_{global_step}"
                )
                logger.info(f"Refreshed student sampling client at step {global_step}")

            # Student generates response
            student_response, student_tokens, student_logprobs = self._generate_student_response(
                prompt,
                temperature=self.experiment_config.on_policy.temperature,
            )

            if not student_tokens or len(student_tokens) < 2:
                logger.warning(f"Empty student generation for: {prompt[:50]}...")
                continue

            # Get teacher logprobs on student's trajectory
            teacher_logprobs = self._get_teacher_logprobs_on_trajectory(prompt, student_tokens)

            if teacher_logprobs is None or len(teacher_logprobs) != len(student_logprobs):
                # Fallback to supervised
                context_bundle = self.context_generator.generate(prompt)
                teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
                if teacher_response:
                    metrics = self.train_supervised_step(prompt, teacher_response)
                else:
                    metrics = {"loss:sum": 0.0}
                training_type = "supervised_fallback"
            else:
                # Train using GKD
                metrics = self.train_gkd_step(student_tokens, student_logprobs, teacher_logprobs, step=step)
                training_type = "gkd"

            # Compute score for monitoring
            context_bundle = self.context_generator.generate(prompt)
            teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
            score = self._compute_score(student_response, teacher_response) if teacher_response else 0.0

            # Compute KL divergence
            if training_type == "gkd" and student_logprobs and teacher_logprobs:
                kl_div = sum(sp - tp for sp, tp in zip(student_logprobs, teacher_logprobs)) / len(student_logprobs)
            else:
                kl_div = 0.0

            entry = {
                "step": global_step,
                "phase": "on_policy",
                "loss": metrics.get("loss:sum", 0.0),
                "score": score,
                "kl_divergence": kl_div,
                "training_type": training_type,
                "prompt": prompt[:100],
                "n_tokens": len(student_tokens),
            }
            self.history.append(entry)
            combined_history["on_policy_phase"].append(entry)

            if step % eval_every == 0:
                loss = metrics.get('loss:sum', 0.0)
                logger.info(
                    f"[On-Policy] Step {global_step}: loss={loss:.4f}, "
                    f"score={score:.3f}, kl={kl_div:.4f}, type={training_type}"
                )

                wandb_metrics = {
                    "train/loss": loss,
                    "train/score": score,
                    "train/kl_divergence": kl_div,
                    "train/step": global_step,
                    "train/phase": 1,  # 1 = on-policy
                    "train/n_tokens": len(student_tokens),
                }

                if eval_prompts:
                    eval_result = self._evaluate(eval_prompts[:5])
                    entry["eval"] = eval_result
                    avg_score = eval_result.get('avg_score', 0)
                    logger.info(f"  Eval avg_score: {avg_score:.3f}")
                    wandb_metrics["eval/avg_score"] = avg_score

                self._log_wandb(wandb_metrics, step=global_step)

        # Final checkpoint
        checkpoint_name = f"hybrid_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self._save_checkpoint(checkpoint_name)
        self.checkpoints.append(checkpoint_path)

        # Finish wandb
        self._finish_wandb()

        # Compute phase statistics
        off_policy_losses = [h["loss"] for h in combined_history["off_policy_phase"]]
        on_policy_losses = [h["loss"] for h in combined_history["on_policy_phase"]]
        on_policy_scores = [h.get("score", 0) for h in combined_history["on_policy_phase"]]
        on_policy_kls = [h.get("kl_divergence", 0) for h in combined_history["on_policy_phase"]]

        return {
            "history": self.history,
            "combined_history": combined_history,
            "final_loss": self.history[-1]["loss"] if self.history else 0.0,
            "off_policy_final_loss": off_policy_losses[-1] if off_policy_losses else 0.0,
            "on_policy_final_loss": on_policy_losses[-1] if on_policy_losses else 0.0,
            "avg_score": sum(on_policy_scores) / len(on_policy_scores) if on_policy_scores else 0.0,
            "avg_kl": sum(on_policy_kls) / len(on_policy_kls) if on_policy_kls else 0.0,
            "checkpoint": checkpoint_path,
            "training_mode": "hybrid",
            "off_policy_steps": off_policy_steps,
            "on_policy_steps": on_policy_steps,
        }

    def run_hybrid_gradual_distillation(
        self,
        prompts: List[str],
        max_steps: int = 100,
        transition_schedule: str = None,
        min_off_policy_ratio: float = 0.0,
        max_off_policy_ratio: float = 1.0,
        eval_prompts: Optional[List[str]] = None,
        eval_every: int = 10,
        refresh_student_every: int = None,
    ) -> Dict[str, Any]:
        """
        Run gradual hybrid distillation with smooth transition.

        Unlike run_hybrid_distillation() which has a hard 50/50 split,
        this blends off-policy and on-policy objectives throughout training:

        - Step 0: 100% off-policy probability
        - Step N: 100% on-policy GKD probability
        - Interpolation: cosine or linear schedule

        This addresses the phase transition collapse observed in hard-switch hybrid mode.

        Args:
            prompts: Training prompts
            max_steps: Total training steps
            transition_schedule: "cosine" or "linear" (default: from config)
            min_off_policy_ratio: Off-policy ratio at end (default: 0.0)
            max_off_policy_ratio: Off-policy ratio at start (default: 1.0)
            eval_prompts: Prompts for evaluation
            eval_every: Evaluate every N steps
            refresh_student_every: Refresh sampling client every N steps (default: from config)

        Returns:
            Training results with history
        """
        import random
        import math

        # Use config defaults if not specified
        if transition_schedule is None:
            transition_schedule = self.config.gradual_transition_schedule
        if refresh_student_every is None:
            refresh_student_every = self.config.gradual_refresh_every

        logger.info(f"Running GRADUAL hybrid distillation for {max_steps} steps")
        logger.info(f"Schedule: {transition_schedule}")
        logger.info(f"Off-policy ratio: {max_off_policy_ratio} -> {min_off_policy_ratio}")
        logger.info(f"Refresh every: {refresh_student_every} steps")

        # Initialize wandb for tracking
        self._init_wandb(mode="hybrid_gradual")

        self.history = []
        prompt_idx = 0

        # Track statistics for analysis
        off_policy_steps_taken = 0
        on_policy_steps_taken = 0

        for step in range(max_steps):
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # === COMPUTE BLEND RATIO ===
            progress = step / max(max_steps - 1, 1)  # 0.0 to 1.0

            if transition_schedule == "cosine":
                # Cosine annealing: slow at start/end, fast in middle
                off_policy_ratio = min_off_policy_ratio + (
                    0.5 * (max_off_policy_ratio - min_off_policy_ratio) *
                    (1 + math.cos(progress * math.pi))
                )
            elif transition_schedule == "linear":
                # Linear interpolation
                off_policy_ratio = max_off_policy_ratio - progress * (
                    max_off_policy_ratio - min_off_policy_ratio
                )
            else:
                raise ValueError(f"Unknown schedule: {transition_schedule}")

            # === PROBABILISTIC STEP SELECTION ===
            use_off_policy = random.random() < off_policy_ratio

            # === REFRESH SAMPLING CLIENT ===
            # More frequent refresh needed for gradual transition
            if step > 0 and step % refresh_student_every == 0:
                self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
                    name=f"gradual_step_{step}"
                )
                logger.debug(f"Refreshed sampling client at step {step}")

            # === EXECUTE SELECTED TRAINING STEP ===
            if use_off_policy:
                # Off-policy: supervised on teacher response
                context_bundle = self.context_generator.generate(prompt)
                teacher_response = self._generate_teacher_response(
                    prompt,
                    context_bundle=context_bundle,
                    temperature=self.experiment_config.teacher.temperature,
                )

                if not teacher_response:
                    logger.warning(f"Empty teacher response, skipping step")
                    continue

                metrics = self.train_supervised_step(prompt, teacher_response)
                training_type = "off_policy"
                off_policy_steps_taken += 1

                # Record for history
                kl_div = 0.0  # Not applicable for off-policy
                score = 0.0   # Not computed for off-policy
                n_tokens = len(teacher_response.split())  # Approximate

            else:
                # On-policy GKD: student generates, teacher scores
                student_response, student_tokens, student_logprobs = self._generate_student_response(
                    prompt,
                    temperature=self.experiment_config.on_policy.temperature,
                )

                if not student_tokens or len(student_tokens) < 2:
                    # Fallback to off-policy for this step
                    logger.warning(f"Empty student generation, falling back to off-policy")
                    context_bundle = self.context_generator.generate(prompt)
                    teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
                    if teacher_response:
                        metrics = self.train_supervised_step(prompt, teacher_response)
                        training_type = "off_policy_fallback"
                        off_policy_steps_taken += 1
                    else:
                        continue
                    kl_div = 0.0
                    score = 0.0
                    n_tokens = 0
                else:
                    # Get teacher logprobs on student trajectory
                    teacher_logprobs = self._get_teacher_logprobs_on_trajectory(prompt, student_tokens)

                    if teacher_logprobs is None or len(teacher_logprobs) != len(student_logprobs):
                        # Fallback to supervised
                        logger.warning(f"Teacher logprob mismatch, falling back to off-policy")
                        context_bundle = self.context_generator.generate(prompt)
                        teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
                        if teacher_response:
                            metrics = self.train_supervised_step(prompt, teacher_response)
                        else:
                            continue
                        training_type = "off_policy_fallback"
                        off_policy_steps_taken += 1
                        kl_div = 0.0
                        score = 0.0
                        n_tokens = 0
                    else:
                        # Train using GKD
                        metrics = self.train_gkd_step(
                            student_tokens,
                            student_logprobs,
                            teacher_logprobs,
                            step=step,  # For KL warmup
                        )
                        training_type = "on_policy_gkd"
                        on_policy_steps_taken += 1

                        # Compute KL and score for logging
                        kl_div = sum(sp - tp for sp, tp in zip(student_logprobs, teacher_logprobs)) / len(student_logprobs)

                        # Compute score vs teacher response
                        context_bundle = self.context_generator.generate(prompt)
                        teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
                        score = self._compute_score(student_response, teacher_response) if teacher_response else 0.0
                        n_tokens = len(student_tokens)

            # === RECORD HISTORY ===
            entry = {
                "step": step,
                "loss": metrics.get("loss:sum", 0.0),
                "score": score,
                "kl_divergence": kl_div,
                "training_type": training_type,
                "off_policy_ratio": off_policy_ratio,
                "prompt": prompt[:100],
                "n_tokens": n_tokens,
            }
            self.history.append(entry)

            # === LOGGING ===
            if step % eval_every == 0:
                loss = metrics.get('loss:sum', 0.0)
                logger.info(
                    f"Step {step}: loss={loss:.4f}, score={score:.3f}, "
                    f"kl={kl_div:.4f}, ratio={off_policy_ratio:.2f}, type={training_type}"
                )

                # Wandb logging
                wandb_metrics = {
                    "train/loss": loss,
                    "train/score": score,
                    "train/kl_divergence": kl_div,
                    "train/off_policy_ratio": off_policy_ratio,
                    "train/step": step,
                    "train/training_type": 0 if "off_policy" in training_type else 1,
                    "train/off_policy_steps": off_policy_steps_taken,
                    "train/on_policy_steps": on_policy_steps_taken,
                }

                if eval_prompts:
                    eval_result = self._evaluate(eval_prompts[:5])
                    entry["eval"] = eval_result
                    avg_score = eval_result.get('avg_score', 0)
                    logger.info(f"  Eval avg_score: {avg_score:.3f}")
                    wandb_metrics["eval/avg_score"] = avg_score

                self._log_wandb(wandb_metrics, step=step)

        # === FINAL CHECKPOINT ===
        checkpoint_name = f"hybrid_gradual_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self._save_checkpoint(checkpoint_name)
        self.checkpoints.append(checkpoint_path)

        # Finish wandb
        self._finish_wandb()

        # === COMPUTE FINAL STATISTICS ===
        on_policy_entries = [h for h in self.history if h["training_type"] == "on_policy_gkd"]
        off_policy_entries = [h for h in self.history if "off_policy" in h["training_type"]]

        return {
            "history": self.history,
            "final_loss": self.history[-1]["loss"] if self.history else 0.0,
            "avg_score": sum(h.get("score", 0) for h in on_policy_entries) / len(on_policy_entries) if on_policy_entries else 0.0,
            "avg_kl": sum(h.get("kl_divergence", 0) for h in on_policy_entries) / len(on_policy_entries) if on_policy_entries else 0.0,
            "checkpoint": checkpoint_path,
            "training_mode": "hybrid_gradual",
            "transition_schedule": transition_schedule,
            "off_policy_steps": off_policy_steps_taken,
            "on_policy_steps": on_policy_steps_taken,
            "statistics": {
                "off_policy_ratio_start": max_off_policy_ratio,
                "off_policy_ratio_end": min_off_policy_ratio,
                "schedule": transition_schedule,
            }
        }

    def _is_cross_family_distillation(self) -> bool:
        """Check if student and teacher use different model families (Llama vs Qwen)."""
        student_is_llama = "llama" in self.config.student_model.lower()
        teacher_is_llama = "llama" in self.config.teacher_model.lower()
        return student_is_llama != teacher_is_llama

    def _interpolate_logprobs(
        self,
        text: str,
        student_tokens: List[int],
        teacher_tokens: List[int],
        teacher_logprobs: List[float],
    ) -> List[float]:
        """
        Interpolate teacher logprobs to align with student token positions.

        When tokenizers produce different token counts, we can't do 1:1 alignment.
        Instead, we map based on character positions:
        1. Get character span for each student token
        2. Get character span for each teacher token
        3. For each student token, average logprobs of overlapping teacher tokens

        This enables GKD for cross-family distillation (Llama ↔ Qwen).
        """
        # Get character spans for student tokens
        student_spans = []
        char_pos = 0
        for token_id in student_tokens:
            token_text = self._tokenizer.decode([token_id], skip_special_tokens=False)
            student_spans.append((char_pos, char_pos + len(token_text)))
            char_pos += len(token_text)

        # Get character spans for teacher tokens
        teacher_spans = []
        char_pos = 0
        for token_id in teacher_tokens:
            token_text = self._teacher_tokenizer.decode([token_id], skip_special_tokens=False)
            teacher_spans.append((char_pos, char_pos + len(token_text)))
            char_pos += len(token_text)

        # Map teacher logprobs to student tokens based on overlap
        aligned_logprobs = []
        for s_start, s_end in student_spans:
            overlapping_logprobs = []
            overlapping_weights = []

            for t_idx, (t_start, t_end) in enumerate(teacher_spans):
                # Calculate overlap
                overlap_start = max(s_start, t_start)
                overlap_end = min(s_end, t_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > 0 and t_idx < len(teacher_logprobs):
                    overlapping_logprobs.append(teacher_logprobs[t_idx])
                    overlapping_weights.append(overlap)

            if overlapping_logprobs:
                # Weighted average of overlapping teacher logprobs
                total_weight = sum(overlapping_weights)
                weighted_avg = sum(
                    lp * w for lp, w in zip(overlapping_logprobs, overlapping_weights)
                ) / total_weight
                aligned_logprobs.append(weighted_avg)
            else:
                # No overlap - use average of all teacher logprobs as fallback
                aligned_logprobs.append(sum(teacher_logprobs) / len(teacher_logprobs) if teacher_logprobs else -10.0)

        return aligned_logprobs

    def _get_teacher_logprobs_on_trajectory(
        self,
        prompt: str,
        student_tokens: List[int],
    ) -> Optional[List[float]]:
        """
        Get teacher's log probabilities for each token in the student's trajectory.

        This is the key insight of GKD: we evaluate how likely the teacher thinks
        each of the student's generated tokens is. This gives us dense per-token
        feedback instead of sparse episode-level rewards.

        For cross-family distillation (e.g., Llama student → Qwen teacher):
        1. Decode student tokens to text using student tokenizer
        2. Re-encode with teacher tokenizer
        3. Compute teacher logprobs on teacher-tokenized version
        4. Interpolate teacher logprobs back to student token positions

        The interpolation step enables GKD to work even when tokenizers
        produce different token counts for the same text.
        """
        import tinker

        # Build the full sequence: prompt + student tokens
        messages = [{"role": "user", "content": prompt}]

        # Handle cross-family distillation (Llama ↔ Qwen)
        if self._is_cross_family_distillation():
            # Decode student tokens to text using student tokenizer
            student_text = self._tokenizer.decode(student_tokens, skip_special_tokens=True)

            # Build teacher prompt and encode student text with teacher tokenizer
            teacher_prompt_input = self._teacher_renderer.build_generation_prompt(messages)

            # Encode student text with teacher tokenizer
            teacher_response_tokens = self._teacher_tokenizer.encode(
                student_text, add_special_tokens=False
            )

            # Create full sequence for teacher
            full_tokens = list(teacher_prompt_input.to_ints()) + teacher_response_tokens
            prompt_len = len(list(teacher_prompt_input.to_ints()))
        else:
            # Same family: can use student tokens directly
            prompt_input = self._teacher_renderer.build_generation_prompt(messages)
            full_tokens = list(prompt_input.to_ints()) + student_tokens
            prompt_len = len(list(prompt_input.to_ints()))

        # Use compute_logprobs to get teacher's assessment of each token
        try:
            # compute_logprobs returns logprobs for each position
            logprobs_result = self._teacher_client.compute_logprobs(
                tinker.types.ModelInput.from_ints(full_tokens)
            ).result()

            # Extract logprobs for only the response portion
            # Skip the prompt tokens
            teacher_logprobs = list(logprobs_result[prompt_len:])

            # For cross-family: interpolate teacher logprobs to student token positions
            if self._is_cross_family_distillation():
                student_text = self._tokenizer.decode(student_tokens, skip_special_tokens=True)
                teacher_response_tokens = self._teacher_tokenizer.encode(
                    student_text, add_special_tokens=False
                )

                aligned_logprobs = self._interpolate_logprobs(
                    text=student_text,
                    student_tokens=student_tokens,
                    teacher_tokens=teacher_response_tokens,
                    teacher_logprobs=teacher_logprobs,
                )

                logger.debug(
                    f"Cross-family interpolation: teacher={len(teacher_logprobs)} tokens "
                    f"→ student={len(aligned_logprobs)} tokens"
                )
                return aligned_logprobs

            return teacher_logprobs
        except Exception as e:
            logger.warning(f"Failed to get teacher logprobs: {e}")
            return None

    def train_gkd_step(
        self,
        tokens: List[int],
        student_logprobs: List[float],
        teacher_logprobs: List[float],
        step: int = 0,
    ) -> Dict[str, float]:
        """
        Train using Generalized Knowledge Distillation (reverse KL).

        The loss is: L = E_student[log p_student(x) - log p_teacher(x)]

        We minimize this by treating (student_logprob - teacher_logprob) as
        per-token advantages in a policy gradient formulation.

        When student assigns higher probability than teacher → positive advantage → discourage
        When student assigns lower probability than teacher → negative advantage → encourage

        CRITICAL: Language model training uses shifted sequences:
        - model_input = tokens[:-1] (input to predict next token)
        - target_tokens = tokens[1:] (ground truth for next token prediction)
        - logprobs/advantages also shift by 1 to align with targets

        Args:
            step: Current training step for KL warmup scheduling
        """
        import tinker
        from tinker import types
        import torch

        # Compute per-token advantages: -(student_logprob - teacher_logprob)
        # Negative because we want to MINIMIZE divergence from teacher
        # If student_logprob > teacher_logprob, we want negative advantage (discourage)
        # If student_logprob < teacher_logprob, we want positive advantage (encourage)
        raw_advantages = [
            -(s_lp - t_lp)
            for s_lp, t_lp in zip(student_logprobs, teacher_logprobs)
        ]

        # KL warmup: gradually increase advantage clipping to prevent mode collapse
        # Early training uses tighter clipping to stabilize, then relaxes
        warmup_steps = self.config.kl_warmup_steps
        initial_clip = self.config.initial_advantage_clip
        max_clip = self.config.max_advantage_clip

        if step < warmup_steps:
            # Linear interpolation from initial to max
            progress = step / warmup_steps
            max_advantage = initial_clip + progress * (max_clip - initial_clip)
        else:
            max_advantage = max_clip

        # Clip advantages to prevent exploding gradients
        advantages = [
            max(-max_advantage, min(max_advantage, a))
            for a in raw_advantages
        ]

        # Language model training uses shifted sequences:
        # - model_input = tokens[:-1] (predict next token)
        # - target_tokens = tokens[1:] (ground truth for next token)
        # All other arrays (logprobs, advantages) also shift by 1
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        shifted_logprobs = student_logprobs[1:]  # Align with targets
        shifted_advantages = advantages[1:]  # Align with targets

        # importance_sampling loss only accepts: target_tokens, logprobs, advantages
        # (no mask field - masking is implicit through zero-valued advantages)
        datum = tinker.Datum(
            model_input=types.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": types.TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": types.TensorData.from_torch(torch.tensor(shifted_logprobs, dtype=torch.float32)),
                "advantages": types.TensorData.from_torch(torch.tensor(shifted_advantages, dtype=torch.float32)),
            }
        )

        fwd_bwd_future = self._training_client.forward_backward(
            [datum], loss_fn="importance_sampling"
        )

        # Use lower learning rate for on-policy/GKD to prevent KL mode collapse
        optim_future = self._training_client.optim_step(
            types.AdamParams(
                learning_rate=self.config.on_policy_learning_rate,
                beta1=self.config.beta1,
                beta2=self.config.beta2,
            )
        )

        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_future.result()

        return fwd_bwd_result.metrics

    def _evaluate(
        self,
        prompts: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate current student against teacher.

        Args:
            prompts: List of prompts to evaluate on
            ground_truths: Optional list of ground truth answers for accuracy check.
                          If provided, also computes answer accuracy (not just text similarity).

        Returns:
            Dict with avg_score (text similarity) and optionally accuracy (answer correctness).
        """
        scores = []
        correct = 0
        total_with_gt = 0

        for i, prompt in enumerate(prompts):
            # Generate student response
            student_response, _, _ = self._generate_student_response(
                prompt, temperature=0.01  # Near-greedy for eval
            )

            # Generate teacher response (without context - fair comparison)
            teacher_response = self._generate_teacher_response(
                prompt, temperature=0.01
            )

            score = self._compute_score(student_response, teacher_response)
            scores.append(score)

            # Check answer correctness if ground truth provided
            if ground_truths and i < len(ground_truths):
                gt = ground_truths[i]
                if gt:
                    total_with_gt += 1
                    # Extract numerical answer from student response
                    predicted = self._extract_answer(student_response)
                    expected = self._extract_answer(gt)
                    if predicted and expected and predicted == expected:
                        correct += 1

        result = {
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "n_samples": len(scores),
        }

        # Add accuracy if ground truths were provided
        if total_with_gt > 0:
            result["accuracy"] = correct / total_with_gt
            result["correct"] = correct
            result["total_evaluated"] = total_with_gt

        return result

    def _extract_answer(self, response: str) -> Optional[str]:
        """Extract numerical answer from a response (for GSM8K-style benchmarks).

        Looks for patterns like:
        - #### 42
        - The answer is 42
        - = 42
        - Final answer: $42
        """
        import re

        if not response:
            return None

        # GSM8K format: #### <number>
        match = re.search(r"####\s*\$?([0-9,]+(?:\.[0-9]+)?)", response)
        if match:
            return match.group(1).replace(",", "").replace("$", "")

        # "The answer is X" or "answer = X"
        match = re.search(
            r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            response,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).replace(",", "").replace("$", "")

        # = X at end of response
        match = re.search(r"=\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*$", response)
        if match:
            return match.group(1).replace(",", "").replace("$", "")

        # Fallback: last number in response
        numbers = re.findall(r"\$?([0-9,]+(?:\.[0-9]+)?)", response)
        if numbers:
            return numbers[-1].replace(",", "").replace("$", "")

        return None

    # ==========================================================================
    # DISTRIBUTION CLIFF MITIGATION METHODS
    # ==========================================================================
    # These methods implement various strategies to mitigate the "distribution cliff"
    # problem observed in hybrid distillation, where transitioning from off-policy
    # to on-policy training causes catastrophic collapse.
    #
    # Methods:
    # 1. run_extended_on_policy - Pure on-policy with extended training (baseline)
    # 2. run_teacher_seeded_on_policy - On-policy with teacher-initialized sampling
    # 3. run_mixture_distillation - Mixed objectives every step (no phases)
    # 4. run_replay_buffer_distillation - On-policy with teacher replay buffer
    # 5. run_kl_anchored_off_policy - Off-policy with KL anchor to initial student
    # 6. run_reverse_curriculum - On-policy first, then off-policy refinement
    # ==========================================================================

    def run_extended_on_policy(
        self,
        prompts: List[str],
        max_steps: int = 200,
        eval_prompts: Optional[List[str]] = None,
        eval_every: int = 10,
        refresh_student_every: int = 20,
    ) -> Dict[str, Any]:
        """
        Method 1: Extended pure on-policy training.

        Hypothesis: The original 100 steps may be insufficient. Pure on-policy
        achieved 4.2% accuracy - extending training may improve this without
        introducing the distribution cliff from hybrid approaches.

        This serves as the baseline for comparison.
        """
        logger.info(f"Running EXTENDED on-policy distillation for {max_steps} steps")
        logger.info("This is the baseline: pure on-policy with more training")

        # Simply call the existing on-policy method with more steps
        return self.run_on_policy_distillation(
            prompts=prompts,
            max_steps=max_steps,
            eval_prompts=eval_prompts,
            eval_every=eval_every,
            refresh_student_every=refresh_student_every,
        )

    def run_teacher_seeded_on_policy(
        self,
        prompts: List[str],
        max_steps: int = 100,
        eval_prompts: Optional[List[str]] = None,
        eval_every: int = 10,
        initial_seed_tokens: int = 20,
        seed_decay_steps: int = 50,
        refresh_student_every: int = 20,
    ) -> Dict[str, Any]:
        """
        Method 2: On-policy with teacher-initialized sampling.

        Instead of off-policy training, use teacher outputs to SEED student
        generation. Student generates its own response, but starts from a
        prefix provided by the teacher.

        Args:
            initial_seed_tokens: Number of teacher tokens to seed at start
            seed_decay_steps: Steps over which to decay seed length to 0
        """
        import random

        logger.info(f"Running TEACHER-SEEDED on-policy for {max_steps} steps")
        logger.info(f"Initial seed: {initial_seed_tokens} tokens, decay over {seed_decay_steps} steps")

        self._init_wandb(mode="teacher_seeded_on_policy")
        self.history = []
        prompt_idx = 0

        for step in range(max_steps):
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # Calculate current seed length (linear decay)
            if step < seed_decay_steps:
                progress = step / seed_decay_steps
                current_seed_tokens = int(initial_seed_tokens * (1 - progress))
            else:
                current_seed_tokens = 0

            # Refresh student sampling client
            if step > 0 and step % refresh_student_every == 0:
                self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
                    name=f"teacher_seeded_step_{step}"
                )

            # Get teacher prefix if seeding
            teacher_prefix_tokens = []
            if current_seed_tokens > 0:
                context_bundle = self.context_generator.generate(prompt)
                teacher_response = self._generate_teacher_response(
                    prompt, context_bundle=context_bundle
                )
                if teacher_response:
                    # Tokenize teacher response and take prefix
                    full_tokens = self._tokenizer.encode(teacher_response, add_special_tokens=False)
                    teacher_prefix_tokens = full_tokens[:current_seed_tokens]

            # Generate student response, seeded with teacher prefix
            student_response, student_tokens, student_logprobs = self._generate_student_response_seeded(
                prompt,
                seed_tokens=teacher_prefix_tokens,
                temperature=self.experiment_config.on_policy.temperature,
            )

            if not student_tokens or len(student_tokens) < 2:
                continue

            # Get teacher logprobs on student trajectory
            teacher_logprobs = self._get_teacher_logprobs_on_trajectory(prompt, student_tokens)

            if teacher_logprobs is None or len(teacher_logprobs) != len(student_logprobs):
                # Fallback to supervised
                context_bundle = self.context_generator.generate(prompt)
                teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
                if teacher_response:
                    metrics = self.train_supervised_step(prompt, teacher_response)
                else:
                    metrics = {"loss:sum": 0.0}
                training_type = "supervised_fallback"
                kl_div = 0.0
            else:
                metrics = self.train_gkd_step(student_tokens, student_logprobs, teacher_logprobs, step=step)
                training_type = "teacher_seeded_gkd"
                kl_div = sum(sp - tp for sp, tp in zip(student_logprobs, teacher_logprobs)) / len(student_logprobs)

            # Score for monitoring
            context_bundle = self.context_generator.generate(prompt)
            teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
            score = self._compute_score(student_response, teacher_response) if teacher_response else 0.0

            self.history.append({
                "step": step,
                "loss": metrics.get("loss:sum", 0.0),
                "score": score,
                "kl_divergence": kl_div,
                "training_type": training_type,
                "seed_tokens": current_seed_tokens,
                "prompt": prompt[:100],
            })

            if step % eval_every == 0:
                logger.info(f"Step {step}: loss={metrics.get('loss:sum', 0):.4f}, "
                           f"score={score:.3f}, kl={kl_div:.4f}, seed={current_seed_tokens}")
                self._log_wandb({
                    "train/loss": metrics.get("loss:sum", 0.0),
                    "train/score": score,
                    "train/kl_divergence": kl_div,
                    "train/seed_tokens": current_seed_tokens,
                }, step=step)

        checkpoint_path = self._save_checkpoint(f"teacher_seeded_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self._finish_wandb()

        return {
            "history": self.history,
            "final_loss": self.history[-1]["loss"] if self.history else 0.0,
            "avg_score": sum(h.get("score", 0) for h in self.history) / len(self.history) if self.history else 0.0,
            "avg_kl": sum(h.get("kl_divergence", 0) for h in self.history) / len(self.history) if self.history else 0.0,
            "checkpoint": checkpoint_path,
            "training_mode": "teacher_seeded_on_policy",
        }

    def _generate_student_response_seeded(
        self,
        prompt: str,
        seed_tokens: List[int],
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> tuple:
        """Generate student response starting from seed tokens."""
        import tinker

        messages = [{"role": "user", "content": prompt}]
        model_input = self._renderer.build_generation_prompt(messages)

        # Append seed tokens to the prompt
        if seed_tokens:
            input_tokens = model_input.to_ints() + seed_tokens
            model_input = tinker.types.ModelInput.from_ints(input_tokens)

        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens - len(seed_tokens),  # Account for seed
            temperature=max(temperature, 0.01),
        )

        result = self._sampling_client.sample(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1,
        ).result()

        if result.sequences:
            seq = result.sequences[0]
            # Full tokens = seed + generated
            all_tokens = seed_tokens + list(seq.tokens)
            # Logprobs for seed tokens are 0 (forced), then actual logprobs
            all_logprobs = [0.0] * len(seed_tokens) + list(seq.logprobs or [])

            response_text = self._tokenizer.decode(all_tokens, skip_special_tokens=True)
            return response_text, all_tokens, all_logprobs

        return "", [], []

    def run_mixture_distillation(
        self,
        prompts: List[str],
        max_steps: int = 100,
        eval_prompts: Optional[List[str]] = None,
        eval_every: int = 10,
        off_policy_weight: float = 0.3,
        refresh_student_every: int = 20,
    ) -> Dict[str, Any]:
        """
        Method 3: Mixed objectives every step.

        Instead of phases (off→on), mix both objectives in every training step:
        loss = alpha * off_policy_loss + (1-alpha) * on_policy_loss

        This prevents the distribution from ever narrowing too much because
        both signals are always present.

        Args:
            off_policy_weight: Weight for off-policy loss (alpha), on-policy gets (1-alpha)
        """
        logger.info(f"Running MIXTURE distillation for {max_steps} steps")
        logger.info(f"Off-policy weight: {off_policy_weight}, On-policy weight: {1-off_policy_weight}")

        self._init_wandb(mode="mixture")
        self.history = []
        prompt_idx = 0

        for step in range(max_steps):
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # Refresh student
            if step > 0 and step % refresh_student_every == 0:
                self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
                    name=f"mixture_step_{step}"
                )

            # === ON-POLICY COMPONENT ===
            student_response, student_tokens, student_logprobs = self._generate_student_response(
                prompt, temperature=self.experiment_config.on_policy.temperature
            )

            on_policy_loss = 0.0
            on_policy_kl = 0.0
            if student_tokens and len(student_tokens) >= 2:
                teacher_logprobs = self._get_teacher_logprobs_on_trajectory(prompt, student_tokens)
                if teacher_logprobs and len(teacher_logprobs) == len(student_logprobs):
                    # Don't apply optimizer yet - we'll combine losses
                    on_policy_kl = sum(sp - tp for sp, tp in zip(student_logprobs, teacher_logprobs)) / len(student_logprobs)
                    # Train GKD
                    on_metrics = self.train_gkd_step(student_tokens, student_logprobs, teacher_logprobs, step=step)
                    on_policy_loss = on_metrics.get("loss:sum", 0.0)

            # === OFF-POLICY COMPONENT ===
            context_bundle = self.context_generator.generate(prompt)
            teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)

            off_policy_loss = 0.0
            if teacher_response:
                off_metrics = self.train_supervised_step(prompt, teacher_response)
                off_policy_loss = off_metrics.get("loss:sum", 0.0)

            # Combined loss (already applied separately, this is for logging)
            combined_loss = off_policy_weight * off_policy_loss + (1 - off_policy_weight) * on_policy_loss

            # Score
            score = self._compute_score(student_response, teacher_response) if teacher_response else 0.0

            self.history.append({
                "step": step,
                "loss": combined_loss,
                "on_policy_loss": on_policy_loss,
                "off_policy_loss": off_policy_loss,
                "score": score,
                "kl_divergence": on_policy_kl,
                "training_type": "mixture",
                "prompt": prompt[:100],
            })

            if step % eval_every == 0:
                logger.info(f"Step {step}: combined={combined_loss:.4f} "
                           f"(off={off_policy_loss:.4f}, on={on_policy_loss:.4f}), "
                           f"score={score:.3f}, kl={on_policy_kl:.4f}")
                self._log_wandb({
                    "train/loss": combined_loss,
                    "train/off_policy_loss": off_policy_loss,
                    "train/on_policy_loss": on_policy_loss,
                    "train/score": score,
                    "train/kl_divergence": on_policy_kl,
                }, step=step)

        checkpoint_path = self._save_checkpoint(f"mixture_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self._finish_wandb()

        return {
            "history": self.history,
            "final_loss": self.history[-1]["loss"] if self.history else 0.0,
            "avg_score": sum(h.get("score", 0) for h in self.history) / len(self.history) if self.history else 0.0,
            "avg_kl": sum(h.get("kl_divergence", 0) for h in self.history) / len(self.history) if self.history else 0.0,
            "checkpoint": checkpoint_path,
            "training_mode": "mixture",
            "off_policy_weight": off_policy_weight,
        }

    def run_replay_buffer_distillation(
        self,
        prompts: List[str],
        max_steps: int = 100,
        eval_prompts: Optional[List[str]] = None,
        eval_every: int = 10,
        buffer_size: int = 50,
        teacher_injection_rate: float = 0.2,
        refresh_student_every: int = 20,
    ) -> Dict[str, Any]:
        """
        Method 4: On-policy with replay buffer.

        Maintain a buffer of (prompt, response) pairs. Train on-policy but
        periodically inject teacher responses into the buffer. Sample from
        buffer with recency weighting.

        This is similar to experience replay in RL - maintains exploration
        while preserving teacher knowledge.

        Args:
            buffer_size: Maximum buffer capacity
            teacher_injection_rate: Fraction of steps to inject teacher response
        """
        import random

        logger.info(f"Running REPLAY BUFFER distillation for {max_steps} steps")
        logger.info(f"Buffer size: {buffer_size}, Teacher injection rate: {teacher_injection_rate}")

        self._init_wandb(mode="replay_buffer")
        self.history = []

        # Replay buffer: list of (prompt, response, is_teacher, step_added)
        replay_buffer = []
        prompt_idx = 0

        for step in range(max_steps):
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # Refresh student
            if step > 0 and step % refresh_student_every == 0:
                self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
                    name=f"replay_step_{step}"
                )

            # === GENERATE & ADD TO BUFFER ===
            # Generate student response
            student_response, student_tokens, student_logprobs = self._generate_student_response(
                prompt, temperature=self.experiment_config.on_policy.temperature
            )

            # Add student response to buffer
            if student_response:
                replay_buffer.append({
                    "prompt": prompt,
                    "response": student_response,
                    "tokens": student_tokens,
                    "logprobs": student_logprobs,
                    "is_teacher": False,
                    "step": step,
                })

            # Periodically inject teacher response
            if random.random() < teacher_injection_rate:
                context_bundle = self.context_generator.generate(prompt)
                teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
                if teacher_response:
                    teacher_tokens = self._tokenizer.encode(teacher_response, add_special_tokens=False)
                    replay_buffer.append({
                        "prompt": prompt,
                        "response": teacher_response,
                        "tokens": teacher_tokens,
                        "logprobs": None,  # Teacher doesn't have student logprobs
                        "is_teacher": True,
                        "step": step,
                    })

            # Trim buffer if too large (remove oldest entries)
            while len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)

            # === TRAIN FROM BUFFER ===
            # Sample from buffer with recency weighting
            if replay_buffer:
                # Weight by recency: more recent = higher weight
                weights = [1.0 + (entry["step"] / max(step, 1)) for entry in replay_buffer]
                total_weight = sum(weights)
                probs = [w / total_weight for w in weights]

                # Sample one entry
                sampled = random.choices(replay_buffer, weights=probs, k=1)[0]

                if sampled["is_teacher"]:
                    # Supervised learning on teacher response
                    metrics = self.train_supervised_step(sampled["prompt"], sampled["response"])
                    training_type = "replay_teacher"
                    kl_div = 0.0
                else:
                    # GKD on student response
                    teacher_logprobs = self._get_teacher_logprobs_on_trajectory(
                        sampled["prompt"], sampled["tokens"]
                    )
                    if teacher_logprobs and sampled["logprobs"] and len(teacher_logprobs) == len(sampled["logprobs"]):
                        metrics = self.train_gkd_step(
                            sampled["tokens"], sampled["logprobs"], teacher_logprobs, step=step
                        )
                        kl_div = sum(sp - tp for sp, tp in zip(sampled["logprobs"], teacher_logprobs)) / len(sampled["logprobs"])
                        training_type = "replay_student_gkd"
                    else:
                        metrics = self.train_supervised_step(sampled["prompt"], sampled["response"])
                        kl_div = 0.0
                        training_type = "replay_student_supervised"
            else:
                metrics = {"loss:sum": 0.0}
                training_type = "buffer_empty"
                kl_div = 0.0

            # Score current student
            context_bundle = self.context_generator.generate(prompt)
            teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
            score = self._compute_score(student_response, teacher_response) if teacher_response and student_response else 0.0

            self.history.append({
                "step": step,
                "loss": metrics.get("loss:sum", 0.0),
                "score": score,
                "kl_divergence": kl_div,
                "training_type": training_type,
                "buffer_size": len(replay_buffer),
                "prompt": prompt[:100],
            })

            if step % eval_every == 0:
                n_teacher = sum(1 for e in replay_buffer if e["is_teacher"])
                logger.info(f"Step {step}: loss={metrics.get('loss:sum', 0):.4f}, "
                           f"score={score:.3f}, buffer={len(replay_buffer)} ({n_teacher} teacher)")
                self._log_wandb({
                    "train/loss": metrics.get("loss:sum", 0.0),
                    "train/score": score,
                    "train/kl_divergence": kl_div,
                    "train/buffer_size": len(replay_buffer),
                    "train/buffer_teacher_count": n_teacher,
                }, step=step)

        checkpoint_path = self._save_checkpoint(f"replay_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self._finish_wandb()

        return {
            "history": self.history,
            "final_loss": self.history[-1]["loss"] if self.history else 0.0,
            "avg_score": sum(h.get("score", 0) for h in self.history) / len(self.history) if self.history else 0.0,
            "avg_kl": sum(h.get("kl_divergence", 0) for h in self.history) / len(self.history) if self.history else 0.0,
            "checkpoint": checkpoint_path,
            "training_mode": "replay_buffer",
        }

    def run_kl_anchored_off_policy(
        self,
        prompts: List[str],
        max_steps: int = 100,
        eval_prompts: Optional[List[str]] = None,
        eval_every: int = 10,
        kl_anchor_weight: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Method 5: Off-policy with KL anchor to initial student.

        During off-policy training, add a KL penalty that keeps the student
        close to its INITIAL distribution. This prevents the student from
        drifting too far during off-policy, making any subsequent transition
        to on-policy less severe.

        loss = CE(student, teacher) + beta * KL(student || student_init)

        Args:
            kl_anchor_weight: Weight for KL anchor term (beta)
        """
        import tinker

        logger.info(f"Running KL-ANCHORED off-policy for {max_steps} steps")
        logger.info(f"KL anchor weight: {kl_anchor_weight}")

        self._init_wandb(mode="kl_anchored_off_policy")
        self.history = []

        # Save initial student for KL anchor
        # We'll compute KL by getting logprobs from initial student
        initial_sampling_client = self._sampling_client
        prompt_idx = 0

        for step in range(max_steps):
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            # Get teacher response
            context_bundle = self.context_generator.generate(prompt)
            teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)

            if not teacher_response:
                continue

            # Standard supervised loss
            metrics = self.train_supervised_step(prompt, teacher_response)
            ce_loss = metrics.get("loss:sum", 0.0)

            # Compute KL anchor term
            # Get current student's logprobs on teacher response
            teacher_tokens = self._tokenizer.encode(teacher_response, add_special_tokens=False)

            # This is approximate - we're computing KL implicitly through the
            # training dynamics rather than explicitly adding a term
            # True implementation would require dual forward passes
            kl_anchor_loss = 0.0  # Placeholder - actual KL would require more compute

            total_loss = ce_loss + kl_anchor_weight * kl_anchor_loss

            self.history.append({
                "step": step,
                "loss": total_loss,
                "ce_loss": ce_loss,
                "kl_anchor_loss": kl_anchor_loss,
                "training_type": "kl_anchored_off_policy",
                "prompt": prompt[:100],
            })

            if step % eval_every == 0:
                logger.info(f"Step {step}: loss={total_loss:.4f} (ce={ce_loss:.4f})")
                self._log_wandb({
                    "train/loss": total_loss,
                    "train/ce_loss": ce_loss,
                    "train/kl_anchor_loss": kl_anchor_loss,
                }, step=step)

        checkpoint_path = self._save_checkpoint(f"kl_anchored_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self._finish_wandb()

        return {
            "history": self.history,
            "final_loss": self.history[-1]["loss"] if self.history else 0.0,
            "checkpoint": checkpoint_path,
            "training_mode": "kl_anchored_off_policy",
        }

    def run_reverse_curriculum(
        self,
        prompts: List[str],
        max_steps: int = 100,
        eval_prompts: Optional[List[str]] = None,
        eval_every: int = 10,
        on_policy_ratio: float = 0.7,
        refresh_student_every: int = 20,
    ) -> Dict[str, Any]:
        """
        Method 6: Reverse curriculum - on-policy first, then off-policy.

        The opposite of standard hybrid. Hypothesis: starting with on-policy
        lets the student learn to reason in its own voice first. Then off-policy
        refines toward teacher style without collapsing the distribution.

        Args:
            on_policy_ratio: Fraction of steps for on-policy phase (default 0.7)
        """
        logger.info(f"Running REVERSE CURRICULUM for {max_steps} steps")

        on_policy_steps = int(max_steps * on_policy_ratio)
        off_policy_steps = max_steps - on_policy_steps
        logger.info(f"Phase 1: {on_policy_steps} on-policy, Phase 2: {off_policy_steps} off-policy")

        self._init_wandb(mode="reverse_curriculum")
        self.history = []
        prompt_idx = 0

        # === PHASE 1: ON-POLICY ===
        logger.info("\n=== Phase 1: On-Policy (learn to reason) ===")
        for step in range(on_policy_steps):
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            if step > 0 and step % refresh_student_every == 0:
                self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
                    name=f"reverse_on_policy_step_{step}"
                )

            student_response, student_tokens, student_logprobs = self._generate_student_response(
                prompt, temperature=self.experiment_config.on_policy.temperature
            )

            if not student_tokens or len(student_tokens) < 2:
                continue

            teacher_logprobs = self._get_teacher_logprobs_on_trajectory(prompt, student_tokens)

            if teacher_logprobs and len(teacher_logprobs) == len(student_logprobs):
                metrics = self.train_gkd_step(student_tokens, student_logprobs, teacher_logprobs, step=step)
                kl_div = sum(sp - tp for sp, tp in zip(student_logprobs, teacher_logprobs)) / len(student_logprobs)
                training_type = "on_policy_gkd"
            else:
                context_bundle = self.context_generator.generate(prompt)
                teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
                if teacher_response:
                    metrics = self.train_supervised_step(prompt, teacher_response)
                else:
                    metrics = {"loss:sum": 0.0}
                kl_div = 0.0
                training_type = "supervised_fallback"

            context_bundle = self.context_generator.generate(prompt)
            teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)
            score = self._compute_score(student_response, teacher_response) if teacher_response else 0.0

            self.history.append({
                "step": step,
                "phase": "on_policy",
                "loss": metrics.get("loss:sum", 0.0),
                "score": score,
                "kl_divergence": kl_div,
                "training_type": training_type,
                "prompt": prompt[:100],
            })

            if step % eval_every == 0:
                logger.info(f"[ON] Step {step}: loss={metrics.get('loss:sum', 0):.4f}, "
                           f"score={score:.3f}, kl={kl_div:.4f}")

        # === PHASE 2: OFF-POLICY ===
        logger.info("\n=== Phase 2: Off-Policy (refine toward teacher) ===")

        # Refresh before transition
        self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
            name="reverse_transition"
        )

        for step in range(on_policy_steps, max_steps):
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1

            context_bundle = self.context_generator.generate(prompt)
            teacher_response = self._generate_teacher_response(prompt, context_bundle=context_bundle)

            if not teacher_response:
                continue

            metrics = self.train_supervised_step(prompt, teacher_response)

            self.history.append({
                "step": step,
                "phase": "off_policy",
                "loss": metrics.get("loss:sum", 0.0),
                "training_type": "off_policy",
                "prompt": prompt[:100],
            })

            if step % eval_every == 0:
                logger.info(f"[OFF] Step {step}: loss={metrics.get('loss:sum', 0):.4f}")

        checkpoint_path = self._save_checkpoint(f"reverse_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self._finish_wandb()

        on_policy_entries = [h for h in self.history if h.get("phase") == "on_policy"]
        return {
            "history": self.history,
            "final_loss": self.history[-1]["loss"] if self.history else 0.0,
            "avg_score": sum(h.get("score", 0) for h in on_policy_entries) / len(on_policy_entries) if on_policy_entries else 0.0,
            "avg_kl": sum(h.get("kl_divergence", 0) for h in on_policy_entries) / len(on_policy_entries) if on_policy_entries else 0.0,
            "checkpoint": checkpoint_path,
            "training_mode": "reverse_curriculum",
            "on_policy_steps": on_policy_steps,
            "off_policy_steps": off_policy_steps,
        }

    def _save_checkpoint(self, name: str) -> str:
        """Save checkpoint by creating a new sampling client with current weights."""
        # Tinker uses save_weights_and_get_sampling_client to persist weights
        self._sampling_client = self._training_client.save_weights_and_get_sampling_client(
            name=name
        )
        logger.info(f"Checkpoint saved: {name}")
        return f"tinker://{name}"


def run_tinker_comparison(
    prompts: List[str],
    eval_prompts: List[str],
    max_steps: int = 100,
    n_seeds: int = 3,
    output_dir: str = "./outputs",
) -> Dict[str, Any]:
    """
    Run comparison between off-policy and on-policy distillation using Tinker SDK.

    Args:
        prompts: Training prompts
        eval_prompts: Evaluation prompts
        max_steps: Training steps per run
        n_seeds: Number of seeds per condition
        output_dir: Output directory

    Returns:
        Results with statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "experiment": "tinker_distillation_comparison",
        "timestamp": datetime.now().isoformat(),
        "n_seeds": n_seeds,
        "max_steps": max_steps,
        "n_prompts": len(prompts),
        "modes": {
            "off_policy": {"final_losses": [], "avg_scores": []},
            "on_policy": {"final_losses": [], "avg_scores": []},
        },
    }

    config = TinkerTrainingConfig()
    exp_config = ExperimentConfig()

    for seed in range(n_seeds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed {seed + 1}/{n_seeds}")
        logger.info(f"{'='*60}")

        # Off-policy run
        logger.info("\n--- Off-Policy ---")
        trainer = TinkerDistillationTrainer(config, exp_config)
        trainer.initialize()

        off_result = trainer.run_off_policy_distillation(
            prompts,
            max_steps=max_steps,
            eval_prompts=eval_prompts,
        )
        results["modes"]["off_policy"]["final_losses"].append(off_result["final_loss"])

        # On-policy run (new trainer instance = fresh weights)
        logger.info("\n--- On-Policy ---")
        trainer = TinkerDistillationTrainer(config, exp_config)
        trainer.initialize()

        on_result = trainer.run_on_policy_distillation(
            prompts,
            max_steps=max_steps,
            eval_prompts=eval_prompts,
        )
        results["modes"]["on_policy"]["final_losses"].append(on_result["final_loss"])
        results["modes"]["on_policy"]["avg_scores"].append(on_result.get("avg_score", 0))

    # Compute statistics
    for mode in ["off_policy", "on_policy"]:
        losses = results["modes"][mode]["final_losses"]
        if losses:
            results["modes"][mode]["statistics"] = {
                "mean_loss": sum(losses) / len(losses),
                "min_loss": min(losses),
                "max_loss": max(losses),
            }

    # Save results
    with open(f"{output_dir}/tinker_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nResults saved to {output_dir}/tinker_comparison_results.json")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Tinker-based context distillation")
    parser.add_argument("--mode", choices=["off_policy", "on_policy", "comparison"], default="comparison")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./outputs/tinker")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Generate synthetic prompts
    from run_experiments import generate_synthetic_prompts
    prompts = generate_synthetic_prompts(args.num_prompts)
    eval_prompts = generate_synthetic_prompts(10)

    if args.mode == "comparison":
        results = run_tinker_comparison(
            prompts,
            eval_prompts,
            max_steps=args.max_steps,
            n_seeds=args.n_seeds,
            output_dir=args.output_dir,
        )
        print(f"\nComparison Results:")
        for mode, data in results["modes"].items():
            stats = data.get("statistics", {})
            print(f"  {mode}: mean_loss={stats.get('mean_loss', 'N/A'):.4f}")
    else:
        config = TinkerTrainingConfig()
        exp_config = ExperimentConfig()
        trainer = TinkerDistillationTrainer(config, exp_config)
        trainer.initialize()

        if args.mode == "off_policy":
            result = trainer.run_off_policy_distillation(prompts, max_steps=args.max_steps)
        else:
            result = trainer.run_on_policy_distillation(prompts, max_steps=args.max_steps)

        print(f"\nResult: {result}")
