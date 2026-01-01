"""
Configuration for Context Distillation experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ContextType(str, Enum):
    """Types of context to distill."""
    FEW_SHOT = "few_shot"
    RETRIEVAL = "retrieval"
    SYSTEM_INSTRUCTIONS = "system_instructions"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    HYBRID = "hybrid"


class TrainingMode(str, Enum):
    """Training paradigm."""
    OFF_POLICY = "off_policy"  # Standard distillation
    ON_POLICY = "on_policy"  # Student generates, teacher corrects
    HYBRID = "hybrid"  # Mix of both


class FeedbackType(str, Enum):
    """How teacher provides feedback."""
    DIRECT_CORRECTION = "direct_correction"  # Teacher provides correct answer
    PREFERENCE = "preference"  # Teacher ranks student outputs
    CRITIQUE = "critique"  # Teacher explains what's wrong
    REWARD = "reward"  # Scalar reward signal


@dataclass
class TeacherConfig:
    """Teacher model configuration.

    Available Tinker models (Dec 2025):
    - Instruction: Qwen/Qwen3-235B-A22B-Instruct-2507, meta-llama/Llama-3.3-70B-Instruct
    - Reasoning: moonshotai/Kimi-K2-Thinking, openai/gpt-oss-120b
    - Hybrid: deepseek-ai/DeepSeek-V3.1, Qwen/Qwen3-32B
    See: https://tinker-docs.thinkingmachines.ai/model-lineup
    """
    model_id: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    api_url: str = "https://api.tinker.computer/v1"
    temperature: float = 0.3  # Lower for consistency
    max_tokens: int = 1024


@dataclass
class StudentConfig:
    """Student model configuration.

    For local training (HuggingFace):
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (open, no auth needed)
    - microsoft/phi-2 (open, no auth needed)
    - Qwen/Qwen2.5-0.5B (open, no auth needed)

    For Tinker GPU training:
    - meta-llama/Llama-3.1-8B (recommended)
    - Qwen/Qwen3-8B-Base
    """
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Open model, no HF auth needed
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 1024
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class ContextConfig:
    """Context generation configuration."""
    context_type: ContextType = ContextType.SYSTEM_INSTRUCTIONS  # Default to system instructions (no example_pool required)
    num_few_shot: int = 5
    retrieval_top_k: int = 3
    system_prompt: str = ""
    max_context_tokens: int = 2048


@dataclass
class OnPolicyConfig:
    """On-policy training configuration."""
    feedback_type: FeedbackType = FeedbackType.DIRECT_CORRECTION
    num_student_samples: int = 3  # Samples per prompt for exploration
    temperature: float = 0.7  # Student sampling temperature
    use_self_consistency: bool = True
    correction_threshold: float = 0.5  # Only correct if student score < this
    kl_penalty: float = 0.1  # Penalty for diverging from base model


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    experiment_name: str = "context_distillation"
    training_mode: TrainingMode = TrainingMode.ON_POLICY

    # Sub-configurations
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    on_policy: OnPolicyConfig = field(default_factory=OnPolicyConfig)

    # Data
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    context_examples_path: Optional[str] = None
    num_train_samples: int = 1000
    num_eval_samples: int = 200

    # Output
    output_dir: str = "./outputs"
    save_checkpoints: bool = True
    log_to_wandb: bool = False
    wandb_project: str = "context-distillation"

    # Training
    max_training_steps: int = 10000
    eval_every: int = 500
    save_every: int = 1000

    # Random seed
    seed: int = 42


# Preset configurations
PRESETS: Dict[str, ExperimentConfig] = {
    "few_shot_off_policy": ExperimentConfig(
        experiment_name="few_shot_off_policy",
        training_mode=TrainingMode.OFF_POLICY,
        context=ContextConfig(
            context_type=ContextType.FEW_SHOT,
            num_few_shot=5,
        ),
    ),
    "few_shot_on_policy": ExperimentConfig(
        experiment_name="few_shot_on_policy",
        training_mode=TrainingMode.ON_POLICY,
        context=ContextConfig(
            context_type=ContextType.FEW_SHOT,
            num_few_shot=5,
        ),
    ),
    "retrieval_distillation": ExperimentConfig(
        experiment_name="retrieval_distillation",
        training_mode=TrainingMode.ON_POLICY,
        context=ContextConfig(
            context_type=ContextType.RETRIEVAL,
            retrieval_top_k=5,
        ),
    ),
    "cot_distillation": ExperimentConfig(
        experiment_name="cot_distillation",
        training_mode=TrainingMode.ON_POLICY,
        context=ContextConfig(
            context_type=ContextType.CHAIN_OF_THOUGHT,
        ),
        on_policy=OnPolicyConfig(
            feedback_type=FeedbackType.CRITIQUE,
        ),
    ),
}


def get_config(preset: Optional[str] = None, **overrides) -> ExperimentConfig:
    """Get configuration, optionally from preset with overrides."""
    if preset and preset in PRESETS:
        config = PRESETS[preset]
    else:
        config = ExperimentConfig()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
