# On-Policy Context Distillation

## Research Question

From the [original project spec](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/on-policy-context-distillation.md):

> How does on-policy distillation compare to off-policy distillation for training student models to match teacher models in few-shot learning contexts?

## TL;DR

**On-policy is strictly better. Off-policy causes catastrophic collapse.**

We tested two setups:

| Setup | Result |
|-------|--------|
| **Context distillation** (same model ± context) | 0% downstream — only teaches format matching |
| **Size distillation** (small ← large model) | Hybrid collapses; pure on-policy works |

**Winner**: `teacher_seeded` — on-policy GKD with decaying teacher prefix tokens (58-71% GSM8K accuracy).

See [EXPERIMENT_FINDINGS.md](./EXPERIMENT_FINDINGS.md) for the full analysis.

---

## Final Results

| Method | Qwen (4B←30B) | Llama (8B←70B) | Off-Policy? |
|--------|:-------------:|:--------------:|:-----------:|
| **teacher_seeded** | **58.6%** | **71.0%** | No |
| on_policy_gkd | 53.2% | 67.4% | No |
| extended_on_policy | 51.0% | 64.4% | No |
| replay_buffer | 2.8% | 7.4% | Partial |
| hybrid (off→on) | 0.0% | 0.0% | Yes |
| mixture (blend) | 0.0% | 0.0% | Yes |
| kl_anchored | 0.0% | 0.0% | Yes |
| reverse_curriculum | 0.0% | 0.0% | Yes |

**Pattern**: Every method with any off-policy component collapses. Pure on-policy works.

---

## The Distribution Cliff Problem

When transitioning from off-policy to on-policy training:

```
Step 0-49:   Off-policy (supervised on teacher outputs)
Step 50:     PHASE TRANSITION
             → Student generates with modified weights
             → Produces "teacher-like garbage" (similar tokens, broken reasoning)
             → Teacher assigns low logprobs
             → GKD pushes student toward degenerate solutions
Step 50-100: COLLAPSE (scores drop to 0%, unrecoverable)
```

**Root cause**: Off-policy teaches **token mimicry** without **reasoning**. When forced to generate independently, the student produces syntactically similar but semantically broken output.

---

## Why Teacher Seeding Works

`teacher_seeded` provides off-policy benefits without off-policy's failure mode:

```
Step 0:   [Teacher: 20 tokens] [Student: rest]
Step 25:  [Teacher: 10 tokens] [Student: rest]
Step 50:  [Teacher: 0 tokens]  [Student: all]
```

- Teacher prefix keeps student generations coherent
- Student learns from meaningful teacher feedback
- Gradual handoff prevents the hard phase transition that causes collapse

---

## Quick Start

```bash
# Run with Qwen (recommended — best model family)
python run_cliff_mitigation_experiment.py --model-family qwen --n-seeds 10

# Run with Llama
python run_cliff_mitigation_experiment.py --model-family llama --n-seeds 10

# Quick test (3 seeds)
python run_cliff_mitigation_experiment.py --quick-test

# Run specific methods
python run_cliff_mitigation_experiment.py --methods teacher_seeded on_policy_gkd
```

---

## Project Structure

```
context-distillation/
├── README.md                           # This file
├── EXPERIMENT_FINDINGS.md              # Full analysis with project history
├── run_cliff_mitigation_experiment.py  # Cliff mitigation experiment (8 methods)
├── run_tinker_experiment.py            # Original 3-mode experiment
├── tinker_trainer.py                   # Tinker SDK trainer with all methods
├── config.py                           # Configuration classes
├── context_generator.py                # Few-shot context generation
├── requirements.txt                    # Python dependencies
└── .env.example                        # Environment variable template
```

---

## Model Families

| Family | Student | Teacher | Gap |
|--------|---------|---------|:---:|
| `qwen` | Qwen3-4B-Instruct | Qwen3-30B-A3B-Instruct | 7.5x |
| `llama` | Llama-3.1-8B-Instruct | Llama-3.3-70B-Instruct | 8.75x |

**Deprecated** (0% downstream accuracy):
- `same-model-qwen` / `same-model-llama` — context distillation only teaches format matching

---

## Key Takeaways

1. **Context distillation doesn't work for capability transfer** — use size distillation instead
2. **Never mix on-policy and off-policy** — any off-policy exposure causes collapse
3. **Teacher seeding is the best approach** — soft transition without off-policy corruption
4. **The GKD paper's hybrid recommendation doesn't generalize** — capability gaps change everything

---

## References

- [On-Policy Distillation of Language Models (GKD)](https://arxiv.org/abs/2306.13649) - Agarwal et al. 2024
- [Learning by Distilling Context](https://arxiv.org/abs/2209.15189) - Snell et al. 2022
- [Original Project Spec](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/on-policy-context-distillation.md)
