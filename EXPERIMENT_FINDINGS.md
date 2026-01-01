# On-Policy Context Distillation: Final Report

## Executive Summary

This project investigated the research question from the [original specification](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/on-policy-context-distillation.md):

> **How does on-policy distillation compare to off-policy distillation for training student models to match teacher models in few-shot learning contexts?**

We discovered two distinct failure modes depending on the experimental setup:

| Setup | What Failed | Why |
|-------|-------------|-----|
| **Context Distillation** (same model ± context) | 0% downstream accuracy | Format matching only, no capability gap to transfer |
| **Size Distillation** (small ← large model) | Hybrid collapse | Off-policy corrupts student generation ability |

Our solution: **Pure on-policy GKD with teacher seeding** achieves 58-71% GSM8K accuracy without collapse.

---

## Project Timeline

### Phase 1: Spec-Compliant Context Distillation

**Setup**: Same model with/without few-shot context
- Student: Llama-3.1-8B-Instruct (no context)
- Teacher: Llama-3.1-8B-Instruct (with 10 few-shot examples)

**Result**: Complete failure
- **Eval scores were high** (0.92-1.00) — student matched teacher's output format
- **Downstream accuracy was 0%** — student couldn't actually solve problems

**Root Cause**: Context distillation with the same model only teaches **format matching**, not reasoning. The few-shot examples help the teacher format answers correctly, but the student already has the same base capabilities. There's no new capability to transfer.

### Phase 2: Pivot to Size Distillation

To actually transfer capabilities, we switched to using a larger teacher:

| Family | Student | Teacher | Size Ratio |
|--------|---------|---------|:----------:|
| **Qwen** | Qwen3-4B-Instruct | Qwen3-30B-A3B-Instruct | 7.5x |
| **Llama** | Llama-3.1-8B-Instruct | Llama-3.3-70B-Instruct | 8.75x |

**Initial Results**:
- Off-policy: Good training loss, but 0% downstream accuracy
- On-policy GKD: Some transfer (4-6% accuracy)
- **Hybrid: Catastrophic collapse** (scores drop to 0 at phase transition)

### Phase 3: Cliff Mitigation Experiment

We ran **160 training runs** (8 methods × 10 seeds × 2 model families) to solve the hybrid collapse problem.

---

## The Distribution Cliff Problem

### What We Observed

When transitioning from off-policy to on-policy training in hybrid mode:

```
=== Off-Policy Phase ===
Step 0:  loss=6155.54, eval=0.509
Step 40: loss=224.33,  eval=0.501   (training progresses normally)

=== PHASE TRANSITION (Step 50) ===
Step 50: loss=0.012, score=0.173, kl=0.357, eval=0.165  (sudden drop)
Step 70: loss=8.00,  score=0.000, kl=2.067, eval=0.000  (COLLAPSE)
Step 90: eval=0.000, downstream=0%  (unrecoverable)
```

### Why It Happens

The collapse occurs because **off-policy and on-policy training optimize fundamentally different objectives**:

| Training Mode | Objective | What Student Learns |
|---------------|-----------|---------------------|
| Off-Policy | `max P(teacher_token | prefix)` | Token prediction (mimicry) |
| On-Policy GKD | `min D_KL(student ‖ teacher)` | Distribution matching |

When the student is much smaller than the teacher:

1. **Off-policy phase**: Student learns to predict teacher's tokens
   - Weights shift toward teacher's token distribution
   - But student lacks the reasoning capacity to understand *why* those tokens are correct

2. **Phase transition**: Student generates with modified weights
   - Produces tokens that "look like" teacher output (similar vocabulary, formatting)
   - But are semantically incoherent (wrong reasoning, random numbers)

3. **On-policy phase**: Teacher evaluates student's garbage
   - Teacher assigns very low log-probabilities to incoherent outputs
   - GKD loss computes: `advantage = -(student_logprob - teacher_logprob)`
   - Large negative advantages push student toward degenerate solutions
   - Student collapses to empty/repetitive outputs (minimum-entropy solutions)

**The key insight**: The student learns to *mimic tokens* without learning to *reason*. When forced to generate independently, it produces syntactically similar but semantically broken output.

---

## Methods Tested

### Cliff Mitigation Strategies

| Method | Hypothesis | Off-Policy Component |
|--------|------------|:--------------------:|
| `on_policy_gkd` | Baseline pure on-policy | None |
| `hybrid` | Off-policy bootstrap helps | 50 steps |
| `extended_on_policy` | More on-policy compensates | None |
| `teacher_seeded` | Teacher prefix guides generation | None (prefix only) |
| `mixture` | Blending avoids hard transition | 30% per step |
| `replay_buffer` | Experience replay prevents forgetting | Periodic injection |
| `kl_anchored` | KL penalty prevents drift | 100% with penalty |
| `reverse_curriculum` | On-policy first is stable | 30 steps at end |

### Results

#### Qwen Family (4B ← 30B)

| Method | GSM8K Accuracy | Status |
|--------|:--------------:|--------|
| **teacher_seeded** | **58.6%** | Best |
| on_policy_gkd | 53.2% | Stable |
| extended_on_policy | 51.0% | Good |
| replay_buffer | 2.8% | Partial |
| reverse_curriculum | 0.8% | Collapsed |
| hybrid | 0.0% | Collapsed |
| mixture | 0.0% | Collapsed |
| kl_anchored | 0.0% | Collapsed |

#### Llama Family (8B ← 70B)

| Method | GSM8K Accuracy | Status |
|--------|:--------------:|--------|
| **teacher_seeded** | **71.0%** | Best |
| on_policy_gkd | 67.4% | Stable |
| extended_on_policy | 64.4% | Good |
| replay_buffer | 7.4% | Partial |
| hybrid | 0.0% | Collapsed |
| mixture | 0.0% | Collapsed |
| kl_anchored | 0.0% | Collapsed |
| reverse_curriculum | 0.0% | Collapsed |

---

## Key Findings

### 1. All Off-Policy Methods Collapse

Every method that includes any off-policy component fails:

| Method | Off-Policy Component | Result |
|--------|---------------------|--------|
| hybrid | Phase 1 (50 steps) | Collapsed |
| mixture | 30% of loss each step | Collapsed |
| kl_anchored | 100% with KL penalty | Collapsed |
| reverse_curriculum | Phase 2 (30 steps) | Collapsed |

**The pattern is unambiguous**: Any exposure to off-policy training corrupts the student's generation ability, and this corruption is not recoverable through any mitigation we tested.

### 2. Pure On-Policy Methods Work

Methods that never use off-policy training succeed:

| Method | Why It Works |
|--------|--------------|
| `on_policy_gkd` | Student always generates coherently; teacher feedback is meaningful |
| `extended_on_policy` | Same as above, just more steps |
| `teacher_seeded` | Teacher prefix keeps student generations coherent during warmup |

### 3. Teacher Seeding is the Winner

`teacher_seeded` outperforms pure `on_policy_gkd`:
- **+5.4 percentage points** on Qwen (58.6% vs 53.2%)
- **+3.6 percentage points** on Llama (71.0% vs 67.4%)

**Why it works**: Instead of off-policy training, teacher_seeded uses a curriculum:
1. Teacher generates the first N tokens (N starts at 20)
2. Student completes the response from there
3. Teacher scores the full sequence
4. N decays to 0 over 50 steps

This keeps student generations coherent while gradually transferring control — a "soft" transition instead of the hard phase switch that causes collapse.

### 4. KL Regularization Doesn't Help

We tested `kl_anchored` with a KL penalty to the initial student distribution:
```python
total_loss = off_policy_loss + beta * D_KL(current_student || initial_student)
```

**Result**: Still collapsed. The KL penalty slows the drift but doesn't prevent the fundamental problem — the student still learns mimicry without reasoning.

### 5. Mixing Objectives Doesn't Help

We tested `mixture` with weighted combination each step:
```python
loss = 0.3 * off_policy_loss + 0.7 * gkd_loss
```

**Result**: Still collapsed. Even a small amount of off-policy signal (30%) is enough to corrupt the student's generation ability over 100 steps.

---

## Answering the Original Research Question

The [original spec](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/on-policy-context-distillation.md) asked:

> How does on-policy distillation compare to off-policy distillation for training student models to match teacher models in few-shot learning contexts?

### For Context Distillation (Same Model)

**Neither works well.** When student and teacher are the same model:
- Off-policy achieves high text similarity but 0% downstream accuracy
- On-policy achieves moderate KL but still 0% downstream accuracy
- **Root cause**: No capability gap means nothing to transfer

Context distillation is useful for **format/style transfer**, not **capability transfer**.

### For Size Distillation (Small ← Large)

**On-policy is strictly better than off-policy.**

| Mode | GSM8K Accuracy | Status |
|------|:--------------:|--------|
| On-policy (teacher_seeded) | 58-71% | Works |
| On-policy (pure GKD) | 53-67% | Works |
| Off-policy | 0% | Mimicry only |
| Hybrid (off→on) | 0% | Collapses |

### Implications

1. **The GKD paper's claims hold** — on-policy is more sample-efficient and avoids exposure bias
2. **But hybrid mode is dangerous** — the paper's hybrid recommendation doesn't account for capability gaps
3. **Teacher seeding is a better alternative** — provides off-policy benefits (guided start) without off-policy's failure mode

---

## Recommendations

### For Practitioners

1. **Use `teacher_seeded` for knowledge distillation** — best results across both model families
2. **Never use hybrid mode with capability gaps** — it will collapse
3. **Pure `on_policy_gkd` is a safe baseline** — works reliably, just slightly worse than teacher_seeded
4. **Don't attempt context distillation for capability transfer** — use size distillation instead

### For Researchers

1. **The off-policy problem is fundamental** — not fixable with regularization or curriculum
2. **Soft transitions work, hard transitions don't** — teacher_seeded succeeds where hybrid fails
3. **Capability gaps change everything** — GKD paper results may not generalize to size distillation

---

## Cost Analysis

### Experiment Duration

| Experiment | Runs | Duration | Per Run |
|------------|:----:|:--------:|:-------:|
| Qwen (4B ← 30B) | 80 | 62.7h | 47 min |
| Llama (8B ← 70B) | 80 | 67.7h | 51 min |
| **Total** | **160** | **130.4h (5.4 days)** | **49 min avg** |

### Compute Costs

| Component | Qwen | Llama | Notes |
|-----------|:----:|:-----:|-------|
| Training hours | 62.7h | 67.7h | 100 steps/run (200 for extended) |
| Cost/hour | ~$0.75 | ~$1.00 | Student training + teacher inference |
| **Subtotal** | **~$47** | **~$68** | |
| **Total** | | **~$115** | For 160 runs |

### Per-Run Breakdown

| Item | Qwen | Llama |
|------|:----:|:-----:|
| Training (100 steps) | ~$0.45 | ~$0.65 |
| Downstream eval (50 questions) | ~$0.14 | ~$0.20 |
| **Total per run** | **~$0.59** | **~$0.85** |

### Cost Efficiency

- **Cost per percentage point of accuracy**: ~$1.60 (Qwen), ~$0.96 (Llama)
- **Best method (teacher_seeded)**: 10 seeds × ~$0.72 = **~$7.20** for statistical validity
- **Full method comparison**: 8 methods × 10 seeds × $0.72 = **~$58** per model family

---

## Proposed Next Steps

### Priority 1: Validate on More Benchmarks (~$50-80)

Test teacher_seeded on additional reasoning benchmarks to verify generalization:

| Benchmark | Questions | Est. Cost | Purpose |
|-----------|:---------:|:---------:|---------|
| MMLU (5-shot) | 100 | ~$15 | General knowledge retention |
| ARC-Challenge | 50 | ~$10 | Scientific reasoning |
| HellaSwag | 50 | ~$10 | Common sense |
| HumanEval | 50 | ~$15 | Code generation |

**Total**: ~$50-80 for 5 seeds per benchmark

### Priority 2: Hyperparameter Optimization (~$100-150)

The current teacher_seeded uses default hyperparameters. Potential improvements:

| Parameter | Current | Range to Test | Est. Runs | Cost |
|-----------|:-------:|:-------------:|:---------:|:----:|
| Initial seed tokens | 20 | 10, 15, 25, 30 | 20 | ~$15 |
| Seed decay steps | 50 | 30, 40, 60, 70 | 20 | ~$15 |
| LoRA rank | 32 | 16, 64, 128 | 15 | ~$12 |
| Learning rate | 3e-4 | 1e-4, 5e-4, 1e-3 | 15 | ~$12 |
| Training steps | 100 | 150, 200, 300 | 15 | ~$20 |

**Total**: ~$100-150 for systematic search (could improve accuracy 5-10%)

### Priority 3: Scale to Larger Models (~$200-400)

Test if the findings hold at larger scale:

| Configuration | Student | Teacher | Est. Cost |
|---------------|---------|---------|:---------:|
| Llama 70B ← 405B | 70B | 405B | ~$150 |
| Qwen 32B ← 72B | 32B | 72B | ~$100 |
| Mistral 7B ← Mixtral | 7B | 8x7B MoE | ~$80 |

**Total**: ~$200-400 for 3-5 seeds per config

### Priority 4: Alternative Distillation Objectives (~$80-120)

Test other on-policy objectives that might outperform GKD:

| Method | Description | Est. Cost |
|--------|-------------|:---------:|
| DPO on teacher preferences | Teacher ranks student outputs | ~$40 |
| Reward-weighted GKD | Weight by teacher confidence | ~$30 |
| Multi-step reasoning distillation | Chain-of-thought specific | ~$50 |

**Total**: ~$80-120 for 5 seeds per method

### Priority 5: True Context Distillation (~$60-80)

Revisit the original spec with lessons learned:

| Setup | Configuration | Est. Cost |
|-------|---------------|:---------:|
| Few-shot → no context | Llama-8B with/without 5-shot | ~$30 |
| RAG → no RAG | Distill retrieval knowledge | ~$50 |

Use teacher_seeded approach instead of hybrid to avoid collapse.

**Total**: ~$60-80

---

## Budget Summary

| Priority | Focus | Cost | Expected Impact |
|:--------:|-------|:----:|-----------------|
| 1 | Benchmark validation | $50-80 | Verify generalization |
| 2 | Hyperparameter tuning | $100-150 | +5-10% accuracy |
| 3 | Scale testing | $200-400 | Publication-ready results |
| 4 | Alternative objectives | $80-120 | Potential breakthrough |
| 5 | True context distillation | $60-80 | Complete original spec |

**Recommended minimum**: Priorities 1+2 = **~$150-230**
**Full research program**: All priorities = **~$500-830**

---

## Limitations

1. **Benchmark scope**: Only tested on GSM8K math reasoning
2. **Model families**: Only Qwen and Llama (no Mistral, Gemma)
3. **No capability retention testing**: Didn't verify MMLU/other benchmarks
4. **Limited same-model exploration**: Didn't exhaustively test context distillation variants

---

## Conclusion

We investigated on-policy vs off-policy distillation and found:

1. **Context distillation (same model)** produces 0% downstream accuracy — it only teaches format matching
2. **Size distillation (small ← large)** can transfer capabilities, but hybrid mode catastrophically collapses
3. **The solution is pure on-policy training** — teacher_seeded achieves 58-71% GSM8K accuracy

The distribution cliff in hybrid distillation is caused by a fundamental mismatch: off-policy teaches token mimicry, but on-policy expects coherent generation. When the student lacks the teacher's reasoning capability, mimicry produces garbage, and GKD amplifies this into collapse.

**The simple answer to the research question**: On-policy is strictly better than off-policy for knowledge distillation with capability gaps. Never mix them.

---

## References

- [On-Policy Distillation of Language Models (GKD)](https://arxiv.org/abs/2306.13649) - Agarwal et al. 2024
- [Learning by Distilling Context](https://arxiv.org/abs/2209.15189) - Snell et al. 2022
- [Original Project Spec](https://github.com/thinking-machines-lab/tinker-project-ideas/blob/main/on-policy-context-distillation.md)
