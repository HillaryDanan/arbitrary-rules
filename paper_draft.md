# Arbitrary Rule Following Reveals Compositional Limits in Large Language Models

Hillary Danan¹*

¹ Independent Researcher

*Correspondence: hillarydanan@gmail.com

---

## Abstract

Large language models (LLMs) exhibit failures on tasks requiring compositional generalization. However, standard benchmarks may overlap with training distributions. We introduce an **arbitrary rule following** paradigm using procedurally-generated rules that combine random conditions and actions, minimizing training coverage. Across two models tested at high power (Claude Sonnet 4, GPT-4o; N=600 trials each, 120/level), we find one robust result: **input-dependent rules** (where output constraints are computed from input properties) achieve only 7-10% accuracy, while simpler rules achieve 74-83%. However, we also find substantial run-to-run variance (±20-40 percentage points on some levels), and the relative difficulty of different rule types is inconsistent across runs. The input-dependent constraint finding replicates; other findings require larger samples or different designs to stabilize.

---

## Introduction

A defining feature of human cognition is **compositional generalization**—the capacity to understand and produce novel combinations from known components (Fodor & Pylyshyn, 1988; Lake & Baroni, 2018). Large language models (LLMs) have achieved remarkable performance on diverse benchmarks, leading to debates about whether they exhibit genuine compositional abilities or sophisticated pattern matching (Brown et al., 2020; Wei et al., 2022).

A key challenge in evaluating compositional generalization is **training contamination**: any benchmark composed of natural language may overlap with training distributions. We address this with an **arbitrary rule following** paradigm using procedurally generated rules from random combinations of conditions and actions.

We test frontier LLMs across five complexity levels, from single unconditional actions to input-dependent rules requiring computed input properties to constrain output structure. We report both initial results and a high-power replication, revealing which findings are robust and which are unstable.

---

## Results

### Experimental Design

We generated rules across five complexity levels:

| Level | Structure | Active Constraints | Example |
|-------|-----------|-------------------|---------|
| 1 | Single unconditional action | 1 | "Always begin with 'QUACK'" |
| 2 | Two unconditional actions | 2 | "Respond in exactly 5 words AND end with '...maybe'" |
| 3 | IF-ELSE conditional | 1 per branch | "If prompt contains 'the', use ALL CAPS; otherwise end with '- done'" |
| 4 | IF-ELIF-ELSE priority | 1 per branch | Three conditions with priority ordering |
| 5 | Input-dependent | 1 (computed) | "Respond in N words where N = vowels in prompt" |

We ran two rounds:
- **Initial run**: 5 models, 60 trials/level (N=300 per model)
- **Ultra-power replication**: 2 models, 120 trials/level (N=600 per model)

### Initial Results (N=300 per model)

| Model | L1 | L2 | L3 | L4 | L5 | Overall |
|-------|-----|-----|-----|-----|-----|---------|
| GPT-4o Mini | 85.0% | 61.7% | 73.3% | 58.3% | 18.3% | 59.3% |
| GPT-4o | 91.7% | 63.3% | 78.3% | 41.7% | 5.0% | 56.0% |
| GPT-4 Turbo | 80.0% | 50.0% | 71.7% | 40.0% | 21.7% | 52.7% |
| Claude Sonnet 4 | 93.3% | 41.7% | 41.7% | 35.0% | 8.3% | 43.3% |
| Claude 3.5 Haiku | 91.7% | 20.0% | 31.7% | 20.0% | 0.0% | 32.7% |

### Ultra-Power Replication (N=600 per model)

| Model | L1 | L2 | L3 | L4 | L5 | Overall |
|-------|-----|-----|-----|-----|-----|---------|
| Claude Sonnet 4 | 74.2% | 32.5% | 40.8% | 40.0% | 10.0% | 39.5% |
| GPT-4o | 83.3% | 62.5% | 38.3% | 30.0% | 6.7% | 44.2% |

### Cross-Run Comparison

| Model | Level | Initial (N=60) | Ultra (N=120) | Difference |
|-------|-------|----------------|---------------|------------|
| Claude Sonnet 4 | L1 | 93.3% | 74.2% | **-19.1pp** |
| Claude Sonnet 4 | L2 | 41.7% | 32.5% | -9.2pp |
| Claude Sonnet 4 | L3 | 41.7% | 40.8% | -0.9pp |
| Claude Sonnet 4 | L5 | 8.3% | 10.0% | +1.7pp |
| GPT-4o | L1 | 91.7% | 83.3% | -8.4pp |
| GPT-4o | L2 | 63.3% | 62.5% | -0.8pp |
| GPT-4o | L3 | 78.3% | 38.3% | **-40.0pp** |
| GPT-4o | L5 | 5.0% | 6.7% | +1.7pp |

### Finding 1: Input-Dependent Constraints Are Robustly Hard

**This is the only finding that clearly replicates.**

| Model | L5 Initial | L5 Ultra | Range |
|-------|------------|----------|-------|
| Claude Sonnet 4 | 8.3% | 10.0% | 8-10% |
| GPT-4o | 5.0% | 6.7% | 5-7% |

Across all runs and models, Level 5 (input-dependent) accuracy ranges from 0-22%, with most values under 15%. This finding is stable.

### Finding 2: High Variance at Other Levels

Performance at Levels 1-4 shows substantial run-to-run variance:

| Level | Observed Range | Variance Assessment |
|-------|----------------|---------------------|
| L1 | 74-93% | **High variance** (19pp range for same model) |
| L2 | 20-63% | High variance |
| L3 | 32-78% | **Very high variance** (40pp drop for GPT-4o) |
| L4 | 20-58% | High variance |
| L5 | 0-22% | **Low variance** (stable) |

### Finding 3: L2 vs L3 Relationship Is Unstable

Initial data suggested most models find L2 (two constraints) harder than L3 (one conditional constraint). The replication shows this is **unstable**:

| Model | Initial L2 vs L3 | Ultra L2 vs L3 | Stable? |
|-------|------------------|----------------|---------|
| Claude Sonnet 4 | L2 = L3 (41.7% = 41.7%) | L2 < L3 (32.5% < 40.8%) | **Changed** |
| GPT-4o | L2 < L3 (63.3% < 78.3%) | L2 > L3 (62.5% > 38.3%) | **Reversed** |

The L2 vs L3 comparison is not reliable. We cannot make claims about coordination vs. conditional difficulty.

### Finding 4: L1 "High Success" Is Overstated

Initial data suggested L1 achieves 80-93%. The replication shows:
- Claude Sonnet 4 dropped from 93.3% to 74.2%
- GPT-4o dropped from 91.7% to 83.3%

The claim that "single-action rules achieve high success" is weaker than initially reported. 74% is substantially below the 90%+ initially observed.

---

## Discussion

### What Replicates

**Input-dependent constraints (L5) are hard.** This is the clearest finding:
- Range: 0-22% across all models and runs
- Most values: 5-15%
- Stable across initial and replication runs

This suggests that rules requiring computation over inputs (e.g., "respond in N words where N = vowels in prompt") are substantially harder than static rules—a robust finding.

### What Does Not Replicate

**Specific accuracy values.** Run-to-run variance of 10-40 percentage points means point estimates should not be trusted. Ranges are more appropriate.

**L2 vs L3 relationship.** The pattern reversed for GPT-4o between runs. We cannot determine whether coordination is harder than conditionals.

**L1 "high success."** Initial values (90%+) did not replicate (74-83%). The L1 vs L5 contrast remains, but L1 performance is more variable than expected.

### Why High Variance?

Several factors may contribute:

1. **Rule sampling**: Different random rules may have different intrinsic difficulty. Even with the same seed, the expanded rule set (24 vs 12 rules/level) samples different combinations.

2. **Model updates**: API models may be updated between runs (we cannot verify version stability).

3. **Temperature/sampling**: Default API parameters may introduce variance we don't control.

4. **Insufficient power**: Even 120 trials/level may be insufficient for stable estimates on high-variance tasks.

### Relation to Prior Work

Our findings partially align with Lake & Baroni's (2018) demonstration that neural models fail at compositional generalization. The L5 finding—that input-dependent constraints are hard—is consistent with this. However, the high variance we observe suggests caution in interpreting specific patterns.

The instability we document may be relevant to discussions of benchmark reliability (Schaeffer et al., 2023). If single-run results vary by 20-40 percentage points, published benchmarks using similar sample sizes may be unreliable.

### Limitations

**Variance**: The primary limitation. Run-to-run variance limits confidence in all findings except L5.

**Design confounds**: L2 vs L3 confounds constraint count with logical structure. This is now moot given instability.

**Model versions**: We cannot verify API model stability between runs.

**Rule sampling**: Different rule counts between runs (12 vs 24/level) may contribute to variance.

---

## Methods

### Rule Generation

Rules were generated procedurally (random seed = 42) from 9 condition types and 8 action types. Parameters sampled uniformly from predefined pools.

### Complexity Levels

| Level | Logic | Active constraints |
|-------|-------|-------------------|
| 1 | Unconditional | 1 |
| 2 | Unconditional | 2 (simultaneous) |
| 3 | IF-ELSE | 1 per branch |
| 4 | IF-ELIF-ELIF-ELSE | 1 per branch |
| 5 | Input-dependent | 1 (computed from input) |

### Evaluation

Programmatic compliance checking: substring match for text insertion, character checking for case transformation, token count for word limits.

### Models Tested

- Claude Sonnet 4 (Anthropic)
- Claude 3.5 Haiku (Anthropic)
- GPT-4o, GPT-4o Mini, GPT-4 Turbo (OpenAI)

---

## Conclusion

The arbitrary rule following paradigm reveals one robust finding: **input-dependent constraints are substantially harder than static rules** (5-15% vs. variable but higher). This replicates across models and runs.

Other findings are less stable. Run-to-run variance of 10-40 percentage points means that specific accuracy values, and comparisons between intermediate levels, should be interpreted cautiously.

We recommend:
1. **Report ranges**, not point estimates
2. **Replicate** before claiming specific patterns
3. **Focus on the L5 finding**—it's robust
4. **Treat other comparisons as exploratory**

**Data and code**: https://github.com/HillaryDanan/arbitrary-rules

---

## References

Brown, T., et al. (2020). Language models are few-shot learners. *NeurIPS*.

Fodor, J. A., & Pylyshyn, Z. W. (1988). Connectionism and cognitive architecture. *Cognition*.

Lake, B., & Baroni, M. (2018). Generalization without systematicity. *ICML*.

Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? *NeurIPS*.

Wei, J., et al. (2022). Emergent abilities of large language models. *TMLR*.

---

## Supplementary Materials

### Table S1: All Results Summary

| Model | Run | N/level | L1 | L2 | L3 | L4 | L5 |
|-------|-----|---------|-----|-----|-----|-----|-----|
| Claude Sonnet 4 | Initial | 60 | 93.3% | 41.7% | 41.7% | 35.0% | 8.3% |
| Claude Sonnet 4 | Ultra | 120 | 74.2% | 32.5% | 40.8% | 40.0% | 10.0% |
| GPT-4o | Initial | 60 | 91.7% | 63.3% | 78.3% | 41.7% | 5.0% |
| GPT-4o | Ultra | 120 | 83.3% | 62.5% | 38.3% | 30.0% | 6.7% |
| GPT-4o Mini | Initial | 60 | 85.0% | 61.7% | 73.3% | 58.3% | 18.3% |
| GPT-4 Turbo | Initial | 60 | 80.0% | 50.0% | 71.7% | 40.0% | 21.7% |
| Claude 3.5 Haiku | Initial | 60 | 91.7% | 20.0% | 31.7% | 20.0% | 0.0% |

### Table S2: Variance Analysis

| Level | Min | Max | Range | Assessment |
|-------|-----|-----|-------|------------|
| L1 | 74.2% | 93.3% | 19.1pp | Unstable |
| L2 | 20.0% | 63.3% | 43.3pp | Very unstable |
| L3 | 31.7% | 78.3% | 46.6pp | Very unstable |
| L4 | 20.0% | 58.3% | 38.3pp | Very unstable |
| L5 | 0.0% | 21.7% | 21.7pp | Moderate (but all values low) |

### Table S3: L5 Stability Check

| Model | Run 1 | Run 2 | Difference |
|-------|-------|-------|------------|
| Claude Sonnet 4 | 8.3% | 10.0% | +1.7pp |
| GPT-4o | 5.0% | 6.7% | +1.7pp |

L5 shows the smallest cross-run variance, supporting its robustness as the key finding.
