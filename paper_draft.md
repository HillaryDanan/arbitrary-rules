# Arbitrary Rule Following Reveals Compositional Limits in Large Language Models

Hillary Danan¹*

¹ Independent Researcher

*Correspondence: hillarydanan@gmail.com

---

## Abstract

Large language models (LLMs) exhibit striking failures on tasks requiring compositional generalization—the ability to systematically combine known components in novel ways. However, standard benchmarks may overlap with training distributions, confounding retrieval with construction. We introduce an **arbitrary rule following** paradigm using procedurally-generated rules that combine random conditions and actions, minimizing training coverage. Across five models (GPT-4o, GPT-4o Mini, GPT-4 Turbo, Claude Sonnet 4, Claude 3.5 Haiku; N=300 trials each), we find: (1) single-action rules achieve 80-93% accuracy while self-referential rules achieve 0-22% (Cohen's h = 1.25-2.56); (2) errors are predominantly binary—rules ignored entirely rather than partially followed (81-91% across models); (3) coordinating two simultaneous actions is harder than conditional branching for most models; and (4) model scale does not predict performance—GPT-4o Mini outperforms larger models. These findings replicate across architectures and suggest that current LLMs struggle with novel constraint coordination regardless of scale, consistent with prior work on compositional generalization failures.

---

## Introduction

A defining feature of human cognition is **compositional generalization**—the capacity to understand and produce novel combinations from known components (Fodor & Pylyshyn, 1988; Lake & Baroni, 2018). This capacity enables humans to understand sentences never heard before, follow novel instructions, and generalize systematically to new situations.

Large language models (LLMs) have achieved remarkable performance on diverse benchmarks, leading to debates about whether they exhibit genuine compositional abilities or sophisticated pattern matching (Brown et al., 2020; Wei et al., 2022). Recent work suggests the latter: LLMs fail systematically on tasks requiring novel combinations, even when component skills are demonstrably present (Lake & Baroni, 2018; Dziri et al., 2023; Press et al., 2023).

A key challenge in evaluating compositional generalization is **training contamination**: any benchmark composed of natural language may overlap with training distributions. A model that appears to "generalize" might instead retrieve similar patterns. This confound makes it difficult to distinguish genuine construction from sophisticated retrieval.

We address this challenge with an **arbitrary rule following** paradigm. Rules are procedurally generated from random combinations of conditions (e.g., "if the prompt contains the letter 'k'") and actions (e.g., "respond in exactly 7 words"). These combinations are semantically arbitrary—they serve no communicative purpose and are unlikely to exist in training corpora. Success requires understanding the rule's structure and applying it correctly to novel inputs.

We test five frontier LLMs across five complexity levels, from single unconditional actions to self-referential rules requiring computed input properties to constrain output structure. Our findings reveal consistent patterns across models and architectures, contributing to understanding the boundaries of LLM compositional abilities.

---

## Results

### Experimental Design

We generated rules across five complexity levels:

| Level | Structure | Example |
|-------|-----------|---------|
| 1 | Single unconditional action | "Always begin your response with 'QUACK'" |
| 2 | Two unconditional actions | "Always respond in exactly 5 words AND end with '...maybe'" |
| 3 | IF-ELSE conditional | "If the prompt contains 'the', use ALL CAPS; otherwise, end with '- done'" |
| 4 | IF-ELIF-ELSE priority chain | Three conditions with priority ordering plus fallback |
| 5 | Self-referential | "Respond in exactly N words where N = number of vowels in the prompt" |

Rules were embedded in system prompts. Test prompts were neutral questions (e.g., "What color is the sky?") to isolate rule-following from content generation. Each model completed 300 trials (60 per level). See Methods for generation details.

### Per-Level Accuracy Across Models

| Model | L1 | L2 | L3 | L4 | L5 | Overall |
|-------|-----|-----|-----|-----|-----|---------|
| GPT-4o Mini | 85.0% | 61.7% | 73.3% | 58.3% | 18.3% | 59.3% |
| GPT-4o | 91.7% | 63.3% | 78.3% | 41.7% | 5.0% | 56.0% |
| GPT-4 Turbo | 80.0% | 50.0% | 71.7% | 40.0% | 21.7% | 52.7% |
| Claude Sonnet 4 | 93.3% | 41.7% | 41.7% | 35.0% | 8.3% | 43.3% |
| Claude 3.5 Haiku | 91.7% | 20.0% | 31.7% | 20.0% | 0.0% | 32.7% |

### Finding 1: Universal Degradation Pattern

All models show high accuracy on single-action rules (Level 1: 80-93%) and near-failure on self-referential rules (Level 5: 0-22%). This pattern holds across both providers and all model sizes tested.

Effect sizes (Cohen's h, Level 1 vs Level 5):
- Claude 3.5 Haiku: h = 2.56
- GPT-4o: h = 2.10
- Claude Sonnet 4: h = 2.03
- GPT-4o Mini: h = 1.46
- GPT-4 Turbo: h = 1.25

All effect sizes exceed h = 0.8 (conventionally "large"), indicating robust degradation across the complexity hierarchy.

### Finding 2: Binary Error Structure

When models fail, they predominantly ignore rules entirely rather than partially complying. We classified failures as:
- **Binary failure**: No rule components satisfied
- **Partial failure**: Some but not all components satisfied

| Model | Binary Failure Rate |
|-------|---------------------|
| Claude 3.5 Haiku | 91.1% |
| Claude Sonnet 4 | 87.5% |
| GPT-4o | 84.8% |
| GPT-4o Mini | 84.4% |
| GPT-4 Turbo | 81.7% |

All models show >80% binary failure rate. This pattern is consistent with retrieval-based processing: when no matching pattern exists, the system defaults to standard response generation rather than attempting partial rule satisfaction.

### Finding 3: Coordination Difficulty Exceeds Conditional Difficulty

We compared Level 2 (two simultaneous unconditional actions) with Level 3 (conditional with single action per branch):

| Model | L2 (Coordination) | L3 (Conditional) | Difference |
|-------|-------------------|------------------|------------|
| GPT-4 Turbo | 50.0% | 71.7% | -21.7% |
| GPT-4o | 63.3% | 78.3% | -15.0% |
| GPT-4o Mini | 61.7% | 73.3% | -11.6% |
| Claude 3.5 Haiku | 20.0% | 31.7% | -11.7% |
| Claude Sonnet 4 | 41.7% | 41.7% | 0.0% |

Four of five models find coordination (L2) harder than conditionals (L3). This suggests that maintaining multiple simultaneous constraints poses greater difficulty than conditional branching with single outputs—a finding that merits further investigation.

### Finding 4: Scale Does Not Predict Performance

Within the OpenAI model family:
- GPT-4o Mini (smallest): 59.3% overall
- GPT-4o: 56.0% overall
- GPT-4 Turbo: 52.7% overall

The smallest model outperforms larger models on this task. This is inconsistent with the hypothesis that scale alone improves compositional generalization, and consistent with prior findings that emergent abilities may reflect evaluation metrics rather than genuine capability improvements (Schaeffer et al., 2023).

### Statistical Reliability

95% confidence intervals (Wilson score method) for Claude Sonnet 4:

| Level | Accuracy | 95% CI |
|-------|----------|--------|
| 1 | 93.3% | [84.1%, 97.4%] |
| 2 | 41.7% | [30.1%, 54.3%] |
| 3 | 41.7% | [30.1%, 54.3%] |
| 4 | 35.0% | [24.2%, 47.6%] |
| 5 | 8.3% | [3.6%, 18.1%] |

The Level 1 vs Level 5 difference (85 percentage points) far exceeds the confidence interval widths, indicating robust differentiation.

---

## Discussion

### Summary of Findings

We introduced an arbitrary rule following paradigm to test compositional generalization in LLMs with minimized training contamination. Across five models:

1. **Universal degradation**: All models show high performance on simple rules and near-failure on self-referential rules
2. **Binary failures**: Errors are predominantly all-or-nothing (>80% across models)
3. **Coordination difficulty**: Most models find simultaneous constraints harder than conditional branching
4. **Scale-independence**: Larger models do not outperform smaller ones

### Relation to Prior Work

Our findings are consistent with Lake & Baroni's (2018) demonstration that sequence-to-sequence models fail at compositional generalization despite succeeding on in-distribution tests. The arbitrary rule paradigm extends this work by minimizing the possibility that apparent generalization reflects training distribution overlap.

The binary error structure we observe aligns with Dziri et al.'s (2023) analysis of transformer failures on compositional tasks, which found that models often fail to engage with task structure rather than making systematic errors within it.

The finding that scale does not improve performance is consistent with Schaeffer et al.'s (2023) argument that apparent emergent abilities may reflect metric properties rather than genuine capability transitions. Our task—with guaranteed low training coverage—may provide a cleaner test of this hypothesis.

### Limitations

**Training coverage claims**: While arbitrary rule combinations are unlikely in training data, component patterns (e.g., "always begin with X") certainly exist. We claim *reduced* training coverage, not zero. Future work could quantify this via training data analysis where available.

**Sample sizes**: With 60 trials per level per model, confidence intervals remain moderately wide. Some action types had fewer than 15 trials. Higher-powered replications would strengthen conclusions.

**Level design**: Our Level 3 uses single actions per conditional branch. Different designs (e.g., multiple actions per branch) might yield different patterns. The L2/L3 comparison should be interpreted cautiously given this design choice.

**Evaluation**: Programmatic evaluation may miss edge cases. Human evaluation on a subset would strengthen validity.

**Model selection**: We tested five models from two providers. Broader testing across open-source models and different architectures would improve generalizability.

**Mechanistic understanding**: We describe patterns but do not explain mechanisms. Why do models fail at coordination? Attention analysis, probing, or ablation studies could provide insight.

### Working Hypotheses

The following interpretations are speculative and require further investigation:

*Why binary failures?* One possibility is that rule-following requires explicit retrieval of a matching pattern. When no pattern exists, the system defaults to standard generation rather than constructing behavior from rule components. This would be consistent with LLMs operating as sophisticated pattern matchers rather than general-purpose reasoners.

*Why is coordination hard?* Satisfying multiple simultaneous constraints may require maintaining and integrating multiple pieces of information during generation. If models generate token-by-token with limited lookahead, coordinating end-state constraints (e.g., word count AND specific ending) may be architecturally difficult.

*Why doesn't scale help?* If the bottleneck is architectural (e.g., autoregressive generation limiting planning capacity) rather than knowledge-based, scaling may improve pattern coverage without improving compositional construction.

These hypotheses are consistent with our data but not uniquely predicted by it. Alternative explanations may exist.

---

## Methods

### Rule Generation

Rules were generated procedurally (random seed = 42) from:

**Condition types (9):** letter_contains, letter_starts, word_count_parity, char_count_threshold, vowel_count_threshold, word_position_letter, contains_word, char_position, letter_ends

**Action types (8):** prepend, append, all_caps, all_lower, word_limit, include_count, include_symbol, reverse_first_word

Parameters (letters, words, thresholds) were sampled uniformly from predefined pools.

### Complexity Levels

| Level | Logic | Actions per branch |
|-------|-------|-------------------|
| 1 | Unconditional | 1 |
| 2 | Unconditional | 2 (simultaneous) |
| 3 | IF-ELSE | 1 per branch |
| 4 | IF-ELIF-ELIF-ELSE | 1 per branch |
| 5 | Self-referential | 1 (output depends on computed input property) |

### Evaluation Criteria

Compliance was evaluated programmatically:
- **Text insertion** (prepend, append): Case-insensitive substring match
- **Case transformation**: All alphabetic characters checked
- **Word count**: Whitespace-split token count
- **Symbol count**: Exact count match
- **Self-referential**: Output property matches computed input property

### Models Tested

- Claude Sonnet 4 (Anthropic, December 2024)
- Claude 3.5 Haiku (Anthropic)
- GPT-4o (OpenAI)
- GPT-4o Mini (OpenAI)
- GPT-4 Turbo (OpenAI)

All models accessed via official APIs with default parameters (temperature not modified from API defaults).

### Statistical Analysis

- **Confidence intervals**: Wilson score method
- **Effect sizes**: Cohen's h for proportion comparisons
- **Significance**: Not reported; we focus on effect sizes and confidence intervals per modern statistical recommendations (Cumming, 2014)

---

## Conclusion

The arbitrary rule following paradigm provides a simple, reproducible test of compositional generalization with minimized training contamination. Our core finding—that all tested models show large degradation from simple to complex rules, with binary error structure and no benefit from scale—is robust across five models from two providers.

These results are consistent with the hypothesis that current LLMs perform sophisticated pattern matching rather than genuine compositional construction, though they do not definitively establish this. The paradigm is extensible and we encourage replication with additional models, rule spaces, and sample sizes.

**Data and code availability**: https://github.com/HillaryDanan/arbitrary-rules

---

## References

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems, 33*, 1877-1901.

Cumming, G. (2014). The new statistics: Why and how. *Psychological Science, 25*(1), 7-29.

Dziri, N., Lu, X., Sclar, M., Li, X. L., Jian, L., Lin, B. Y., ... & Choi, Y. (2023). Faith and fate: Limits of transformers on compositionality. *Advances in Neural Information Processing Systems, 36*.

Fodor, J. A., & Pylyshyn, Z. W. (1988). Connectionism and cognitive architecture: A critical analysis. *Cognition, 28*(1-2), 3-71.

Lake, B., & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. *International Conference on Machine Learning*, 2873-2882.

Press, O., Zhang, M., Min, S., Min, S., Schmidt, L., Smith, N., & Lewis, M. (2023). Measuring and narrowing the compositionality gap in language models. *Findings of EMNLP 2023*.

Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? *Advances in Neural Information Processing Systems, 36*.

Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., ... & Fedus, W. (2022). Emergent abilities of large language models. *Transactions on Machine Learning Research*.

---

## Supplementary Materials

### Table S1: Action Type Success Rates (Claude Sonnet 4)

| Action | N | Success Rate | 95% CI |
|--------|---|--------------|--------|
| word_limit | 31 | 25.8% | [16.7%, 37.4%] |
| include_count | 15 | 33.3% | [17.2%, 54.6%] |
| all_caps | 17 | 64.7% | [41.3%, 82.7%] |
| prepend | 25 | 80.0% | [58.4%, 91.9%] |
| include_symbol | 38 | 81.6% | [66.6%, 90.8%] |
| append | 41 | 82.9% | [67.3%, 91.9%] |
| all_lower | 12 | 91.7% | [78.2%, 97.1%] |

*Note: Some action types have small N; interpret with caution.*

### Table S2: Binary Failure Rate by Level (Claude Sonnet 4)

| Level | Total Failures | Binary Failures | Partial Failures | Binary Rate |
|-------|----------------|-----------------|------------------|-------------|
| 1 | 4 | 4 | 0 | 100% |
| 2 | 35 | 14 | 21 | 40% |
| 3 | 35 | 35 | 0 | 100% |
| 4 | 39 | 39 | 0 | 100% |
| 5 | 55 | 55 | 0 | 100% |

*Note: Level 2 shows partial failures because two independent actions can be satisfied independently. Levels 3-5 have structurally linked actions.*

### Table S3: Test-Retest Reliability (Claude Sonnet 4)

Two independent runs showed high consistency:

| Level | Run 1 | Run 2 | Difference |
|-------|-------|-------|------------|
| 1 | 93.3% | 93.3% | 0.0% |
| 2 | 41.7% | 45.0% | 3.3% |
| 3 | 41.7% | 40.0% | 1.7% |
| 4 | 35.0% | 31.7% | 3.3% |
| 5 | 8.3% | 6.7% | 1.6% |

Mean absolute difference: 2.0 percentage points.
