# Arbitrary Rule Following Experiment

Testing compositional construction vs. pattern matching in large language models using procedurally-generated rules with guaranteed zero training coverage.

## Overview

This experiment tests predictions from the Abstraction Primitive Hypothesis (APH) by presenting LLMs with arbitrary rules they cannot have seen during training. Rules are constructed from random combinations of conditions and actions, isolating the ability to *construct* novel behaviors from the ability to *retrieve* trained patterns.

## Key Findings

**Tested models:** Claude Sonnet 4, Claude 3.5 Haiku, GPT-4o, GPT-4o Mini, GPT-4 Turbo

### Per-Level Accuracy

| Model | L1 | L2 | L3 | L4 | L5 | Overall |
|-------|-----|-----|-----|-----|-----|---------|
| GPT-4o Mini | 85% | 62% | 73% | 58% | 18% | 59% |
| GPT-4o | 92% | 63% | 78% | 42% | 5% | 56% |
| GPT-4 Turbo | 80% | 50% | 72% | 40% | 22% | 53% |
| Claude Sonnet 4 | 93% | 42% | 42% | 35% | 8% | 43% |
| Claude 3.5 Haiku | 92% | 20% | 32% | 20% | 0% | 33% |

### Main Findings

**Confirmed across all models:**
- L1 (single action): High success (80-93%)
- L5 (self-referential): Near-failure (0-22%)
- Binary error structure (81-91%): Rules ignored entirely, not partially followed
- Large effect sizes (Cohen's h = 1.25-2.56)
- Scale ≠ performance: GPT-4o Mini outperforms larger models

**Model-specific:**
- L2 vs L3 pattern varies: Most models find coordination (L2) harder than conditionals (L3), but Claude Sonnet 4 finds them equally difficult

## Complexity Levels

| Level | Structure | Example |
|-------|-----------|---------|
| 1 | Single unconditional action | "Always begin with 'QUACK'" |
| 2 | Two unconditional actions | "Always use ALL CAPS and end with '!!'" |
| 3 | IF-ELSE conditional | "If prompt contains 'the', prepend 'YES'; otherwise append 'NO'" |
| 4 | Priority chain (IF-ELIF-ELSE) | Three conditions with priority ordering |
| 5 | Self-referential | "Respond in exactly N words where N = vowels in prompt" |

## Installation

```bash
pip install -r requirements.txt
```

Required API keys (set as environment variables):
- `ANTHROPIC_API_KEY` - For Claude models
- `OPENAI_API_KEY` - For GPT models
- `GOOGLE_API_KEY` - For Gemini models (optional)

## Usage

### Run experiment

```bash
# Single model (default: Claude Sonnet 4)
python run.py

# Specific model
python run.py --model gpt-4o

# High statistical power (60 trials per level)
python run.py --high-power

# Compare frontier models
python run.py --compare --high-power

# List available models
python run.py --list-models
```

### Analyze results

```bash
# Single model analysis
python analyze_aph.py results/results_claude_sonnet_4_*.json

# Multi-model comparison
python analyze_multimodel.py results/
```

## File Structure

```
├── run.py                 # Main entry point
├── experiment.py          # Experiment logic
├── rule_generator.py      # Procedural rule generation
├── provider.py            # Multi-provider API abstraction
├── config.py              # Configuration and parameters
├── analyze_aph.py         # Single-model APH analysis
├── analyze_multimodel.py  # Cross-model comparison
├── results/               # Experiment outputs (JSON)
└── paper_draft.md         # Manuscript draft
```

## Configuration

Key parameters in `config.py`:

- **Statistical:** Minimum 30 trials per level (adjustable)
- **Rule generation:** 9 condition types × 8 action types
- **Reproducibility:** Random seed = 42

## Methodology

1. **Procedural generation:** Rules are constructed from random combinations, ensuring no training coverage
2. **Neutral prompts:** Test prompts are simple factual questions to isolate rule-following
3. **Programmatic evaluation:** Compliance checked automatically (string matching, counting)
4. **Statistical rigor:** Wilson confidence intervals, Cohen's h effect sizes

## Citation

```bibtex
@article{danan2025arbitrary,
  title={Arbitrary Rule Following Reveals Compositional Limits in Large Language Models},
  author={Danan, Hillary},
  year={2025}
}
```

## Related Work

- [Abstraction Primitive Hypothesis](https://github.com/HillaryDanan/abstraction-intelligence)
- Lake & Baroni (2018). Generalization without systematicity. *ICML*.
- Dziri et al. (2023). Faith and Fate: Limits of Transformers on Compositionality. *NeurIPS*.

## License

MIT
