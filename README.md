# Arbitrary Rule Following Experiment

Testing compositional generalization in large language models using procedurally-generated rules with minimized training coverage.

## Overview

This experiment tests whether LLMs can follow arbitrary rules constructed from random combinations of conditions and actions. Rules are procedurally generated to minimize overlap with training distributions, isolating the ability to *construct* novel behaviors from the ability to *retrieve* trained patterns.

## Key Findings

Across 5 models (GPT-4o, GPT-4o Mini, GPT-4 Turbo, Claude Sonnet 4, Claude 3.5 Haiku), N=300 trials each:

| Model | L1 | L2 | L3 | L4 | L5 | Overall |
|-------|-----|-----|-----|-----|-----|---------|
| GPT-4o Mini | 85% | 62% | 73% | 58% | 18% | 59% |
| GPT-4o | 92% | 63% | 78% | 42% | 5% | 56% |
| GPT-4 Turbo | 80% | 50% | 72% | 40% | 22% | 53% |
| Claude Sonnet 4 | 93% | 42% | 42% | 35% | 8% | 43% |
| Claude 3.5 Haiku | 92% | 20% | 32% | 20% | 0% | 33% |

**Findings:**
1. All models: L1 high (80-93%), L5 near-failure (0-22%)
2. Binary error structure (81-91%): rules ignored entirely, not partially followed
3. Coordination (L2) harder than conditionals (L3) for most models
4. Scale ≠ performance: GPT-4o Mini outperforms larger models

## Complexity Levels

| Level | Structure | Example |
|-------|-----------|---------|
| 1 | Single action | "Always begin with 'QUACK'" |
| 2 | Two simultaneous actions | "Use ALL CAPS and end with '!!'" |
| 3 | IF-ELSE conditional | "If prompt contains 'the', prepend 'YES'; otherwise append 'NO'" |
| 4 | IF-ELIF-ELSE chain | Three conditions with priority ordering |
| 5 | Self-referential | "Respond in N words where N = vowels in prompt" |

## Installation

```bash
pip install -r requirements.txt
```

Set API keys:
```bash
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
```

## Usage

```bash
# Default (30 trials/level)
python run.py

# Specific model
python run.py --model gpt-4o

# High power (60 trials/level) - recommended
python run.py --high-power

# Ultra power (120 trials/level) - for publication
python run.py --ultra-power

# Compare multiple models
python run.py --compare --high-power

# List available models
python run.py --list-models
```

## Analysis

```bash
# Single model
python analyze_aph.py results/results_*.json

# Multi-model comparison
python analyze_multimodel.py results/
```

## Statistical Power

| Trials/Level | CI Width (at 50%) | Use Case |
|--------------|-------------------|----------|
| 30 | ±18% | Quick testing |
| 60 | ±13% | Standard |
| 120 | ±9% | Publication |
| 200 | ±7% | High confidence |

## Files

```
├── run.py                 # Main entry point
├── experiment.py          # Experiment logic
├── rule_generator.py      # Procedural rule generation
├── provider.py            # Multi-provider API abstraction
├── config.py              # Configuration
├── analyze_aph.py         # Single-model analysis
├── analyze_multimodel.py  # Cross-model comparison
├── paper_draft.md         # Manuscript
└── results/               # Output JSON files
```

## Citation

```bibtex
@article{danan2025arbitrary,
  title={Arbitrary Rule Following Reveals Compositional Limits in Large Language Models},
  author={Danan, Hillary},
  year={2025}
}
```

## References

- Lake & Baroni (2018). Generalization without systematicity. *ICML*.
- Dziri et al. (2023). Faith and Fate: Limits of Transformers on Compositionality. *NeurIPS*.
- Schaeffer et al. (2023). Are emergent abilities a mirage? *NeurIPS*.

## License

MIT
