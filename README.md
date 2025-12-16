# Arbitrary Rule Following Experiment

Testing compositional generalization in large language models using procedurally-generated rules.

## Overview

This experiment tests whether LLMs can follow arbitrary rules constructed from random combinations of conditions and actions. Rules are procedurally generated to minimize overlap with training distributions.

## Key Finding

**One robust result:** Input-dependent rules (L5) are consistently hard (5-15% accuracy). This replicates across models and runs.

**Caution:** Other findings show high run-to-run variance (±20-40pp). Specific accuracy values at L1-L4 are unstable.

## Results Summary

### Initial Run (N=300 per model)

| Model | L1 | L2 | L3 | L4 | L5 |
|-------|-----|-----|-----|-----|-----|
| GPT-4o Mini | 85% | 62% | 73% | 58% | 18% |
| GPT-4o | 92% | 63% | 78% | 42% | 5% |
| Claude Sonnet 4 | 93% | 42% | 42% | 35% | 8% |

### Ultra-Power Replication (N=600 per model)

| Model | L1 | L2 | L3 | L4 | L5 |
|-------|-----|-----|-----|-----|-----|
| Claude Sonnet 4 | **74%** | 33% | 41% | 40% | 10% |
| GPT-4o | 83% | 63% | **38%** | 30% | 7% |

**Notable:** Claude L1 dropped 19pp; GPT-4o L3 dropped 40pp between runs.

## Complexity Levels

| Level | Structure | Example |
|-------|-----------|---------|
| 1 | Single action | "Always begin with 'QUACK'" |
| 2 | Two actions | "Use ALL CAPS and end with '!!'" |
| 3 | IF-ELSE | "If prompt contains 'the', prepend 'YES'; else append 'NO'" |
| 4 | IF-ELIF-ELSE | Three conditions with priority |
| 5 | Input-dependent | "Respond in N words where N = vowels in prompt" |

## What Replicates vs What Doesn't

| Finding | Status |
|---------|--------|
| L5 is hard (5-15%) | ✅ Robust |
| L1 > L5 (large effect) | ✅ Robust |
| Specific L1-L4 values | ❌ Unstable (±20-40pp) |
| L2 vs L3 pattern | ❌ Reversed between runs |

## Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
```

## Usage

```bash
python run.py --model gpt-4o          # Single model
python run.py --high-power            # 60 trials/level
python run.py --ultra-power           # 120 trials/level
python run.py --compare --ultra-power # Multi-model comparison
```

## Known Limitations

- **High variance**: 20-40pp run-to-run differences at L1-L4
- **L2 vs L3 unstable**: Pattern reversed for GPT-4o between runs
- **Rule sampling**: Different rules may have different difficulty

## Recommendation

Focus on the **L5 finding** (input-dependent constraints are hard). Treat L1-L4 comparisons as exploratory.

## Citation

```bibtex
@article{danan2025arbitrary,
  title={Arbitrary Rule Following Reveals Compositional Limits in Large Language Models},
  author={Danan, Hillary},
  year={2025}
}
```

## License

MIT
