#!/usr/bin/env python3
"""
L5 Decomposition Experiment
===========================
Testing WHERE input-dependent constraints fail.

L5 requires: Compute → Bind → Constrain
This experiment isolates each component.

Conditions:
  A. COUNT_ONLY      - Can the model count vowels?
  B. FIXED_CONSTRAINT - Can the model constrain output length (fixed N)?
  C. COUNT_AND_REPORT - Can the model count and report while responding?
  D. FULL_L5         - Can the model use computed value as constraint?
  E. SCRATCHPAD      - Does explicit intermediate step help?

Design: Within-subjects (same prompts across all conditions)

Hypotheses:
  H1: Counting is hard         → A fails
  H2: Constraint is hard       → B fails  
  H3: Binding is hard          → A,B,C pass; D fails
  H4: Scratchpad helps         → D fails, E passes (working memory)
  H5: Scratchpad doesn't help  → D,E both fail (architectural)

Usage:
  python l5_decomposition.py                    # Default model
  python l5_decomposition.py --model gpt-4o    # Specific model
  python l5_decomposition.py --trials 30       # More trials
  python l5_decomposition.py --all-models      # Compare models
"""

import argparse
import json
import random
import re
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import provider from existing codebase
try:
    from provider import create_provider, BaseProvider
    from config import AVAILABLE_MODELS
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure you're in the arbitrary_rules directory and have installed requirements:")
    print("  cd arbitrary_rules")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Default model
DEFAULT_MODEL = "claude-sonnet-4"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DecompositionConfig:
    """Experiment configuration."""
    random_seed: int = 42
    trials_per_condition: int = 20
    
    # Test prompts - designed to have varying vowel counts
    test_prompts: List[str] = None
    
    # Fixed constraint value for condition B
    fixed_word_count: int = 7
    
    def __post_init__(self):
        if self.test_prompts is None:
            # Prompts with known, varying vowel counts
            # Each prompt designed to be answerable in variable lengths
            self.test_prompts = [
                "What color is the sky?",           # 6 vowels
                "Name a fruit.",                     # 4 vowels
                "What do cats eat?",                 # 5 vowels
                "Is water wet?",                     # 4 vowels
                "How are you today?",                # 7 vowels
                "What time is it?",                  # 5 vowels
                "Where do birds fly?",               # 5 vowels
                "Can dogs swim?",                    # 3 vowels
                "Why is grass green?",               # 5 vowels
                "Do fish sleep?",                    # 3 vowels
                "What makes ice cold?",              # 5 vowels
                "How do planes fly?",                # 5 vowels
                "What is your name?",                # 5 vowels
                "Tell me a fact.",                   # 4 vowels
                "When does snow fall?",              # 5 vowels
                "What do bees make?",                # 5 vowels
                "Are clouds soft?",                  # 4 vowels
                "How big is the moon?",              # 6 vowels
                "Do trees grow?",                    # 4 vowels
                "What season is it?",                # 6 vowels
                "Can you count?",                    # 4 vowels
                "What do cows produce?",             # 7 vowels
                "Is the sun hot?",                   # 4 vowels
                "Where is north?",                   # 4 vowels
                "How many legs do spiders have?",    # 9 vowels
                "What color are bananas?",           # 8 vowels
                "Do penguins fly?",                  # 5 vowels
                "What is two plus two?",             # 7 vowels
                "Can robots think?",                 # 4 vowels
                "What floats on water?",             # 7 vowels
            ]


# =============================================================================
# CONDITIONS
# =============================================================================

class Condition:
    """Base class for experimental conditions."""
    name: str
    description: str
    
    @staticmethod
    def get_system_prompt(config: DecompositionConfig) -> str:
        raise NotImplementedError
    
    @staticmethod
    def evaluate(prompt: str, response: str, config: DecompositionConfig) -> Dict[str, Any]:
        raise NotImplementedError


class ConditionA(Condition):
    """COUNT_ONLY: Can the model count vowels?"""
    name = "A_COUNT_ONLY"
    description = "Count vowels in prompt and report the number"
    
    @staticmethod
    def get_system_prompt(config: DecompositionConfig) -> str:
        return """Count the number of vowels (a, e, i, o, u) in the user's message.
Respond with ONLY the number, nothing else.
Example: If the message is "Hello world", respond with: 3"""
    
    @staticmethod
    def evaluate(prompt: str, response: str, config: DecompositionConfig) -> Dict[str, Any]:
        expected = count_vowels(prompt)
        
        # Extract number from response
        numbers = re.findall(r'\d+', response.strip())
        if numbers:
            reported = int(numbers[0])
        else:
            # Try to parse written numbers
            reported = parse_written_number(response.strip())
        
        correct = (reported == expected) if reported is not None else False
        
        return {
            "expected_count": expected,
            "reported_count": reported,
            "correct": correct,
            "response_clean": response.strip()[:100]
        }


class ConditionB(Condition):
    """FIXED_CONSTRAINT: Can the model constrain output length (fixed N)?"""
    name = "B_FIXED_CONSTRAINT"
    description = "Respond in exactly N words (fixed N)"
    
    @staticmethod
    def get_system_prompt(config: DecompositionConfig) -> str:
        n = config.fixed_word_count
        return f"""You must respond in EXACTLY {n} words. Not {n-1}, not {n+1}, exactly {n} words.
Count carefully before responding."""
    
    @staticmethod
    def evaluate(prompt: str, response: str, config: DecompositionConfig) -> Dict[str, Any]:
        expected = config.fixed_word_count
        actual = count_words(response)
        correct = (actual == expected)
        
        return {
            "expected_words": expected,
            "actual_words": actual,
            "correct": correct,
            "off_by": actual - expected,
            "response_clean": response.strip()[:100]
        }


class ConditionC(Condition):
    """COUNT_AND_REPORT: Can the model count and report while responding?"""
    name = "C_COUNT_AND_REPORT"
    description = "Count vowels, report count, then respond naturally"
    
    @staticmethod
    def get_system_prompt(config: DecompositionConfig) -> str:
        return """First, count the vowels (a, e, i, o, u) in the user's message.
Start your response with "VOWELS: N" where N is the count.
Then respond naturally to their question.

Example:
User: "What is red?"
Response: "VOWELS: 3
Red is a primary color often associated with passion and energy." """
    
    @staticmethod
    def evaluate(prompt: str, response: str, config: DecompositionConfig) -> Dict[str, Any]:
        expected = count_vowels(prompt)
        
        # Look for "VOWELS: N" pattern
        match = re.search(r'VOWELS?\s*[:\-]?\s*(\d+)', response, re.IGNORECASE)
        if match:
            reported = int(match.group(1))
        else:
            reported = None
        
        correct = (reported == expected) if reported is not None else False
        
        return {
            "expected_count": expected,
            "reported_count": reported,
            "correct": correct,
            "format_followed": match is not None,
            "response_clean": response.strip()[:100]
        }


class ConditionD(Condition):
    """FULL_L5: Can the model use computed value as constraint?"""
    name = "D_FULL_L5"
    description = "Respond in exactly N words where N = vowels in prompt"
    
    @staticmethod
    def get_system_prompt(config: DecompositionConfig) -> str:
        return """Count the vowels (a, e, i, o, u) in the user's message.
Then respond in EXACTLY that many words.

Example: If the message has 5 vowels, respond in exactly 5 words.

This is critical: your response must have EXACTLY as many words as there are vowels in the prompt."""
    
    @staticmethod
    def evaluate(prompt: str, response: str, config: DecompositionConfig) -> Dict[str, Any]:
        expected_vowels = count_vowels(prompt)
        actual_words = count_words(response)
        correct = (actual_words == expected_vowels)
        
        return {
            "vowels_in_prompt": expected_vowels,
            "words_in_response": actual_words,
            "correct": correct,
            "off_by": actual_words - expected_vowels,
            "response_clean": response.strip()[:100]
        }


class ConditionE(Condition):
    """SCRATCHPAD: Does explicit intermediate step help?"""
    name = "E_SCRATCHPAD"
    description = "Explicitly count first, then constrain (chain of thought)"
    
    @staticmethod
    def get_system_prompt(config: DecompositionConfig) -> str:
        return """Follow these steps EXACTLY:

STEP 1: Count the vowels (a, e, i, o, u) in the user's message.
        Write: "Counting vowels: [list each vowel you find]"
        Write: "Total vowels: N"

STEP 2: Now respond to their question in EXACTLY N words (the number from Step 1).
        Write: "Response (N words): [your N-word response]"

You MUST show your counting work before responding."""
    
    @staticmethod
    def evaluate(prompt: str, response: str, config: DecompositionConfig) -> Dict[str, Any]:
        expected_vowels = count_vowels(prompt)
        
        # Check if they reported a count
        count_match = re.search(r'[Tt]otal\s*[Vv]owels?\s*[:\-]?\s*(\d+)', response)
        reported_count = int(count_match.group(1)) if count_match else None
        
        # Check if counting was correct
        count_correct = (reported_count == expected_vowels) if reported_count else False
        
        # Find the response portion and count words
        response_match = re.search(r'[Rr]esponse.*?[:\-]\s*(.+?)(?:\n|$)', response, re.DOTALL)
        if response_match:
            response_text = response_match.group(1).strip()
            # Clean up any trailing formatting
            response_text = re.sub(r'\s*\([^)]*\)\s*$', '', response_text)
        else:
            # Fall back to last line or full response
            lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
            response_text = lines[-1] if lines else response
        
        actual_words = count_words(response_text)
        
        # For scratchpad, success = used correct count as constraint
        # (they might have miscounted but correctly used their count)
        if reported_count is not None:
            used_own_count = (actual_words == reported_count)
        else:
            used_own_count = False
        
        # Full success = correct count AND correct word count
        full_correct = (actual_words == expected_vowels)
        
        return {
            "vowels_in_prompt": expected_vowels,
            "reported_count": reported_count,
            "count_correct": count_correct,
            "words_in_response": actual_words,
            "used_own_count": used_own_count,
            "full_correct": full_correct,
            "response_clean": response.strip()[:200]
        }


# All conditions
CONDITIONS = [ConditionA, ConditionB, ConditionC, ConditionD, ConditionE]


# =============================================================================
# UTILITIES
# =============================================================================

def count_vowels(text: str) -> int:
    """Count vowels (a, e, i, o, u) in text, case-insensitive."""
    return sum(1 for c in text.lower() if c in 'aeiou')


def count_words(text: str) -> int:
    """Count words in text (whitespace-split, basic cleaning)."""
    # Remove common non-word artifacts
    text = re.sub(r'^\s*VOWELS?\s*[:\-]?\s*\d+\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*[Rr]esponse.*?[:\-]\s*', '', text)
    
    # Split on whitespace
    words = text.split()
    
    # Filter out empty strings and pure punctuation
    words = [w for w in words if w and re.search(r'[a-zA-Z0-9]', w)]
    
    return len(words)


def parse_written_number(text: str) -> Optional[int]:
    """Parse simple written numbers."""
    text = text.lower().strip()
    
    number_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15
    }
    
    for word, num in number_map.items():
        if word in text:
            return num
    
    return None


# =============================================================================
# EXPERIMENT
# =============================================================================

@dataclass
class Trial:
    """Single trial result."""
    condition: str
    prompt: str
    response: str
    evaluation: Dict[str, Any]
    correct: bool
    latency_ms: float
    error: Optional[str] = None


class L5DecompositionExperiment:
    """Main experiment class."""
    
    def __init__(self, provider: BaseProvider, config: DecompositionConfig):
        self.provider = provider
        self.config = config
        random.seed(config.random_seed)
    
    def run_trial(self, condition_cls, prompt: str) -> Trial:
        """Run a single trial."""
        system_prompt = condition_cls.get_system_prompt(self.config)
        
        start = time.time()
        try:
            response = self.provider.call(
                system_prompt=system_prompt,
                user_prompt=prompt
            )
            latency = (time.time() - start) * 1000
            
            if response.success:
                evaluation = condition_cls.evaluate(prompt, response.content, self.config)
                correct = evaluation.get('correct', evaluation.get('full_correct', False))
                
                return Trial(
                    condition=condition_cls.name,
                    prompt=prompt,
                    response=response.content,
                    evaluation=evaluation,
                    correct=correct,
                    latency_ms=latency
                )
            else:
                return Trial(
                    condition=condition_cls.name,
                    prompt=prompt,
                    response="",
                    evaluation={},
                    correct=False,
                    latency_ms=latency,
                    error=response.error
                )
        except Exception as e:
            return Trial(
                condition=condition_cls.name,
                prompt=prompt,
                response="",
                evaluation={},
                correct=False,
                latency_ms=(time.time() - start) * 1000,
                error=str(e)
            )
    
    def run(self, parallel: bool = True, max_workers: int = 5) -> Dict[str, Any]:
        """Run full experiment."""
        
        print("=" * 70)
        print("L5 DECOMPOSITION EXPERIMENT")
        print("=" * 70)
        print(f"Model: {self.provider.model_config.name}")
        print(f"Trials per condition: {self.config.trials_per_condition}")
        print(f"Conditions: {len(CONDITIONS)}")
        print(f"Total trials: {self.config.trials_per_condition * len(CONDITIONS)}")
        print("=" * 70)
        
        # Select prompts for this run
        prompts = self.config.test_prompts[:self.config.trials_per_condition]
        if len(prompts) < self.config.trials_per_condition:
            # Repeat if needed
            prompts = (prompts * (self.config.trials_per_condition // len(prompts) + 1))
            prompts = prompts[:self.config.trials_per_condition]
        
        # Build all tasks
        all_tasks = []
        for condition_cls in CONDITIONS:
            for prompt in prompts:
                all_tasks.append((condition_cls, prompt))
        
        random.shuffle(all_tasks)  # Randomize order
        
        print(f"\n🚀 Running {len(all_tasks)} trials...")
        
        all_trials = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.run_trial, cond, prompt): (cond, prompt)
                    for cond, prompt in all_tasks
                }
                
                completed = 0
                for future in as_completed(futures):
                    trial = future.result()
                    all_trials.append(trial)
                    completed += 1
                    
                    if completed % 10 == 0:
                        print(f"  Progress: {completed}/{len(all_tasks)}")
        else:
            for i, (cond, prompt) in enumerate(all_tasks):
                trial = self.run_trial(cond, prompt)
                all_trials.append(trial)
                
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(all_tasks)}")
        
        # Compute results
        results = self._compute_results(all_trials)
        results['model'] = self.provider.model_config.name
        results['config'] = asdict(self.config)
        results['config']['test_prompts'] = prompts  # Only used prompts
        results['trials'] = [asdict(t) for t in all_trials]
        
        return results
    
    def _compute_results(self, trials: List[Trial]) -> Dict[str, Any]:
        """Compute summary statistics."""
        
        by_condition = {}
        for cond_cls in CONDITIONS:
            cond_trials = [t for t in trials if t.condition == cond_cls.name]
            n_correct = sum(1 for t in cond_trials if t.correct)
            n_total = len(cond_trials)
            
            by_condition[cond_cls.name] = {
                'description': cond_cls.description,
                'n_trials': n_total,
                'n_correct': n_correct,
                'accuracy': n_correct / n_total if n_total > 0 else 0,
                'errors': sum(1 for t in cond_trials if t.error)
            }
            
            # Condition-specific metrics
            if cond_cls.name == 'E_SCRATCHPAD':
                # For scratchpad, also track count accuracy and binding
                count_correct = sum(1 for t in cond_trials 
                                   if t.evaluation.get('count_correct', False))
                used_own = sum(1 for t in cond_trials 
                              if t.evaluation.get('used_own_count', False))
                
                by_condition[cond_cls.name]['count_accuracy'] = count_correct / n_total if n_total > 0 else 0
                by_condition[cond_cls.name]['used_own_count_rate'] = used_own / n_total if n_total > 0 else 0
        
        return {
            'by_condition': by_condition,
            'timestamp': datetime.now().isoformat()
        }


# =============================================================================
# OUTPUT
# =============================================================================

def print_results(results: Dict[str, Any]):
    """Print formatted results."""
    
    print("\n" + "=" * 70)
    print("RESULTS: L5 DECOMPOSITION")
    print("=" * 70)
    print(f"Model: {results['model']}")
    
    by_cond = results['by_condition']
    
    # Main results table
    print("\n" + "-" * 70)
    print(f"{'Condition':<20} {'Description':<35} {'Accuracy':<15}")
    print("-" * 70)
    
    for cond_name in ['A_COUNT_ONLY', 'B_FIXED_CONSTRAINT', 'C_COUNT_AND_REPORT', 
                      'D_FULL_L5', 'E_SCRATCHPAD']:
        data = by_cond[cond_name]
        acc = data['accuracy']
        acc_str = f"{acc:.1%} ({data['n_correct']}/{data['n_trials']})"
        print(f"{cond_name:<20} {data['description'][:35]:<35} {acc_str:<15}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    a = by_cond['A_COUNT_ONLY']['accuracy']
    b = by_cond['B_FIXED_CONSTRAINT']['accuracy']
    c = by_cond['C_COUNT_AND_REPORT']['accuracy']
    d = by_cond['D_FULL_L5']['accuracy']
    e = by_cond['E_SCRATCHPAD']['accuracy']
    
    print(f"\nComponent Accuracies:")
    print(f"  Counting (A):              {a:.1%}")
    print(f"  Fixed Constraint (B):      {b:.1%}")
    print(f"  Count + Report (C):        {c:.1%}")
    print(f"  Full L5 (D):               {d:.1%}")
    print(f"  Scratchpad (E):            {e:.1%}")
    
    # Hypothesis testing
    print(f"\nHypothesis Testing:")
    
    PASS_THRESHOLD = 0.70
    FAIL_THRESHOLD = 0.30
    
    a_pass = a >= PASS_THRESHOLD
    b_pass = b >= PASS_THRESHOLD
    c_pass = c >= PASS_THRESHOLD
    d_pass = d >= PASS_THRESHOLD
    e_pass = e >= PASS_THRESHOLD
    
    print(f"\n  H1 (Counting is hard):     {'SUPPORTED' if not a_pass else 'NOT SUPPORTED'}")
    print(f"      A={a:.1%} {'<' if not a_pass else '>='} {PASS_THRESHOLD:.0%}")
    
    print(f"\n  H2 (Constraint is hard):   {'SUPPORTED' if not b_pass else 'NOT SUPPORTED'}")
    print(f"      B={b:.1%} {'<' if not b_pass else '>='} {PASS_THRESHOLD:.0%}")
    
    print(f"\n  H3 (Binding is hard):      ", end="")
    if a_pass and b_pass and c_pass and not d_pass:
        print("SUPPORTED")
        print(f"      A,B,C pass; D fails")
    else:
        print("NOT SUPPORTED")
        print(f"      A={a:.1%}, B={b:.1%}, C={c:.1%}, D={d:.1%}")
    
    print(f"\n  H4 (Scratchpad helps):     ", end="")
    if not d_pass and e_pass:
        print("SUPPORTED")
        print(f"      D fails ({d:.1%}), E passes ({e:.1%})")
    elif not d_pass and e > d + 0.15:
        print("PARTIALLY SUPPORTED")
        print(f"      E > D by {e-d:.1%}")
    else:
        print("NOT SUPPORTED")
        print(f"      D={d:.1%}, E={e:.1%}")
    
    # Scratchpad-specific analysis
    if 'count_accuracy' in by_cond['E_SCRATCHPAD']:
        e_count = by_cond['E_SCRATCHPAD']['count_accuracy']
        e_bind = by_cond['E_SCRATCHPAD']['used_own_count_rate']
        print(f"\n  Scratchpad Breakdown:")
        print(f"      Counting correct:      {e_count:.1%}")
        print(f"      Used own count:        {e_bind:.1%}")
        print(f"      Full success:          {e:.1%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if not a_pass:
        print("\n⚠️  COUNTING IS HARD - Model struggles to count vowels accurately")
    
    if not b_pass:
        print("\n⚠️  FIXED CONSTRAINT IS HARD - Model struggles with word limits even when fixed")
    
    if a_pass and b_pass and not d_pass:
        if e_pass:
            print("\n✅ WORKING MEMORY BOTTLENECK - Model can do components but needs explicit scratchpad")
        elif e > d + 0.10:
            print("\n⚠️  PARTIAL WORKING MEMORY EFFECT - Scratchpad helps somewhat")
        else:
            print("\n❌ BINDING LIMITATION - Model cannot use computed values as constraints")
            print("   Even explicit scratchpad doesn't help")


def save_results(results: Dict[str, Any], output_dir: str = "results"):
    """Save results to JSON."""
    Path(output_dir).mkdir(exist_ok=True)
    
    model_slug = results['model'].replace(' ', '_').replace('-', '_').lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"l5_decomposition_{model_slug}_{timestamp}.json"
    filepath = Path(output_dir) / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filepath}")
    return filepath


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="L5 Decomposition Experiment: Where do input-dependent constraints fail?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Conditions:
  A: COUNT_ONLY        - Just count vowels
  B: FIXED_CONSTRAINT  - Respond in exactly 7 words  
  C: COUNT_AND_REPORT  - Count vowels, report, then respond
  D: FULL_L5           - Respond in N words where N = vowels
  E: SCRATCHPAD        - Show counting work, then respond in N words

Examples:
  python l5_decomposition.py
  python l5_decomposition.py --model gpt-4o --trials 30
  python l5_decomposition.py --all-models
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        choices=list(AVAILABLE_MODELS.keys()),
        help=f"Model to test (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=20,
        help="Trials per condition (default: 20)"
    )
    
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run on all available models"
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially (not parallel)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Determine models to test
    if args.all_models:
        models_to_test = []
        for name, config in AVAILABLE_MODELS.items():
            key_var = {
                'anthropic': 'ANTHROPIC_API_KEY',
                'openai': 'OPENAI_API_KEY',
                'google': 'GOOGLE_API_KEY'
            }.get(config.provider)
            
            if key_var and os.getenv(key_var):
                models_to_test.append(name)
        
        if not models_to_test:
            print("❌ No API keys found")
            sys.exit(1)
        
        print(f"Testing {len(models_to_test)} models: {', '.join(models_to_test)}")
    else:
        models_to_test = [args.model]
    
    # Run experiments
    all_results = []
    
    for model_name in models_to_test:
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print('='*70)
        
        try:
            model_config = AVAILABLE_MODELS[model_name]
            provider = create_provider(model_config)
        except Exception as e:
            print(f"❌ Could not create provider for {model_name}: {e}")
            continue
        
        config = DecompositionConfig(trials_per_condition=args.trials)
        experiment = L5DecompositionExperiment(provider, config)
        
        results = experiment.run(parallel=not args.sequential)
        print_results(results)
        save_results(results, args.output_dir)
        
        all_results.append(results)
    
    # Multi-model comparison
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("MULTI-MODEL COMPARISON")
        print("=" * 70)
        
        print(f"\n{'Model':<25} {'A (Count)':<12} {'B (Fixed)':<12} {'C (Report)':<12} {'D (L5)':<12} {'E (Scratch)':<12}")
        print("-" * 85)
        
        for r in all_results:
            model = r['model']
            by_c = r['by_condition']
            
            a = by_c['A_COUNT_ONLY']['accuracy']
            b = by_c['B_FIXED_CONSTRAINT']['accuracy']
            c = by_c['C_COUNT_AND_REPORT']['accuracy']
            d = by_c['D_FULL_L5']['accuracy']
            e = by_c['E_SCRATCHPAD']['accuracy']
            
            print(f"{model:<25} {a:<12.1%} {b:<12.1%} {c:<12.1%} {d:<12.1%} {e:<12.1%}")


if __name__ == "__main__":
    main()
