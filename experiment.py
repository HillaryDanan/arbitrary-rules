"""
Arbitrary Rule Following Experiment
====================================
Tests APH's core prediction: LLMs fail on construction tasks
requiring application of arbitrary in-context rules.

This experiment:
1. Procedurally generates arbitrary rules (no human bias)
2. Tests model's ability to follow rules at 5 complexity levels
3. Analyzes degradation patterns to distinguish construction vs retrieval
"""

import json
import time
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from collections import defaultdict

from config import ExperimentConfig, COMPLEXITY_DESCRIPTIONS, AVAILABLE_MODELS
from rule_generator import RuleGenerator, Rule
from provider import create_provider, BaseProvider, APIResponse


@dataclass
class Trial:
    """A single trial in the experiment."""
    trial_id: str
    rule_id: str
    complexity_level: int
    rule_text: str
    prompt: str
    expected_actions: List[str]
    model_response: str
    compliant: bool
    actions_required: int
    actions_met: int
    evaluation_details: List[Dict]
    latency_ms: float
    api_success: bool
    error: Optional[str] = None


@dataclass
class ExperimentResult:
    """Results from the full experiment."""
    experiment_name: str
    model_name: str
    timestamp: str
    config_used: Dict[str, Any]
    trials: List[Trial]
    summary: Dict[str, Any]


class ArbitraryRuleExperiment:
    """Main experiment class."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rule_generator = RuleGenerator(config.rules)
        
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        
        # Initialize provider if API key available
        self.provider = None
        if config.api_key:
            try:
                self.provider = create_provider(config.model)
            except (ImportError, ValueError) as e:
                print(f"⚠️  Could not create provider: {e}")
    
    def _create_system_prompt(self, rule: Rule) -> str:
        """Create the system prompt with the rule."""
        return f"""You are participating in a rule-following experiment. You must follow the rule below EXACTLY when responding to any prompt.

RULE:
{rule.get_rule_text()}

IMPORTANT:
- Follow the rule precisely as stated
- The rule takes priority over normal response conventions
- Apply the rule to every response you give
- If the rule says to prepend something, it should be the VERY FIRST thing in your response
- If the rule says to append something, it should be the VERY LAST thing in your response
- If the rule specifies a word count, count carefully and respond with EXACTLY that many words

After following the rule, you may briefly answer the question if space/format allows."""

    def _create_user_prompt(self, prompt: str) -> str:
        """Create the user prompt."""
        return f"""Please respond to this prompt while following the rule:

{prompt}"""

    def run_trial(
        self, 
        rule: Rule, 
        prompt: str, 
        trial_num: int
    ) -> Trial:
        """Run a single trial."""
        trial_id = f"{rule.rule_id}_trial_{trial_num:03d}"
        
        system_prompt = self._create_system_prompt(rule)
        user_prompt = self._create_user_prompt(prompt)
        
        # Get expected actions for this prompt
        applicable_actions = rule.get_applicable_actions(prompt)
        expected_actions = [a.description for a in applicable_actions]
        
        # Make API call
        if self.provider:
            response = self.provider.call(system_prompt, user_prompt)
        else:
            # Mock response for testing
            response = APIResponse(
                success=True,
                content="Mock response - no API key",
                latency_ms=0,
                error=None
            )
        
        # Evaluate compliance
        if response.success:
            evaluation = rule.evaluate_response(
                response.content, 
                prompt,
                self.config.evaluation.case_sensitive
            )
        else:
            evaluation = {
                "compliant": False,
                "actions_required": len(applicable_actions),
                "actions_met": 0,
                "details": [{"error": response.error}]
            }
        
        return Trial(
            trial_id=trial_id,
            rule_id=rule.rule_id,
            complexity_level=rule.complexity_level,
            rule_text=rule.get_rule_text(),
            prompt=prompt,
            expected_actions=expected_actions,
            model_response=response.content,
            compliant=evaluation["compliant"],
            actions_required=evaluation["actions_required"],
            actions_met=evaluation["actions_met"],
            evaluation_details=evaluation["details"],
            latency_ms=response.latency_ms,
            api_success=response.success,
            error=response.error
        )
    
    def run_experiment(self, parallel: bool = True, max_workers: int = 5) -> ExperimentResult:
        """Run the full experiment."""
        print("=" * 60)
        print("ARBITRARY RULE FOLLOWING EXPERIMENT")
        print("=" * 60)
        print(f"Model: {self.config.model.name}")
        print(f"Complexity levels: {self.config.statistical.num_complexity_levels}")
        print(f"Rules per level: {self.config.statistical.rules_per_level}")
        print(f"Prompts per rule: {self.config.statistical.prompts_per_rule}")
        
        total_trials = (
            self.config.statistical.num_complexity_levels *
            self.config.statistical.rules_per_level *
            self.config.statistical.prompts_per_rule
        )
        print(f"Total trials: {total_trials}")
        print("=" * 60)
        
        all_trials = []
        
        # Generate all rules and prompts first
        print("\n📝 Generating rules...")
        all_tasks = []
        
        for level in range(1, self.config.statistical.num_complexity_levels + 1):
            print(f"  Level {level}: {COMPLEXITY_DESCRIPTIONS[level]['name']}")
            rules = self.rule_generator.generate_rules_for_level(
                level, 
                self.config.statistical.rules_per_level
            )
            
            for rule in rules:
                prompts = self.rule_generator.select_test_prompts(
                    rule,
                    self.config.statistical.prompts_per_rule
                )
                
                for i, prompt in enumerate(prompts):
                    all_tasks.append({
                        "rule": rule,
                        "prompt": prompt,
                        "trial_num": i + 1,
                        "level": level
                    })
        
        print(f"\n✅ Generated {len(all_tasks)} trial configurations")
        
        # Run trials
        if parallel and self.provider:
            print(f"\n🚀 Running trials in parallel (max_workers={max_workers})...")
            all_trials = self._run_parallel(all_tasks, max_workers)
        else:
            print("\n🚀 Running trials sequentially...")
            all_trials = self._run_sequential(all_tasks)
        
        # Compute summary statistics
        summary = self._compute_summary(all_trials)
        
        # Create result object
        result = ExperimentResult(
            experiment_name="arbitrary_rule_following",
            model_name=self.config.model.name,
            timestamp=datetime.now().isoformat(),
            config_used={
                "random_seed": self.config.random_seed,
                "rules_per_level": self.config.statistical.rules_per_level,
                "prompts_per_rule": self.config.statistical.prompts_per_rule,
                "num_complexity_levels": self.config.statistical.num_complexity_levels,
                "model": self.config.model.api_name,
                "evaluation_mode": self.config.evaluation.evaluation_mode
            },
            trials=all_trials,
            summary=summary
        )
        
        return result
    
    def _run_sequential(self, tasks: List[Dict]) -> List[Trial]:
        """Run trials sequentially."""
        trials = []
        total = len(tasks)
        
        for i, task in enumerate(tasks):
            trial = self.run_trial(
                task["rule"],
                task["prompt"],
                task["trial_num"]
            )
            trials.append(trial)
            
            # Progress update
            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  Progress: {i + 1}/{total} ({100*(i+1)/total:.1f}%)")
        
        return trials
    
    def _run_parallel(self, tasks: List[Dict], max_workers: int) -> List[Trial]:
        """Run trials in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        trials = []
        total = len(tasks)
        completed = 0
        
        def run_single(task):
            return self.run_trial(
                task["rule"],
                task["prompt"],
                task["trial_num"]
            )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single, task): task for task in tasks}
            
            for future in as_completed(futures):
                trial = future.result()
                trials.append(trial)
                completed += 1
                
                if completed % 20 == 0 or completed == total:
                    print(f"  Progress: {completed}/{total} ({100*completed/total:.1f}%)")
        
        # Sort by trial_id to maintain order
        trials.sort(key=lambda t: t.trial_id)
        return trials
    
    def _compute_summary(self, trials: List[Trial]) -> Dict[str, Any]:
        """Compute summary statistics."""
        stats = self.config.statistical
        
        # Group by complexity level
        by_level = defaultdict(list)
        for trial in trials:
            by_level[trial.complexity_level].append(trial)
        
        level_accuracies = {}
        level_details = {}
        
        for level in sorted(by_level.keys()):
            level_trials = by_level[level]
            compliant_count = sum(1 for t in level_trials if t.compliant)
            total_count = len(level_trials)
            accuracy = compliant_count / total_count if total_count > 0 else 0
            
            # Partial compliance (actions met / actions required)
            total_actions_required = sum(t.actions_required for t in level_trials)
            total_actions_met = sum(t.actions_met for t in level_trials)
            partial_compliance = total_actions_met / total_actions_required if total_actions_required > 0 else 0
            
            # Latency stats
            latencies = [t.latency_ms for t in level_trials if t.api_success]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            level_accuracies[level] = accuracy
            level_details[level] = {
                "accuracy": accuracy,
                "compliant_trials": compliant_count,
                "total_trials": total_count,
                "partial_compliance": partial_compliance,
                "avg_latency_ms": avg_latency,
                "description": COMPLEXITY_DESCRIPTIONS[level]["name"],
                "aph_mapping": COMPLEXITY_DESCRIPTIONS[level]["aph_mapping"]
            }
        
        # Overall statistics
        total_compliant = sum(1 for t in trials if t.compliant)
        total_trials = len(trials)
        overall_accuracy = total_compliant / total_trials if total_trials > 0 else 0
        
        # Check for degradation pattern (APH prediction)
        easy_levels = [1, 2]
        hard_levels = [4, 5]
        
        easy_accuracy = sum(level_accuracies.get(l, 0) for l in easy_levels) / len(easy_levels)
        hard_accuracy = sum(level_accuracies.get(l, 0) for l in hard_levels) / len(hard_levels)
        degradation = easy_accuracy - hard_accuracy
        
        # Determine if predictions are confirmed
        prediction_easy_success = easy_accuracy >= stats.easy_level_threshold
        prediction_degradation = degradation >= stats.degradation_threshold
        prediction_hard_failure = hard_accuracy <= stats.hard_level_threshold
        
        # Check for sharp degradation (threshold collapse pattern)
        sharp_degradation_detected = False
        degradation_point = None
        for level in range(1, self.config.statistical.num_complexity_levels):
            if level in level_accuracies and level + 1 in level_accuracies:
                drop = level_accuracies[level] - level_accuracies[level + 1]
                if drop >= stats.sharp_degradation_threshold:
                    sharp_degradation_detected = True
                    degradation_point = level
                    break
        
        # APH prediction: construction vs retrieval
        # Construction signature: graceful degradation
        # Retrieval signature: sharp threshold collapse
        aph_prediction_confirmed = (
            prediction_degradation and 
            (sharp_degradation_detected or prediction_hard_failure)
        )
        
        summary = {
            "overall_accuracy": overall_accuracy,
            "total_trials": total_trials,
            "compliant_trials": total_compliant,
            "accuracy_by_level": level_accuracies,
            "level_details": level_details,
            "easy_level_accuracy": easy_accuracy,
            "hard_level_accuracy": hard_accuracy,
            "degradation": degradation,
            "sharp_degradation_detected": sharp_degradation_detected,
            "degradation_point": degradation_point,
            "predictions": {
                "easy_success": prediction_easy_success,
                "degradation_observed": prediction_degradation,
                "hard_failure": prediction_hard_failure,
                "aph_prediction_confirmed": aph_prediction_confirmed
            },
            "thresholds_used": {
                "easy_level_threshold": stats.easy_level_threshold,
                "degradation_threshold": stats.degradation_threshold,
                "hard_level_threshold": stats.hard_level_threshold,
                "sharp_degradation_threshold": stats.sharp_degradation_threshold
            }
        }
        
        return summary
    
    def save_results(self, result: ExperimentResult, filename: str = None) -> Path:
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arbitrary_rules_{timestamp}.json"
        
        output_path = self.config.results_dir / filename
        
        # Convert to dict (handle dataclasses)
        result_dict = {
            "experiment_name": result.experiment_name,
            "model_name": result.model_name,
            "timestamp": result.timestamp,
            "config_used": result.config_used,
            "summary": result.summary,
            "trials": [asdict(t) for t in result.trials]
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_path}")
        return output_path


def print_summary(result: ExperimentResult):
    """Print a formatted summary of results."""
    s = result.summary
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print(f"\nModel: {result.model_name}")
    print(f"Total Trials: {s['total_trials']}")
    print(f"Overall Accuracy: {s['overall_accuracy']:.1%}")
    
    print("\n📊 ACCURACY BY COMPLEXITY LEVEL:")
    print("-" * 50)
    
    for level, details in s['level_details'].items():
        bar_length = int(details['accuracy'] * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  Level {level} ({details['description']:<20}): {details['accuracy']:>6.1%} |{bar}|")
        print(f"           {details['aph_mapping']:<20}   ({details['compliant_trials']}/{details['total_trials']} trials)")
    
    print("\n📈 DEGRADATION ANALYSIS:")
    print("-" * 50)
    print(f"  Easy levels (1-2) accuracy: {s['easy_level_accuracy']:.1%}")
    print(f"  Hard levels (4-5) accuracy: {s['hard_level_accuracy']:.1%}")
    print(f"  Degradation: {s['degradation']:.1%}")
    
    if s['sharp_degradation_detected']:
        print(f"  ⚠️  Sharp degradation detected at level {s['degradation_point']} → {s['degradation_point'] + 1}")
    
    print("\n🔬 APH PREDICTIONS:")
    print("-" * 50)
    preds = s['predictions']
    
    def status(val):
        return "✅ CONFIRMED" if val else "❌ NOT CONFIRMED"
    
    print(f"  Easy level success (≥{s['thresholds_used']['easy_level_threshold']:.0%}): {status(preds['easy_success'])}")
    print(f"  Degradation observed (≥{s['thresholds_used']['degradation_threshold']:.0%}): {status(preds['degradation_observed'])}")
    print(f"  Hard level failure (≤{s['thresholds_used']['hard_level_threshold']:.0%}): {status(preds['hard_failure'])}")
    print(f"\n  📌 APH PREDICTION (retrieval, not construction): {status(preds['aph_prediction_confirmed'])}")
    
    print("\n" + "=" * 60)
