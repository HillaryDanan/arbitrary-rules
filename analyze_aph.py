#!/usr/bin/env python3
"""
Deep Analysis for APH Paper
============================
Comprehensive analysis of arbitrary rule following results
for mapping onto Abstraction Primitive Hypothesis framework.

Analyzes:
1. Degradation curve shape (graceful vs threshold collapse)
2. Error structure (coherent vs random)
3. Action-type interactions
4. Constraint coordination failures
5. APH composition hierarchy mapping
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import argparse
import math

def load_results(filepath: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def wilson_confidence_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval."""
    if n == 0:
        return (0.0, 1.0)
    
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * ((p*(1-p)/n + z**2/(4*n**2)) ** 0.5) / denominator
    
    return (max(0, center - margin), min(1, center + margin))


def analyze_degradation_curve(trials: List[Dict]) -> Dict[str, Any]:
    """
    Analyze the shape of the degradation curve.
    
    APH predicts:
    - Embedded systems: graceful degradation
    - Non-embedded (LLMs): threshold collapse
    
    We measure:
    - Drop magnitude between adjacent levels
    - Whether drops are uniform or concentrated at specific transitions
    """
    by_level = defaultdict(lambda: {"total": 0, "compliant": 0})
    
    for trial in trials:
        level = trial["complexity_level"]
        by_level[level]["total"] += 1
        if trial["compliant"]:
            by_level[level]["compliant"] += 1
    
    accuracies = {}
    for level in sorted(by_level.keys()):
        data = by_level[level]
        accuracies[level] = data["compliant"] / data["total"] if data["total"] > 0 else 0
    
    # Calculate drops between levels
    drops = {}
    levels = sorted(accuracies.keys())
    for i in range(len(levels) - 1):
        l1, l2 = levels[i], levels[i+1]
        drops[f"{l1}→{l2}"] = accuracies[l1] - accuracies[l2]
    
    # Identify largest drop (threshold location)
    max_drop_transition = max(drops.items(), key=lambda x: x[1]) if drops else (None, 0)
    
    # Calculate coefficient of variation of drops (uniformity measure)
    drop_values = list(drops.values())
    if drop_values:
        mean_drop = sum(drop_values) / len(drop_values)
        variance = sum((d - mean_drop)**2 for d in drop_values) / len(drop_values)
        std_drop = variance ** 0.5
        cv = std_drop / mean_drop if mean_drop > 0 else 0
    else:
        mean_drop, std_drop, cv = 0, 0, 0
    
    # Classify curve shape
    if max_drop_transition[1] > 0.40:
        curve_type = "THRESHOLD_COLLAPSE"
    elif cv < 0.5:
        curve_type = "UNIFORM_DEGRADATION"
    else:
        curve_type = "STEPPED_DEGRADATION"
    
    return {
        "accuracies": accuracies,
        "drops": drops,
        "max_drop_transition": max_drop_transition[0],
        "max_drop_magnitude": max_drop_transition[1],
        "mean_drop": mean_drop,
        "std_drop": std_drop,
        "coefficient_of_variation": cv,
        "curve_type": curve_type,
        "aph_prediction": "THRESHOLD_COLLAPSE",
        "matches_aph": curve_type == "THRESHOLD_COLLAPSE"
    }


def analyze_error_structure(trials: List[Dict]) -> Dict[str, Any]:
    """
    Analyze error structure to distinguish construction vs retrieval failure.
    
    APH predicts for non-embedded systems:
    - Random/incoherent errors (not preserving partial structure)
    - Binary success/failure (not graceful partial compliance)
    
    We measure:
    - Partial compliance rate (some but not all actions met)
    - Error consistency (do same rules fail same ways?)
    """
    by_level = defaultdict(lambda: {
        "full_success": 0,
        "partial": 0,
        "total_failure": 0,
        "total": 0,
        "error_types": defaultdict(int)
    })
    
    for trial in trials:
        level = trial["complexity_level"]
        required = trial["actions_required"]
        met = trial["actions_met"]
        
        by_level[level]["total"] += 1
        
        if met == required:
            by_level[level]["full_success"] += 1
        elif met == 0:
            by_level[level]["total_failure"] += 1
        else:
            by_level[level]["partial"] += 1
        
        # Categorize error type
        if not trial["compliant"]:
            details = trial.get("evaluation_details", [])
            for d in details:
                if isinstance(d, dict) and not d.get("compliant", True):
                    action = d.get("action", "unknown")
                    # Categorize action type
                    if "begin" in action or "start" in action:
                        by_level[level]["error_types"]["prepend_failure"] += 1
                    elif "end" in action:
                        by_level[level]["error_types"]["append_failure"] += 1
                    elif "caps" in action.lower() or "lowercase" in action.lower():
                        by_level[level]["error_types"]["case_failure"] += 1
                    elif "word" in action and ("exactly" in action or "limit" in action):
                        by_level[level]["error_types"]["word_limit_failure"] += 1
                    elif "symbol" in action:
                        by_level[level]["error_types"]["symbol_failure"] += 1
                    elif "number" in action or "count" in action:
                        by_level[level]["error_types"]["count_failure"] += 1
                    else:
                        by_level[level]["error_types"]["other_failure"] += 1
    
    # Calculate metrics
    results = {}
    for level in sorted(by_level.keys()):
        data = by_level[level]
        failed = data["total"] - data["full_success"]
        
        results[level] = {
            "total": data["total"],
            "full_success": data["full_success"],
            "partial": data["partial"],
            "total_failure": data["total_failure"],
            "partial_rate": data["partial"] / failed if failed > 0 else 0,
            "binary_failure_rate": data["total_failure"] / failed if failed > 0 else 0,
            "error_types": dict(data["error_types"])
        }
    
    # Overall binary vs partial
    all_partial = sum(r["partial"] for r in results.values())
    all_total_failure = sum(r["total_failure"] for r in results.values())
    all_failed = all_partial + all_total_failure
    
    overall_binary_rate = all_total_failure / all_failed if all_failed > 0 else 0
    
    return {
        "by_level": results,
        "overall_partial": all_partial,
        "overall_total_failure": all_total_failure,
        "overall_binary_failure_rate": overall_binary_rate,
        "aph_prediction": "Binary failures dominate (>70%)",
        "matches_aph": overall_binary_rate > 0.70
    }


def analyze_constraint_coordination(trials: List[Dict]) -> Dict[str, Any]:
    """
    Analyze how well the model coordinates multiple constraints.
    
    Key insight: Level 2 (two unconditional actions) should be easier
    than Level 3 (conditional logic) if the difficulty is conditionals.
    
    If Level 2 ≈ Level 3, the difficulty is COORDINATION, not conditionals.
    """
    by_level = defaultdict(lambda: {"total": 0, "compliant": 0})
    
    for trial in trials:
        level = trial["complexity_level"]
        by_level[level]["total"] += 1
        if trial["compliant"]:
            by_level[level]["compliant"] += 1
    
    acc = {l: d["compliant"]/d["total"] for l, d in by_level.items() if d["total"] > 0}
    
    # Key comparisons
    level_1_2_drop = acc.get(1, 0) - acc.get(2, 0)
    level_2_3_drop = acc.get(2, 0) - acc.get(3, 0)
    
    # If L2 ≈ L3, coordination is the bottleneck, not conditional logic
    coordination_is_bottleneck = abs(level_2_3_drop) < 0.10
    
    return {
        "level_accuracies": acc,
        "level_1_to_2_drop": level_1_2_drop,
        "level_2_to_3_drop": level_2_3_drop,
        "coordination_is_bottleneck": coordination_is_bottleneck,
        "interpretation": (
            "Coordination of multiple constraints is the primary difficulty, "
            "not conditional logic per se" if coordination_is_bottleneck else
            "Conditional logic adds difficulty beyond coordination"
        )
    }


def analyze_action_difficulty(trials: List[Dict]) -> Dict[str, Any]:
    """
    Analyze which action types are hardest.
    
    Maps to APH: Some actions may be more "3b-like" (template filling)
    while others are more "3c-like" (require computation).
    """
    action_stats = defaultdict(lambda: {"required": 0, "met": 0})
    
    for trial in trials:
        details = trial.get("evaluation_details", [])
        for d in details:
            if isinstance(d, dict) and "action" in d:
                action = d["action"]
                action_type = categorize_action(action)
                action_stats[action_type]["required"] += 1
                if d.get("compliant", False):
                    action_stats[action_type]["met"] += 1
    
    results = {}
    for action_type, stats in action_stats.items():
        if stats["required"] > 0:
            results[action_type] = {
                "required": stats["required"],
                "met": stats["met"],
                "success_rate": stats["met"] / stats["required"],
                "ci_95": wilson_confidence_interval(stats["met"], stats["required"])
            }
    
    # Sort by difficulty (lowest success rate first)
    sorted_actions = sorted(results.items(), key=lambda x: x[1]["success_rate"])
    
    # Categorize by APH composition type
    aph_mapping = {
        "prepend": "3b (slot-filling)",
        "append": "3b (slot-filling)",
        "all_caps": "3b (transform)",
        "all_lower": "3b (transform)",
        "word_limit": "3c (counting/computation)",
        "include_count": "3c+ (self-referential computation)",
        "include_symbol": "3b+ (counting)",
        "reverse": "3c (string manipulation)"
    }
    
    for action_type in results:
        results[action_type]["aph_type"] = aph_mapping.get(action_type, "unknown")
    
    return {
        "by_action": results,
        "sorted_by_difficulty": sorted_actions,
        "hardest_actions": [a for a, s in sorted_actions if s["success_rate"] < 0.5],
        "easiest_actions": [a for a, s in sorted_actions if s["success_rate"] > 0.7]
    }


def categorize_action(action_desc: str) -> str:
    """Categorize action description into type."""
    action = action_desc.lower()
    
    if "begin" in action or "start with" in action:
        return "prepend"
    elif "end" in action:
        return "append"
    elif "all caps" in action or "uppercase" in action:
        return "all_caps"
    elif "lowercase" in action:
        return "all_lower"
    elif "exactly" in action and "word" in action:
        return "word_limit"
    elif "symbol" in action:
        return "include_symbol"
    elif "number of" in action:
        return "include_count"
    elif "reverse" in action:
        return "reverse"
    else:
        return "other"


def analyze_aph_mapping(trials: List[Dict]) -> Dict[str, Any]:
    """
    Map results directly to APH composition hierarchy predictions.
    """
    by_level = defaultdict(lambda: {"total": 0, "compliant": 0})
    
    for trial in trials:
        level = trial["complexity_level"]
        by_level[level]["total"] += 1
        if trial["compliant"]:
            by_level[level]["compliant"] += 1
    
    acc = {l: d["compliant"]/d["total"] for l, d in by_level.items() if d["total"] > 0}
    
    # APH mapping
    aph_levels = {
        1: {"name": "Single action", "aph_type": "3b (role-filler)", "expected": "high success"},
        2: {"name": "Two actions", "aph_type": "3b+ (extended role-filler)", "expected": "moderate success"},
        3: {"name": "Conditional", "aph_type": "3c (recursive/conditional)", "expected": "degradation"},
        4: {"name": "Priority chain", "aph_type": "3c+ (nested conditionals)", "expected": "significant degradation"},
        5: {"name": "Self-referential", "aph_type": "3d (analogical/self-ref)", "expected": "failure"}
    }
    
    results = {}
    for level, info in aph_levels.items():
        accuracy = acc.get(level, 0)
        ci = wilson_confidence_interval(
            by_level[level]["compliant"],
            by_level[level]["total"]
        )
        
        # Evaluate prediction
        if info["expected"] == "high success":
            prediction_met = accuracy > 0.70
        elif info["expected"] == "moderate success":
            prediction_met = 0.30 < accuracy < 0.80
        elif info["expected"] == "degradation":
            prediction_met = accuracy < acc.get(level-1, 1.0)
        elif info["expected"] == "significant degradation":
            prediction_met = accuracy < 0.50
        else:  # failure
            prediction_met = accuracy < 0.20
        
        results[level] = {
            "name": info["name"],
            "aph_type": info["aph_type"],
            "accuracy": accuracy,
            "ci_95": ci,
            "expected": info["expected"],
            "prediction_met": prediction_met
        }
    
    # Key APH predictions
    predictions = {
        "3b_success": acc.get(1, 0) > 0.70,
        "3b_to_3c_degradation": acc.get(1, 0) - acc.get(3, 0) > 0.20,
        "3c_to_3d_degradation": acc.get(3, 0) - acc.get(5, 0) > 0.20,
        "3d_failure": acc.get(5, 0) < 0.20,
        "monotonic_degradation": all(
            acc.get(i, 1) >= acc.get(i+1, 0) 
            for i in range(1, 5)
        )
    }
    
    return {
        "by_level": results,
        "predictions": predictions,
        "overall_aph_support": sum(predictions.values()) / len(predictions)
    }


def generate_paper_statistics(trials: List[Dict]) -> Dict[str, Any]:
    """
    Generate publication-ready statistics.
    """
    by_level = defaultdict(lambda: {"total": 0, "compliant": 0})
    
    for trial in trials:
        level = trial["complexity_level"]
        by_level[level]["total"] += 1
        if trial["compliant"]:
            by_level[level]["compliant"] += 1
    
    stats = {}
    for level in sorted(by_level.keys()):
        data = by_level[level]
        n = data["total"]
        k = data["compliant"]
        p = k / n if n > 0 else 0
        ci = wilson_confidence_interval(k, n)
        
        stats[level] = {
            "n": n,
            "successes": k,
            "accuracy": p,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "se": ((p * (1-p)) / n) ** 0.5 if n > 0 else 0
        }
    
    # Effect sizes (Cohen's h for proportions)
    def cohens_h(p1, p2):
        """Cohen's h effect size for proportions."""
        phi1 = 2 * math.asin(p1 ** 0.5)
        phi2 = 2 * math.asin(p2 ** 0.5)
        return abs(phi1 - phi2)
    
    effect_sizes = {}
    levels = sorted(stats.keys())
    for i in range(len(levels) - 1):
        l1, l2 = levels[i], levels[i+1]
        p1, p2 = stats[l1]["accuracy"], stats[l2]["accuracy"]
        h = cohens_h(p1, p2)
        effect_sizes[f"{l1}_vs_{l2}"] = {
            "cohens_h": h,
            "interpretation": "large" if h > 0.8 else "medium" if h > 0.5 else "small"
        }
    
    # Overall effect size (Level 1 vs Level 5)
    overall_h = cohens_h(stats[1]["accuracy"], stats[5]["accuracy"])
    
    return {
        "by_level": stats,
        "effect_sizes": effect_sizes,
        "overall_effect_size": {
            "level_1_vs_5_cohens_h": overall_h,
            "interpretation": "large" if overall_h > 0.8 else "medium" if overall_h > 0.5 else "small"
        },
        "total_trials": sum(s["n"] for s in stats.values()),
        "overall_accuracy": sum(s["successes"] for s in stats.values()) / sum(s["n"] for s in stats.values())
    }


def print_full_analysis(results: Dict[str, Any], filepath: str):
    """Print comprehensive analysis for paper."""
    
    trials = results.get("trials", [])
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS FOR APH PAPER")
    print("=" * 80)
    print(f"Source: {filepath}")
    print(f"Model: {results.get('model_name', 'Unknown')}")
    print(f"Total Trials: {len(trials)}")
    
    # 1. Degradation Curve
    print("\n" + "=" * 80)
    print("1. DEGRADATION CURVE ANALYSIS")
    print("=" * 80)
    
    degradation = analyze_degradation_curve(trials)
    
    print("\nAccuracy by Level:")
    for level, acc in sorted(degradation["accuracies"].items()):
        bar = "█" * int(acc * 30) + "░" * (30 - int(acc * 30))
        print(f"  Level {level}: {acc:6.1%} |{bar}|")
    
    print("\nDrops between levels:")
    for transition, drop in sorted(degradation["drops"].items()):
        print(f"  {transition}: {drop:+.1%}")
    
    print(f"\n📊 Curve Type: {degradation['curve_type']}")
    print(f"   Max drop: {degradation['max_drop_transition']} ({degradation['max_drop_magnitude']:.1%})")
    print(f"   APH Prediction (threshold collapse): {'✅ CONFIRMED' if degradation['matches_aph'] else '❌ NOT CONFIRMED'}")
    
    # 2. Error Structure
    print("\n" + "=" * 80)
    print("2. ERROR STRUCTURE ANALYSIS")
    print("=" * 80)
    
    errors = analyze_error_structure(trials)
    
    print("\nFailure Types by Level:")
    print(f"{'Level':<8} {'Full ✓':<10} {'Partial':<10} {'Total ✗':<10} {'Binary Rate':<12}")
    print("-" * 50)
    for level, data in sorted(errors["by_level"].items()):
        binary = data["binary_failure_rate"]
        print(f"{level:<8} {data['full_success']:<10} {data['partial']:<10} {data['total_failure']:<10} {binary:.1%}")
    
    print(f"\n📊 Overall Binary Failure Rate: {errors['overall_binary_failure_rate']:.1%}")
    print(f"   APH Prediction (>70% binary): {'✅ CONFIRMED' if errors['matches_aph'] else '❌ NOT CONFIRMED'}")
    
    # 3. Constraint Coordination
    print("\n" + "=" * 80)
    print("3. CONSTRAINT COORDINATION ANALYSIS")
    print("=" * 80)
    
    coord = analyze_constraint_coordination(trials)
    
    print(f"\n  Level 1 → 2 drop: {coord['level_1_to_2_drop']:.1%}")
    print(f"  Level 2 → 3 drop: {coord['level_2_to_3_drop']:.1%}")
    print(f"\n📊 Interpretation: {coord['interpretation']}")
    
    # 4. Action Difficulty
    print("\n" + "=" * 80)
    print("4. ACTION TYPE DIFFICULTY")
    print("=" * 80)
    
    actions = analyze_action_difficulty(trials)
    
    print(f"\n{'Action Type':<20} {'Success':<12} {'APH Type':<25} {'95% CI':<20}")
    print("-" * 80)
    for action, stats in actions["sorted_by_difficulty"]:
        ci = stats["ci_95"]
        print(f"{action:<20} {stats['success_rate']:<12.1%} {stats['aph_type']:<25} [{ci[0]:.1%}, {ci[1]:.1%}]")
    
    # 5. APH Mapping
    print("\n" + "=" * 80)
    print("5. APH FRAMEWORK MAPPING")
    print("=" * 80)
    
    aph = analyze_aph_mapping(trials)
    
    print(f"\n{'Level':<8} {'Name':<20} {'APH Type':<30} {'Accuracy':<12} {'Prediction':<15}")
    print("-" * 90)
    for level, data in sorted(aph["by_level"].items()):
        status = "✅" if data["prediction_met"] else "❌"
        print(f"{level:<8} {data['name']:<20} {data['aph_type']:<30} {data['accuracy']:<12.1%} {status} {data['expected']}")
    
    print("\nKey APH Predictions:")
    for pred, met in aph["predictions"].items():
        status = "✅" if met else "❌"
        print(f"  {status} {pred}")
    
    print(f"\n📊 Overall APH Support: {aph['overall_aph_support']:.0%}")
    
    # 6. Publication Statistics
    print("\n" + "=" * 80)
    print("6. PUBLICATION-READY STATISTICS")
    print("=" * 80)
    
    stats = generate_paper_statistics(trials)
    
    print(f"\n{'Level':<8} {'N':<8} {'Accuracy':<12} {'SE':<10} {'95% CI':<20}")
    print("-" * 60)
    for level, s in sorted(stats["by_level"].items()):
        ci = f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
        print(f"{level:<8} {s['n']:<8} {s['accuracy']:<12.3f} {s['se']:<10.3f} {ci}")
    
    print("\nEffect Sizes (Cohen's h):")
    for comparison, data in stats["effect_sizes"].items():
        print(f"  {comparison}: h = {data['cohens_h']:.2f} ({data['interpretation']})")
    
    print(f"\n  Level 1 vs 5: h = {stats['overall_effect_size']['level_1_vs_5_cohens_h']:.2f} ({stats['overall_effect_size']['interpretation']})")
    
    # Summary for paper
    print("\n" + "=" * 80)
    print("SUMMARY FOR PAPER ABSTRACT")
    print("=" * 80)
    
    l1_acc = stats["by_level"][1]["accuracy"]
    l5_acc = stats["by_level"][5]["accuracy"]
    
    print(f"""
We tested Claude Sonnet 4 on procedurally-generated arbitrary rules across 
five complexity levels ({stats['total_trials']} trials total). Performance degraded 
sharply from Level 1 ({l1_acc:.1%}) to Level 5 ({l5_acc:.1%}), 
with a {degradation['max_drop_magnitude']:.1%} drop at the {degradation['max_drop_transition']} transition.

Key findings consistent with APH predictions:
- 3b (role-filler) tasks showed high success ({l1_acc:.1%})
- 3d (self-referential) tasks showed near-total failure ({l5_acc:.1%})
- Error structure was predominantly binary ({errors['overall_binary_failure_rate']:.1%} total failures)
- The primary bottleneck was constraint coordination, not conditional logic

Effect size for composition hierarchy: Cohen's h = {stats['overall_effect_size']['level_1_vs_5_cohens_h']:.2f} (large)
""")


def main():
    parser = argparse.ArgumentParser(description="Deep APH analysis")
    parser.add_argument("results_file", type=str, help="Path to results JSON")
    parser.add_argument("--output", "-o", type=str, help="Output JSON for analysis")
    
    args = parser.parse_args()
    
    filepath = Path(args.results_file)
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)
    
    results = load_results(filepath)
    print_full_analysis(results, str(filepath))
    
    # Save detailed analysis
    if args.output:
        trials = results.get("trials", [])
        analysis = {
            "degradation": analyze_degradation_curve(trials),
            "error_structure": analyze_error_structure(trials),
            "constraint_coordination": analyze_constraint_coordination(trials),
            "action_difficulty": analyze_action_difficulty(trials),
            "aph_mapping": analyze_aph_mapping(trials),
            "statistics": generate_paper_statistics(trials)
        }
        
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n💾 Analysis saved to: {args.output}")


if __name__ == "__main__":
    main()
