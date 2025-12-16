#!/usr/bin/env python3
"""
Run Arbitrary Rule Following Experiment
========================================
Main entry point for running the experiment.

Supports multiple models: Claude, GPT-4, Gemini

Usage:
    python run.py                        # Run with Claude Sonnet 4
    python run.py --model gpt-4o         # Run with GPT-4o
    python run.py --model gemini-1.5-pro # Run with Gemini
    python run.py --all-models           # Run all available models
    python run.py --trials 50            # More trials per level
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

from config import ExperimentConfig, AVAILABLE_MODELS, ModelConfig, MODEL_COMPARISON_SETS
from experiment import ArbitraryRuleExperiment, print_summary


def check_api_keys(models_to_test: list) -> dict:
    """Check which API keys are available."""
    providers_needed = set(m.provider for m in models_to_test)
    
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY"
    }
    
    available = {}
    missing = []
    
    for provider in providers_needed:
        env_var = key_map.get(provider)
        if env_var and os.getenv(env_var):
            available[provider] = True
        else:
            missing.append(f"{provider} ({env_var})")
    
    return {"available": available, "missing": missing}


def run_single_model(model_config: ModelConfig, args) -> dict:
    """Run experiment for a single model."""
    
    # Determine API key based on provider
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY"
    }
    api_key = os.getenv(key_map.get(model_config.provider, ""))
    
    # Create config with appropriate key
    config = ExperimentConfig(random_seed=args.seed)
    config.api_key = api_key
    config.model = model_config
    
    # Apply command line overrides
    if args.rules_per_level:
        config.statistical.rules_per_level = args.rules_per_level
    
    if args.prompts_per_rule:
        config.statistical.prompts_per_rule = args.prompts_per_rule
    
    # Calculate trials
    total_trials = (
        config.statistical.num_complexity_levels *
        config.statistical.rules_per_level *
        config.statistical.prompts_per_rule
    )
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {model_config.name}")
    print(f"{'='*60}")
    print(f"  Provider: {model_config.provider}")
    print(f"  API Model: {model_config.api_name}")
    print(f"  Total trials: {total_trials}")
    
    # Run experiment
    experiment = ArbitraryRuleExperiment(config)
    
    start_time = datetime.now()
    result = experiment.run_experiment(
        parallel=not args.sequential,
        max_workers=args.parallel
    )
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n⏱️  Completed in {elapsed:.1f} seconds")
    print_summary(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_config.name.lower().replace(" ", "_").replace(".", "_")
    filename = f"results_{model_slug}_{timestamp}.json"
    output_path = experiment.save_results(result, filename)
    
    return {
        "model": model_config.name,
        "results_file": str(output_path),
        "summary": result.summary,
        "elapsed_seconds": elapsed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Arbitrary Rule Following Experiment - Multi-Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  Anthropic:  claude-sonnet-4, claude-sonnet-3.5, claude-haiku-3.5
  OpenAI:     gpt-4o, gpt-4o-mini, gpt-4-turbo
  Google:     gemini-1.5-pro, gemini-1.5-flash

Examples:
    python run.py                           # Claude Sonnet 4 (default)
    python run.py --model gpt-4o            # Test GPT-4o
    python run.py --model gemini-1.5-pro    # Test Gemini
    python run.py --all-models              # Test all models with keys
    python run.py --rules-per-level 10      # 50 trials per level
    python run.py --high-power              # High statistical power (60 trials/level)
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="claude-sonnet-4",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to test (default: claude-sonnet-4)"
    )
    
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Test all models (requires all API keys)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Specific models to test"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)"
    )
    
    parser.add_argument(
        "--rules-per-level", "-r",
        type=int,
        default=None,
        help="Rules per complexity level (default: 6, gives 30 trials/level)"
    )
    
    parser.add_argument(
        "--prompts-per-rule", "-t",
        type=int,
        default=None,
        help="Test prompts per rule (default: 5)"
    )
    
    parser.add_argument(
        "--high-power",
        action="store_true",
        help="High statistical power mode: 12 rules × 5 prompts = 60 trials/level"
    )
    
    parser.add_argument(
        "--ultra-power",
        action="store_true",
        help="Ultra-high power mode: 24 rules × 5 prompts = 120 trials/level (for CI tightening)"
    )
    
    parser.add_argument(
        "--comparison-set",
        type=str,
        choices=list(MODEL_COMPARISON_SETS.keys()),
        help="Use a predefined model comparison set (frontier, fast, all)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially instead of parallel"
    )
    
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run offline test without API calls"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison across Claude, GPT-4o, and Gemini (requires all API keys)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print("\n📋 Available Models:")
        print("-" * 50)
        for key, config in AVAILABLE_MODELS.items():
            print(f"  {key:<20} ({config.provider}): {config.name}")
        return 0
    
    # High power mode
    if args.high_power:
        args.rules_per_level = 12  # 12 × 5 = 60 trials per level
        print("🔬 High statistical power mode: 60 trials per level")
    
    # Ultra-high power mode for CI tightening
    if args.ultra_power:
        args.rules_per_level = 24  # 24 × 5 = 120 trials per level
        print("🔬🔬 Ultra-high power mode: 120 trials per level (for tight CIs)")
    
    # Comparison set mode
    if args.comparison_set:
        args.models = MODEL_COMPARISON_SETS[args.comparison_set]
        print(f"📊 Comparison set '{args.comparison_set}': {args.models}")
    
    # Comparison mode - test major model families
    if args.compare:
        args.models = ["claude-sonnet-4", "gpt-4o", "gemini-1.5-pro"]
        print("📊 Comparison mode: Testing Claude, GPT-4o, and Gemini")
    
    # Determine models to test
    if args.all_models:
        models_to_test = list(AVAILABLE_MODELS.values())
    elif args.models:
        models_to_test = [AVAILABLE_MODELS[m] for m in args.models]
    else:
        models_to_test = [AVAILABLE_MODELS[args.model]]
    
    # Check API keys
    if not args.offline:
        key_status = check_api_keys(models_to_test)
        
        if key_status["missing"]:
            print(f"\n⚠️  Missing API keys for: {', '.join(key_status['missing'])}")
            
            # Filter to only models with available keys
            models_to_test = [
                m for m in models_to_test 
                if m.provider in key_status["available"]
            ]
            
            if not models_to_test:
                print("❌ No models available. Set required API keys:")
                print("   export ANTHROPIC_API_KEY='...'")
                print("   export OPENAI_API_KEY='...'")
                print("   export GOOGLE_API_KEY='...'")
                return 1
            
            print(f"   Will test: {[m.name for m in models_to_test]}")
    else:
        print("🔌 Running in OFFLINE mode (no API calls)")
        models_to_test = [AVAILABLE_MODELS[args.model]]
    
    # Run experiments
    all_results = []
    
    for model_config in models_to_test:
        try:
            result = run_single_model(model_config, args)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error testing {model_config.name}: {e}")
            all_results.append({
                "model": model_config.name,
                "error": str(e)
            })
    
    # Print comparison if multiple models
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("📊 MULTI-MODEL COMPARISON")
        print("=" * 70)
        
        print(f"\n{'Model':<25} {'Overall':<10} {'Easy(1-2)':<12} {'Hard(4-5)':<12} {'Degradation':<12}")
        print("-" * 70)
        
        for r in all_results:
            if "error" in r:
                print(f"{r['model']:<25} ERROR: {r['error'][:40]}")
            else:
                s = r["summary"]
                print(f"{r['model']:<25} {s['overall_accuracy']:<10.1%} {s['easy_level_accuracy']:<12.1%} {s['hard_level_accuracy']:<12.1%} {s['degradation']:<12.1%}")
        
        # APH prediction summary
        print("\n🔬 APH Prediction Confirmation:")
        for r in all_results:
            if "summary" in r:
                confirmed = r["summary"]["predictions"]["aph_prediction_confirmed"]
                status = "✅" if confirmed else "❌"
                print(f"  {status} {r['model']}")
    
    print("\n✅ All experiments complete!")
    for r in all_results:
        if "results_file" in r:
            print(f"   {r['model']}: {r['results_file']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
