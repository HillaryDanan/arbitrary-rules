#!/usr/bin/env python3
"""
Offline Test - Verify Setup Without API Calls
==============================================
Tests that rule generation and evaluation work correctly.
"""

import sys
from config import ExperimentConfig, COMPLEXITY_DESCRIPTIONS
from rule_generator import RuleGenerator, Rule


def test_rule_generation():
    """Test that rules generate correctly at all levels."""
    print("=" * 60)
    print("OFFLINE TEST: Rule Generation")
    print("=" * 60)
    
    config = ExperimentConfig()
    generator = RuleGenerator(config.rules)
    
    all_passed = True
    
    for level in range(1, 6):
        print(f"\n📋 Level {level}: {COMPLEXITY_DESCRIPTIONS[level]['name']}")
        print("-" * 50)
        
        try:
            rule = generator.generate_rule(level)
            print(f"  Rule ID: {rule.rule_id}")
            print(f"  Logic: {rule.logic}")
            print(f"  Rule text:\n    {rule.get_rule_text().replace(chr(10), chr(10) + '    ')}")
            
            # Test with a sample prompt
            test_prompt = "What color is the sky?"
            applicable = rule.get_applicable_actions(test_prompt)
            print(f"\n  Test prompt: '{test_prompt}'")
            print(f"  Applicable actions: {len(applicable)}")
            for action in applicable:
                print(f"    - {action.description}")
            
            # Test evaluation
            mock_response = "BEEP The sky is blue ...maybe"
            evaluation = rule.evaluate_response(mock_response, test_prompt)
            print(f"\n  Mock response: '{mock_response}'")
            print(f"  Compliant: {evaluation['compliant']}")
            print(f"  Actions met: {evaluation['actions_met']}/{evaluation['actions_required']}")
            
            print("  ✅ Level passed")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_passed = False
    
    return all_passed


def test_condition_evaluation():
    """Test that conditions evaluate correctly."""
    print("\n" + "=" * 60)
    print("OFFLINE TEST: Condition Evaluation")
    print("=" * 60)
    
    from rule_generator import Condition, ConditionType
    
    test_cases = [
        (ConditionType.LETTER_CONTAINS, {"letter": "e"}, "Hello world", True),
        (ConditionType.LETTER_CONTAINS, {"letter": "z"}, "Hello world", False),
        (ConditionType.LETTER_STARTS, {"letter": "h"}, "Hello world", True),
        (ConditionType.LETTER_STARTS, {"letter": "w"}, "Hello world", False),
        (ConditionType.WORD_COUNT_PARITY, {"parity": "even"}, "Hello world", True),
        (ConditionType.WORD_COUNT_PARITY, {"parity": "odd"}, "Hello world", False),
        (ConditionType.CONTAINS_WORD, {"word": "the"}, "What is the sky?", True),
        (ConditionType.CONTAINS_WORD, {"word": "the"}, "What is a sky?", False),
        (ConditionType.VOWEL_COUNT_THRESHOLD, {"threshold": 1, "operator": ">"}, "Hello", True),  # "Hello" has 2 vowels
        (ConditionType.VOWEL_COUNT_THRESHOLD, {"threshold": 5, "operator": ">"}, "Hello", False),
    ]
    
    all_passed = True
    
    for cond_type, params, prompt, expected in test_cases:
        condition = Condition(
            condition_type=cond_type,
            parameters=params,
            description=f"Test condition"
        )
        result = condition.evaluate(prompt)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {cond_type.value}: '{prompt}' → {result} (expected {expected})")
        if result != expected:
            all_passed = False
    
    return all_passed


def test_action_compliance():
    """Test that action compliance checking works."""
    print("\n" + "=" * 60)
    print("OFFLINE TEST: Action Compliance")
    print("=" * 60)
    
    from rule_generator import Action, ActionType
    
    test_cases = [
        (ActionType.PREPEND, {"word": "BEEP"}, "BEEP Hello", True),
        (ActionType.PREPEND, {"word": "BEEP"}, "Hello BEEP", False),
        (ActionType.APPEND, {"word": "...maybe"}, "Hello ...maybe", True),
        (ActionType.APPEND, {"word": "...maybe"}, "...maybe Hello", False),
        (ActionType.ALL_CAPS, {}, "HELLO WORLD", True),
        (ActionType.ALL_CAPS, {}, "Hello World", False),
        (ActionType.WORD_LIMIT, {"limit": 3}, "one two three", True),
        (ActionType.WORD_LIMIT, {"limit": 3}, "one two", False),
        (ActionType.INCLUDE_SYMBOL, {"symbol": "★", "count": 2}, "Hello ★ world ★", True),
        (ActionType.INCLUDE_SYMBOL, {"symbol": "★", "count": 2}, "Hello ★ world", False),
    ]
    
    all_passed = True
    prompt = "Test prompt"
    
    for action_type, params, response, expected in test_cases:
        action = Action(
            action_type=action_type,
            parameters=params,
            description=f"Test action"
        )
        compliant, explanation = action.check_compliance(response, prompt)
        status = "✅" if compliant == expected else "❌"
        print(f"  {status} {action_type.value}: '{response[:30]}...' → {compliant} (expected {expected})")
        if compliant != expected:
            all_passed = False
    
    return all_passed


def test_prompt_selection():
    """Test that prompt selection works."""
    print("\n" + "=" * 60)
    print("OFFLINE TEST: Prompt Selection")
    print("=" * 60)
    
    config = ExperimentConfig()
    generator = RuleGenerator(config.rules)
    
    rule = generator.generate_rule(3)  # IF-ELSE rule
    prompts = generator.select_test_prompts(rule, 10)
    
    print(f"  Selected {len(prompts)} prompts for IF-ELSE rule:")
    true_count = sum(1 for p in prompts if rule.conditions[0].evaluate(p))
    false_count = len(prompts) - true_count
    
    print(f"  Condition TRUE: {true_count}")
    print(f"  Condition FALSE: {false_count}")
    
    for i, prompt in enumerate(prompts[:5]):
        matches = rule.conditions[0].evaluate(prompt)
        print(f"    {i+1}. '{prompt}' → {matches}")
    
    return True


def main():
    """Run all offline tests."""
    print("\n🧪 Running offline tests...\n")
    
    tests = [
        ("Rule Generation", test_rule_generation),
        ("Condition Evaluation", test_condition_evaluation),
        ("Action Compliance", test_action_compliance),
        ("Prompt Selection", test_prompt_selection),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Setup is correct.")
        print("   Run 'python run.py' to start the experiment.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
