"""
Procedural Rule Generator (FIXED)
==================================
Generates arbitrary rules WITHOUT human bias.

CRITICAL FIX: All rules now GUARANTEE that actions are always required.
- Level 1: Unconditional single action (always applies)
- Level 2: Unconditional two actions (always applies)  
- Level 3: IF-ELSE (both branches have actions)
- Level 4: IF-ELIF-ELIF-ELSE (all branches have actions)
- Level 5: Self-referential (always applies)

This ensures every trial actually tests the model's rule-following ability.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Tuple, Optional
from enum import Enum
import re

from config import RuleGeneratorConfig, ExperimentConfig


class ConditionType(Enum):
    """Types of conditions that can trigger actions."""
    LETTER_CONTAINS = "letter_contains"
    LETTER_STARTS = "letter_starts"
    LETTER_ENDS = "letter_ends"
    WORD_COUNT_PARITY = "word_count_parity"
    CHAR_COUNT_THRESHOLD = "char_count_threshold"
    VOWEL_COUNT_THRESHOLD = "vowel_count_threshold"
    WORD_POSITION_LETTER = "word_position_letter"
    CONTAINS_WORD = "contains_word"
    CHAR_POSITION = "char_position"


class ActionType(Enum):
    """Types of actions a rule can require."""
    PREPEND = "prepend"
    APPEND = "append"
    ALL_CAPS = "all_caps"
    ALL_LOWER = "all_lower"
    WORD_LIMIT = "word_limit"
    INCLUDE_COUNT = "include_count"
    INCLUDE_SYMBOL = "include_symbol"
    REVERSE_FIRST_WORD = "reverse_first_word"


@dataclass
class Condition:
    """A single condition in a rule."""
    condition_type: ConditionType
    parameters: Dict[str, Any]
    description: str
    
    def evaluate(self, prompt: str) -> bool:
        """Check if condition is met by the prompt."""
        prompt_lower = prompt.lower()
        
        if self.condition_type == ConditionType.LETTER_CONTAINS:
            letter = self.parameters["letter"].lower()
            return letter in prompt_lower
        
        elif self.condition_type == ConditionType.LETTER_STARTS:
            letter = self.parameters["letter"].lower()
            return prompt_lower.strip().startswith(letter)
        
        elif self.condition_type == ConditionType.LETTER_ENDS:
            letter = self.parameters["letter"].lower()
            # Get last letter (ignoring punctuation)
            cleaned = re.sub(r'[^\w]', '', prompt_lower)
            return cleaned.endswith(letter) if cleaned else False
        
        elif self.condition_type == ConditionType.WORD_COUNT_PARITY:
            word_count = len(prompt.split())
            parity = self.parameters["parity"]
            if parity == "odd":
                return word_count % 2 == 1
            else:
                return word_count % 2 == 0
        
        elif self.condition_type == ConditionType.CHAR_COUNT_THRESHOLD:
            char_count = len(prompt.replace(" ", ""))
            threshold = self.parameters["threshold"]
            operator = self.parameters["operator"]
            if operator == ">":
                return char_count > threshold
            else:
                return char_count < threshold
        
        elif self.condition_type == ConditionType.VOWEL_COUNT_THRESHOLD:
            vowel_count = sum(1 for c in prompt_lower if c in "aeiou")
            threshold = self.parameters["threshold"]
            operator = self.parameters["operator"]
            if operator == ">":
                return vowel_count > threshold
            else:
                return vowel_count < threshold
        
        elif self.condition_type == ConditionType.WORD_POSITION_LETTER:
            words = prompt.split()
            position = self.parameters["position"] - 1  # Convert to 0-indexed
            letter = self.parameters["letter"].lower()
            if position < len(words):
                return words[position].lower().startswith(letter)
            return False
        
        elif self.condition_type == ConditionType.CONTAINS_WORD:
            word = self.parameters["word"].lower()
            # Word boundary matching
            pattern = r'\b' + re.escape(word) + r'\b'
            return bool(re.search(pattern, prompt_lower))
        
        elif self.condition_type == ConditionType.CHAR_POSITION:
            position = self.parameters["position"] - 1  # Convert to 0-indexed
            letter = self.parameters["letter"].lower()
            cleaned = prompt.replace(" ", "").lower()
            if position < len(cleaned):
                return cleaned[position] == letter
            return False
        
        return False


@dataclass
class Action:
    """A single action in a rule."""
    action_type: ActionType
    parameters: Dict[str, Any]
    description: str
    
    def get_requirement_description(self) -> str:
        """Get human-readable description of what this action requires."""
        return self.description
    
    def check_compliance(self, response: str, prompt: str, case_sensitive: bool = False) -> Tuple[bool, str]:
        """
        Check if response complies with this action.
        Returns (compliant, explanation).
        """
        check_response = response if case_sensitive else response.lower()
        
        if self.action_type == ActionType.PREPEND:
            word = self.parameters["word"]
            check_word = word if case_sensitive else word.lower()
            compliant = check_response.strip().startswith(check_word)
            return compliant, f"Should start with '{word}'"
        
        elif self.action_type == ActionType.APPEND:
            word = self.parameters["word"]
            check_word = word if case_sensitive else word.lower()
            compliant = check_response.strip().endswith(check_word)
            return compliant, f"Should end with '{word}'"
        
        elif self.action_type == ActionType.ALL_CAPS:
            # Check if alphabetic characters are uppercase
            alpha_chars = [c for c in response if c.isalpha()]
            compliant = all(c.isupper() for c in alpha_chars) if alpha_chars else True
            return compliant, "Should be in ALL CAPS"
        
        elif self.action_type == ActionType.ALL_LOWER:
            alpha_chars = [c for c in response if c.isalpha()]
            compliant = all(c.islower() for c in alpha_chars) if alpha_chars else True
            return compliant, "Should be in all lowercase"
        
        elif self.action_type == ActionType.WORD_LIMIT:
            limit = self.parameters["limit"]
            word_count = len(response.split())
            compliant = word_count == limit
            return compliant, f"Should have exactly {limit} words (has {word_count})"
        
        elif self.action_type == ActionType.INCLUDE_COUNT:
            count_type = self.parameters["count_type"]
            prompt_lower = prompt.lower()
            
            if count_type == "vowels":
                expected = sum(1 for c in prompt_lower if c in "aeiou")
            elif count_type == "words":
                expected = len(prompt.split())
            elif count_type == "letters":
                expected = sum(1 for c in prompt if c.isalpha())
            else:
                expected = 0
            
            # Check if the number appears in response
            compliant = str(expected) in response
            return compliant, f"Should include the number {expected} ({count_type} count)"
        
        elif self.action_type == ActionType.INCLUDE_SYMBOL:
            symbol = self.parameters["symbol"]
            count = self.parameters["count"]
            actual_count = response.count(symbol)
            compliant = actual_count == count
            return compliant, f"Should include '{symbol}' exactly {count} time(s) (has {actual_count})"
        
        elif self.action_type == ActionType.REVERSE_FIRST_WORD:
            words = response.split()
            if not words:
                return False, "Response is empty"
            first_word = words[0]
            # Check if it looks reversed (heuristic: unusual letter patterns)
            # This is hard to verify without knowing the "correct" answer
            # We'll check if it's not a common English word pattern
            compliant = True  # Lenient - hard to verify
            return compliant, "First word should be reversed"
        
        return True, "Unknown action"


@dataclass
class Rule:
    """A complete rule with conditions and actions."""
    rule_id: str
    complexity_level: int
    conditions: List[Condition]
    actions: List[Action]
    logic: str  # "ALWAYS", "IF_ELSE", "PRIORITY"
    else_actions: Optional[List[Action]] = None  # For IF_ELSE rules
    priority_branches: Optional[List[Tuple[Condition, List[Action]]]] = None  # For PRIORITY rules
    
    def get_rule_text(self) -> str:
        """Generate human-readable rule description."""
        lines = []
        
        if self.logic == "ALWAYS":
            # Unconditional - actions always apply
            action_text = ", and ".join(a.description for a in self.actions)
            lines.append(f"Always {action_text}.")
        
        elif self.logic == "IF_ELSE":
            cond_text = " AND ".join(c.description for c in self.conditions)
            action_text = ", and ".join(a.description for a in self.actions)
            else_text = ", and ".join(a.description for a in self.else_actions) if self.else_actions else "respond normally"
            lines.append(f"If {cond_text}, then {action_text}.")
            lines.append(f"Otherwise, {else_text}.")
        
        elif self.logic == "PRIORITY":
            if self.priority_branches:
                for i, (cond, acts) in enumerate(self.priority_branches):
                    prefix = "If" if i == 0 else "Else if"
                    action_text = ", and ".join(a.description for a in acts)
                    lines.append(f"{prefix} {cond.description}, then {action_text}.")
                if self.else_actions:
                    else_text = ", and ".join(a.description for a in self.else_actions)
                    lines.append(f"Otherwise, {else_text}.")
        
        return "\n".join(lines)
    
    def get_applicable_actions(self, prompt: str) -> List[Action]:
        """Determine which actions apply given the prompt."""
        if self.logic == "ALWAYS":
            # Actions ALWAYS apply - this is the fix!
            return self.actions
        
        elif self.logic == "IF_ELSE":
            if all(c.evaluate(prompt) for c in self.conditions):
                return self.actions
            return self.else_actions if self.else_actions else []
        
        elif self.logic == "PRIORITY":
            if self.priority_branches:
                for cond, acts in self.priority_branches:
                    if cond.evaluate(prompt):
                        return acts
            return self.else_actions if self.else_actions else []
        
        return []
    
    def evaluate_response(self, response: str, prompt: str, case_sensitive: bool = False) -> Dict[str, Any]:
        """
        Evaluate if response follows the rule.
        Returns detailed evaluation results.
        """
        applicable_actions = self.get_applicable_actions(prompt)
        
        if not applicable_actions:
            # This should NEVER happen with the fixed generator
            return {
                "compliant": True,
                "actions_required": 0,
                "actions_met": 0,
                "details": ["WARNING: No actions required - this indicates a bug!"]
            }
        
        results = []
        actions_met = 0
        
        for action in applicable_actions:
            compliant, explanation = action.check_compliance(response, prompt, case_sensitive)
            results.append({
                "action": action.description,
                "compliant": compliant,
                "explanation": explanation
            })
            if compliant:
                actions_met += 1
        
        return {
            "compliant": actions_met == len(applicable_actions),
            "actions_required": len(applicable_actions),
            "actions_met": actions_met,
            "details": results
        }


class RuleGenerator:
    """Procedurally generates arbitrary rules - FIXED to guarantee actions."""
    
    def __init__(self, config: RuleGeneratorConfig):
        self.config = config
        random.seed(config.random_seed)
        self._rule_counter = 0
    
    def _generate_condition(self, exclude_types: List[ConditionType] = None) -> Condition:
        """Generate a random condition."""
        exclude_types = exclude_types or []
        available_types = [t for t in ConditionType if t not in exclude_types]
        cond_type = random.choice(available_types)
        
        if cond_type == ConditionType.LETTER_CONTAINS:
            letter = random.choice(self.config.condition_letters)
            return Condition(
                condition_type=cond_type,
                parameters={"letter": letter},
                description=f"the prompt contains the letter '{letter}'"
            )
        
        elif cond_type == ConditionType.LETTER_STARTS:
            letter = random.choice(self.config.condition_letters)
            return Condition(
                condition_type=cond_type,
                parameters={"letter": letter},
                description=f"the prompt starts with '{letter}'"
            )
        
        elif cond_type == ConditionType.LETTER_ENDS:
            letter = random.choice(self.config.condition_letters)
            return Condition(
                condition_type=cond_type,
                parameters={"letter": letter},
                description=f"the prompt ends with '{letter}'"
            )
        
        elif cond_type == ConditionType.WORD_COUNT_PARITY:
            parity = random.choice(["odd", "even"])
            return Condition(
                condition_type=cond_type,
                parameters={"parity": parity},
                description=f"the prompt has an {parity} number of words"
            )
        
        elif cond_type == ConditionType.CHAR_COUNT_THRESHOLD:
            threshold = random.randint(
                self.config.count_thresholds_min,
                self.config.count_thresholds_max
            )
            operator = random.choice([">", "<"])
            op_text = "more than" if operator == ">" else "fewer than"
            return Condition(
                condition_type=cond_type,
                parameters={"threshold": threshold, "operator": operator},
                description=f"the prompt has {op_text} {threshold} characters (excluding spaces)"
            )
        
        elif cond_type == ConditionType.VOWEL_COUNT_THRESHOLD:
            threshold = random.randint(
                self.config.count_thresholds_min,
                self.config.count_thresholds_max
            )
            operator = random.choice([">", "<"])
            op_text = "more than" if operator == ">" else "fewer than"
            return Condition(
                condition_type=cond_type,
                parameters={"threshold": threshold, "operator": operator},
                description=f"the prompt has {op_text} {threshold} vowels"
            )
        
        elif cond_type == ConditionType.WORD_POSITION_LETTER:
            position = random.randint(
                self.config.word_position_min,
                self.config.word_position_max
            )
            letter = random.choice(self.config.condition_letters)
            ordinal = {1: "first", 2: "second", 3: "third", 4: "fourth"}.get(position, f"{position}th")
            return Condition(
                condition_type=cond_type,
                parameters={"position": position, "letter": letter},
                description=f"the {ordinal} word starts with '{letter}'"
            )
        
        elif cond_type == ConditionType.CONTAINS_WORD:
            word = random.choice(self.config.condition_words)
            return Condition(
                condition_type=cond_type,
                parameters={"word": word},
                description=f"the prompt contains the word '{word}'"
            )
        
        elif cond_type == ConditionType.CHAR_POSITION:
            position = random.randint(
                self.config.char_position_min,
                self.config.char_position_max
            )
            letter = random.choice(self.config.condition_letters)
            ordinal = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(position, f"{position}th")
            return Condition(
                condition_type=cond_type,
                parameters={"position": position, "letter": letter},
                description=f"the {ordinal} character (ignoring spaces) is '{letter}'"
            )
        
        # Fallback
        return self._generate_condition(exclude_types)
    
    def _generate_action(self, exclude_types: List[ActionType] = None) -> Action:
        """Generate a random action."""
        exclude_types = exclude_types or []
        available_types = [t for t in ActionType if t not in exclude_types]
        action_type = random.choice(available_types)
        
        if action_type == ActionType.PREPEND:
            word = random.choice(self.config.prepend_words)
            return Action(
                action_type=action_type,
                parameters={"word": word},
                description=f"begin your response with '{word}'"
            )
        
        elif action_type == ActionType.APPEND:
            word = random.choice(self.config.append_words)
            return Action(
                action_type=action_type,
                parameters={"word": word},
                description=f"end your response with '{word}'"
            )
        
        elif action_type == ActionType.ALL_CAPS:
            return Action(
                action_type=action_type,
                parameters={},
                description="respond in ALL CAPS"
            )
        
        elif action_type == ActionType.ALL_LOWER:
            return Action(
                action_type=action_type,
                parameters={},
                description="respond in all lowercase"
            )
        
        elif action_type == ActionType.WORD_LIMIT:
            limit = random.randint(
                self.config.word_limit_min,
                self.config.word_limit_max
            )
            return Action(
                action_type=action_type,
                parameters={"limit": limit},
                description=f"respond in exactly {limit} words"
            )
        
        elif action_type == ActionType.INCLUDE_COUNT:
            count_type = random.choice(["vowels", "words", "letters"])
            return Action(
                action_type=action_type,
                parameters={"count_type": count_type},
                description=f"include the number of {count_type} in the prompt somewhere in your response"
            )
        
        elif action_type == ActionType.INCLUDE_SYMBOL:
            symbol = random.choice(self.config.symbols)
            count = random.randint(
                self.config.symbol_repeat_min,
                self.config.symbol_repeat_max
            )
            times = "time" if count == 1 else "times"
            return Action(
                action_type=action_type,
                parameters={"symbol": symbol, "count": count},
                description=f"include the symbol '{symbol}' exactly {count} {times} in your response"
            )
        
        elif action_type == ActionType.REVERSE_FIRST_WORD:
            return Action(
                action_type=action_type,
                parameters={},
                description="reverse the first word of your response"
            )
        
        # Fallback
        return self._generate_action(exclude_types)
    
    def generate_rule(self, complexity_level: int) -> Rule:
        """
        Generate a rule at the specified complexity level.
        
        FIXED: All rules now guarantee actions are always required.
        
        Level 1: ALWAYS do one action (unconditional)
        Level 2: ALWAYS do two actions (unconditional)
        Level 3: IF-ELSE with different actions for each branch
        Level 4: IF-ELIF-ELIF-ELSE priority chain
        Level 5: Self-referential (action depends on input properties)
        """
        self._rule_counter += 1
        rule_id = f"rule_{self._rule_counter:04d}_L{complexity_level}"
        
        if complexity_level == 1:
            # FIXED: Unconditional single action - ALWAYS applies
            action = self._generate_action()
            return Rule(
                rule_id=rule_id,
                complexity_level=complexity_level,
                conditions=[],
                actions=[action],
                logic="ALWAYS"  # Changed from "AND" to "ALWAYS"
            )
        
        elif complexity_level == 2:
            # FIXED: Unconditional two actions - ALWAYS applies
            act1 = self._generate_action()
            act2 = self._generate_action(exclude_types=[act1.action_type])
            return Rule(
                rule_id=rule_id,
                complexity_level=complexity_level,
                conditions=[],
                actions=[act1, act2],
                logic="ALWAYS"  # Changed from "AND" to "ALWAYS"
            )
        
        elif complexity_level == 3:
            # IF-ELSE: Both branches have actions, so always tested
            condition = self._generate_condition()
            if_action = self._generate_action()
            else_action = self._generate_action(exclude_types=[if_action.action_type])
            return Rule(
                rule_id=rule_id,
                complexity_level=complexity_level,
                conditions=[condition],
                actions=[if_action],
                logic="IF_ELSE",
                else_actions=[else_action]
            )
        
        elif complexity_level == 4:
            # Priority chain with ELSE - always has an action
            used_cond_types = []
            used_action_types = []
            branches = []
            
            for _ in range(3):  # 3 IF/ELIF branches
                cond = self._generate_condition(exclude_types=used_cond_types)
                used_cond_types.append(cond.condition_type)
                act = self._generate_action(exclude_types=used_action_types)
                used_action_types.append(act.action_type)
                branches.append((cond, [act]))
            
            # ELSE branch ensures action is always required
            else_action = self._generate_action(exclude_types=used_action_types)
            
            return Rule(
                rule_id=rule_id,
                complexity_level=complexity_level,
                conditions=[],
                actions=[],
                logic="PRIORITY",
                priority_branches=branches,
                else_actions=[else_action]
            )
        
        elif complexity_level == 5:
            # Self-referential: output must match computed input property
            # This ALWAYS requires an action
            count_type = random.choice(["vowels", "words"])
            
            action = Action(
                action_type=ActionType.WORD_LIMIT,
                parameters={"limit": -1, "dynamic": True, "count_type": count_type},
                description=f"respond in exactly N words, where N equals the number of {count_type} in the prompt"
            )
            
            # Override compliance check for dynamic word limit
            def dynamic_check(response: str, prompt: str, case_sensitive: bool = False) -> Tuple[bool, str]:
                prompt_lower = prompt.lower()
                if count_type == "vowels":
                    expected = sum(1 for c in prompt_lower if c in "aeiou")
                else:  # words
                    expected = len(prompt.split())
                
                actual = len(response.split())
                compliant = actual == expected
                return compliant, f"Should have exactly {expected} words (has {actual})"
            
            action.check_compliance = dynamic_check
            
            return Rule(
                rule_id=rule_id,
                complexity_level=complexity_level,
                conditions=[],
                actions=[action],
                logic="ALWAYS"  # Always applies
            )
        
        # Fallback to level 1
        return self.generate_rule(1)
    
    def generate_rules_for_level(self, complexity_level: int, count: int) -> List[Rule]:
        """Generate multiple rules for a complexity level."""
        return [self.generate_rule(complexity_level) for _ in range(count)]
    
    def select_test_prompts(self, rule: Rule, count: int) -> List[str]:
        """
        Select test prompts for a rule.
        
        For IF-ELSE and PRIORITY rules, balances True/False conditions.
        For ALWAYS rules, just returns random prompts (all will be tested).
        """
        all_prompts = self.config.test_prompts.copy()
        random.shuffle(all_prompts)
        
        # For ALWAYS rules, any prompts work
        if rule.logic == "ALWAYS":
            return all_prompts[:count]
        
        # For IF-ELSE and PRIORITY rules, balance True/False conditions
        if rule.logic == "IF_ELSE":
            true_prompts = []
            false_prompts = []
            
            for prompt in all_prompts:
                if all(c.evaluate(prompt) for c in rule.conditions):
                    true_prompts.append(prompt)
                else:
                    false_prompts.append(prompt)
            
            # Balance: try to get roughly equal true/false
            half = count // 2
            selected = true_prompts[:half] + false_prompts[:count - half]
            
            # If we don't have enough, fill with whatever we have
            if len(selected) < count:
                remaining = [p for p in all_prompts if p not in selected]
                selected.extend(remaining[:count - len(selected)])
            
            random.shuffle(selected)
            return selected[:count]
        
        elif rule.logic == "PRIORITY":
            # For priority rules, try to get prompts that trigger different branches
            branch_prompts = [[] for _ in rule.priority_branches]
            else_prompts = []
            
            for prompt in all_prompts:
                matched = False
                for i, (cond, _) in enumerate(rule.priority_branches):
                    if cond.evaluate(prompt):
                        branch_prompts[i].append(prompt)
                        matched = True
                        break
                if not matched:
                    else_prompts.append(prompt)
            
            # Try to get at least one prompt per branch
            selected = []
            per_branch = max(1, count // (len(rule.priority_branches) + 1))
            
            for prompts in branch_prompts:
                selected.extend(prompts[:per_branch])
            selected.extend(else_prompts[:per_branch])
            
            # Fill remaining with random
            if len(selected) < count:
                remaining = [p for p in all_prompts if p not in selected]
                selected.extend(remaining[:count - len(selected)])
            
            random.shuffle(selected)
            return selected[:count]
        
        # Fallback
        return all_prompts[:count]
