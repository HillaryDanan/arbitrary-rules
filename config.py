"""
Arbitrary Rule Following Experiment - Configuration
====================================================
ALL parameters are centralized here. NO hard-coded values elsewhere.

This experiment tests APH's core prediction:
- Construction (applying novel rules) vs Retrieval (pattern matching)
- Procedurally generated rules ensure no training distribution coverage
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class RuleGeneratorConfig:
    """Configuration for procedural rule generation.
    
    All rule components are defined here to ensure:
    1. No bias from manually written rules
    2. Reproducibility via random seed
    3. Adjustable complexity parameters
    """
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # === CONDITION COMPONENTS ===
    # Letters to use in conditions
    condition_letters: List[str] = field(default_factory=lambda: list("abcdefghijklmnopqrstuvwxyz"))
    
    # Vowels for vowel-based conditions
    vowels: List[str] = field(default_factory=lambda: list("aeiou"))
    
    # Common words for "contains word" conditions
    condition_words: List[str] = field(default_factory=lambda: [
        "the", "a", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "it", "this", "that", "what", "which"
    ])
    
    # Numeric thresholds for count-based conditions
    count_thresholds_min: int = 3
    count_thresholds_max: int = 10
    
    # Character position range
    char_position_min: int = 1
    char_position_max: int = 5
    
    # Word position range
    word_position_min: int = 1
    word_position_max: int = 4
    
    # === ACTION COMPONENTS ===
    # Words/phrases to prepend
    prepend_words: List[str] = field(default_factory=lambda: [
        "QUACK", "BEEP", "ZOOM", "PING", "BOOP", "WHAM", "ZAP", "FIZZ",
        "HONK", "BLIP", "DING", "WHOOSH", "SPLAT", "CLUNK", "BOING"
    ])
    
    # Words/phrases to append
    append_words: List[str] = field(default_factory=lambda: [
        "...maybe", "...indeed", "...perhaps", "- THE END", "- DONE",
        "!!", "???", "- VERIFIED", "- CONFIRMED", "- COMPLETE",
        "[END]", "(fin)", "~ concluded ~", ">>> OUTPUT", "<<< STOP"
    ])
    
    # Symbols to include
    symbols: List[str] = field(default_factory=lambda: [
        "★", "●", "▲", "■", "◆", "○", "△", "□", "◇", "※",
        "→", "←", "↑", "↓", "⊕", "⊗", "∴", "∵", "≈", "≠"
    ])
    
    # Word count limits for "exactly N words" actions
    word_limit_min: int = 3
    word_limit_max: int = 15
    
    # Symbol repeat counts
    symbol_repeat_min: int = 1
    symbol_repeat_max: int = 5
    
    # === TEST PROMPT COMPONENTS ===
    # Neutral test prompts (simple questions with clear answers)
    test_prompts: List[str] = field(default_factory=lambda: [
        "What color is the sky?",
        "How many days are in a week?",
        "What is the capital of France?",
        "Name a fruit that is red.",
        "What animal barks?",
        "How many legs does a spider have?",
        "What season comes after summer?",
        "What is two plus two?",
        "Name a planet in our solar system.",
        "What do you call a baby cat?",
        "What is the opposite of hot?",
        "How many months are in a year?",
        "What color are bananas?",
        "Name something that flies.",
        "What do fish live in?",
        "How many sides does a triangle have?",
        "What is the largest ocean?",
        "Name a vegetable that is green.",
        "What comes after Monday?",
        "How many hours in a day?",
        "What animal says moo?",
        "Name a type of weather.",
        "What is frozen water called?",
        "How many continents are there?",
        "What is the opposite of big?",
        "Name a musical instrument.",
        "What do birds build?",
        "How many wheels on a bicycle?",
        "What color is grass?",
        "Name something you wear on your feet.",
        "What is baby dog called?",
        "How many letters in the alphabet?",
        "What do bees make?",
        "Name a farm animal.",
        "What is the nearest star to Earth?",
        "How many fingers on one hand?",
        "What do you use to write?",
        "Name a breakfast food.",
        "What comes after winter?",
        "How many weeks in a year?",
    ])


@dataclass
class StatisticalConfig:
    """Statistical parameters for rigorous experimental design."""
    
    # Minimum trials per condition (Central Limit Theorem requirement)
    min_trials_per_condition: int = 30
    
    # Trials per complexity level
    trials_per_level: int = 30
    
    # Number of complexity levels to test
    num_complexity_levels: int = 5
    
    # For rule generation: rules per level
    rules_per_level: int = 6  # 6 rules × 5 prompts each = 30 trials per level
    
    # Prompts per rule
    prompts_per_rule: int = 5
    
    # Effect size thresholds (Cohen's d conventions)
    small_effect: float = 0.15
    medium_effect: float = 0.25
    large_effect: float = 0.40
    
    # Prediction thresholds
    # Level 1-2 should have accuracy above this
    easy_level_threshold: float = 0.70
    
    # Level 3-4 degradation: accuracy should drop by at least this much
    degradation_threshold: float = 0.20
    
    # Level 5 should have accuracy below this
    hard_level_threshold: float = 0.40
    
    # Sharp degradation detection: difference between adjacent levels
    sharp_degradation_threshold: float = 0.15
    
    # Report interpretation thresholds
    strong_support_threshold: float = 0.70
    weak_support_threshold: float = 0.40


@dataclass
class EvaluationConfig:
    """Configuration for response evaluation."""
    
    # Strictness levels for rule checking
    # strict: exact match required
    # lenient: partial credit for partial compliance
    evaluation_mode: str = "strict"
    
    # Case sensitivity for text matching
    case_sensitive: bool = False
    
    # Whitespace sensitivity
    whitespace_sensitive: bool = False
    
    # Partial credit weights (for lenient mode)
    partial_credit_prepend: float = 0.5
    partial_credit_append: float = 0.5
    partial_credit_case: float = 0.3
    partial_credit_count: float = 0.4


@dataclass
class ModelConfig:
    """Configuration for model API calls."""
    
    name: str = "Claude Sonnet 4"
    api_name: str = "claude-sonnet-4-20250514"
    provider: str = "anthropic"  # "anthropic", "openai", "google"
    max_tokens: int = 1024
    temperature: float = 0.0  # Deterministic for reproducibility


# Pre-configured models for easy testing
AVAILABLE_MODELS = {
    # Anthropic models
    "claude-sonnet-4": ModelConfig(
        name="Claude Sonnet 4",
        api_name="claude-sonnet-4-20250514",
        provider="anthropic"
    ),
    "claude-sonnet-3.5": ModelConfig(
        name="Claude 3.5 Sonnet", 
        api_name="claude-3-5-sonnet-20241022",
        provider="anthropic"
    ),
    "claude-haiku-3.5": ModelConfig(
        name="Claude 3.5 Haiku",
        api_name="claude-3-5-haiku-20241022", 
        provider="anthropic"
    ),
    # OpenAI models
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        api_name="gpt-4o",
        provider="openai"
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        api_name="gpt-4o-mini",
        provider="openai"
    ),
    "gpt-4-turbo": ModelConfig(
        name="GPT-4 Turbo",
        api_name="gpt-4-turbo",
        provider="openai"
    ),
    # Google models
    "gemini-1.5-pro": ModelConfig(
        name="Gemini 1.5 Pro",
        api_name="gemini-1.5-pro",
        provider="google"
    ),
    "gemini-1.5-flash": ModelConfig(
        name="Gemini 1.5 Flash",
        api_name="gemini-1.5-flash",
        provider="google"
    ),
    "gemini-2.0-flash": ModelConfig(
        name="Gemini 2.0 Flash",
        api_name="gemini-2.0-flash-exp",
        provider="google"
    ),
}

# Model comparison sets for multi-model experiments
MODEL_COMPARISON_SETS = {
    "frontier": ["claude-sonnet-4", "gpt-4o", "gemini-1.5-pro"],
    "fast": ["gpt-4o-mini", "gemini-1.5-flash", "gemini-2.0-flash"],
    "all": ["claude-sonnet-4", "gpt-4o", "gemini-1.5-pro", "gpt-4o-mini", "gemini-1.5-flash"]
}


@dataclass 
class ExperimentConfig:
    """Master configuration for the experiment."""
    
    # API key from environment
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    
    # Sub-configurations
    rules: RuleGeneratorConfig = field(default_factory=RuleGeneratorConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Output settings
    results_dir: Path = field(default_factory=lambda: Path("results"))
    save_raw_responses: bool = True
    
    # Random seed (master seed, propagates to all components)
    random_seed: int = 42
    
    # Timeout for API calls
    timeout_seconds: int = 120
    
    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Propagate random seed
        self.rules.random_seed = self.random_seed
        
        # Validate minimum trials
        total_trials_per_level = self.statistical.rules_per_level * self.statistical.prompts_per_rule
        if total_trials_per_level < self.statistical.min_trials_per_condition:
            print(f"⚠️  Trials per level ({total_trials_per_level}) below minimum "
                  f"({self.statistical.min_trials_per_condition}). Adjusting...")
            # Adjust rules_per_level to meet minimum
            self.statistical.rules_per_level = (
                self.statistical.min_trials_per_condition // self.statistical.prompts_per_rule + 1
            )
            print(f"   Set rules_per_level to {self.statistical.rules_per_level}")
        
        if not self.api_key:
            print("⚠️  ANTHROPIC_API_KEY not found. Set via environment variable.")


# Complexity level descriptions (for documentation and analysis)
COMPLEXITY_DESCRIPTIONS = {
    1: {
        "name": "Single Simple",
        "description": "Single condition → single action",
        "aph_mapping": "~3b (role-filler)",
        "example": "If prompt contains 'e', prepend 'BEEP'"
    },
    2: {
        "name": "Double Simple", 
        "description": "Two conditions (AND) → two actions",
        "aph_mapping": "~3b+ (extended role-filler)",
        "example": "If odd words AND contains 'a', ALL CAPS and append '!'"
    },
    3: {
        "name": "Conditional Branching",
        "description": "IF-ELSE structure with different actions",
        "aph_mapping": "~3c (recursive/conditional)",
        "example": "If contains 'the', prepend X. Otherwise, append Y."
    },
    4: {
        "name": "Priority Chain",
        "description": "IF-ELIF-ELIF-ELSE with priority ordering",
        "aph_mapping": "~3c+ (nested conditionals)",
        "example": "If A: do X. Elif B: do Y. Elif C: do Z. Else: do W."
    },
    5: {
        "name": "Self-Referential",
        "description": "Rule references properties of the output itself",
        "aph_mapping": "~3d (self-referential structure mapping)",
        "example": "Response must contain exactly N words, where N = vowel count in prompt"
    }
}
