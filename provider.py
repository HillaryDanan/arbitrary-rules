"""
API Provider
============
Handles communication with multiple LLM APIs.
Supports: Anthropic (Claude), OpenAI (GPT), Google (Gemini)
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Try to import API clients
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

from config import ModelConfig


@dataclass
class APIResponse:
    """Structured response from API call."""
    success: bool
    content: str
    latency_ms: float
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


class BaseProvider:
    """Base class for API providers."""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._lock = threading.Lock()
        self._call_count = 0
    
    def call(self, system_prompt: str, user_prompt: str) -> APIResponse:
        raise NotImplementedError
    
    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count
    
    def _increment_count(self):
        with self._lock:
            self._call_count += 1


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude API."""
    
    def __init__(self, api_key: str, model_config: ModelConfig):
        super().__init__(model_config)
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package required. Run: pip install anthropic")
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def call(self, system_prompt: str, user_prompt: str) -> APIResponse:
        start_time = time.time()
        
        try:
            response = self.client.messages.create(
                model=self.model_config.api_name,
                max_tokens=self.model_config.max_tokens,
                temperature=self.model_config.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.content[0].text if response.content else ""
            self._increment_count()
            
            return APIResponse(
                success=True,
                content=content,
                latency_ms=latency_ms,
                raw_response={"id": response.id, "model": response.model}
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return APIResponse(
                success=False,
                content="",
                latency_ms=latency_ms,
                error=str(e)
            )


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI GPT API."""
    
    def __init__(self, api_key: str, model_config: ModelConfig):
        super().__init__(model_config)
        if not HAS_OPENAI:
            raise ImportError("openai package required. Run: pip install openai")
        self.client = openai.OpenAI(api_key=api_key)
    
    def call(self, system_prompt: str, user_prompt: str) -> APIResponse:
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_config.api_name,
                max_tokens=self.model_config.max_tokens,
                temperature=self.model_config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content if response.choices else ""
            self._increment_count()
            
            return APIResponse(
                success=True,
                content=content,
                latency_ms=latency_ms,
                raw_response={"id": response.id, "model": response.model}
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return APIResponse(
                success=False,
                content="",
                latency_ms=latency_ms,
                error=str(e)
            )


class GoogleProvider(BaseProvider):
    """Provider for Google Gemini API."""
    
    def __init__(self, api_key: str, model_config: ModelConfig):
        super().__init__(model_config)
        if not HAS_GOOGLE:
            raise ImportError("google-generativeai package required. Run: pip install google-generativeai")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_config.api_name,
            generation_config={
                "max_output_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
            },
            system_instruction=None  # Set per-call
        )
    
    def call(self, system_prompt: str, user_prompt: str) -> APIResponse:
        start_time = time.time()
        
        try:
            # Gemini handles system prompt differently
            model = genai.GenerativeModel(
                model_name=self.model_config.api_name,
                generation_config={
                    "max_output_tokens": self.model_config.max_tokens,
                    "temperature": self.model_config.temperature,
                },
                system_instruction=system_prompt
            )
            
            response = model.generate_content(user_prompt)
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.text if response.text else ""
            self._increment_count()
            
            return APIResponse(
                success=True,
                content=content,
                latency_ms=latency_ms,
                raw_response={"model": self.model_config.api_name}
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return APIResponse(
                success=False,
                content="",
                latency_ms=latency_ms,
                error=str(e)
            )


def create_provider(model_config: ModelConfig) -> BaseProvider:
    """Factory function to create the appropriate provider."""
    provider_type = model_config.provider.lower()
    
    if provider_type == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return AnthropicProvider(api_key, model_config)
    
    elif provider_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIProvider(api_key, model_config)
    
    elif provider_type == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        return GoogleProvider(api_key, model_config)
    
    else:
        raise ValueError(f"Unknown provider: {provider_type}")


# Keep old name for backwards compatibility
ClaudeProvider = AnthropicProvider


class ParallelExecutor:
    """Execute multiple API calls in parallel."""
    
    def __init__(self, provider: BaseProvider, max_workers: int = 5):
        self.provider = provider
        self.max_workers = max_workers
    
    def execute_batch(
        self, 
        tasks: List[Dict[str, str]],
        progress_callback: Optional[callable] = None
    ) -> List[APIResponse]:
        """
        Execute a batch of tasks in parallel.
        """
        results = {}
        completed = 0
        total = len(tasks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self.provider.call,
                    task["system_prompt"],
                    task["user_prompt"]
                ): task
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_id = task.get("task_id", str(id(task)))
                
                try:
                    response = future.result()
                    results[task_id] = response
                except Exception as e:
                    results[task_id] = APIResponse(
                        success=False,
                        content="",
                        latency_ms=0,
                        error=str(e)
                    )
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        
        return [results.get(task.get("task_id", str(id(task)))) for task in tasks]


class MockProvider(BaseProvider):
    """Mock provider for testing without API calls."""
    
    def __init__(self):
        super().__init__(ModelConfig(name="Mock", api_name="mock", provider="mock"))
    
    def call(self, system_prompt: str, user_prompt: str) -> APIResponse:
        self._increment_count()
        mock_content = f"This is a mock response to: {user_prompt[:50]}..."
        
        return APIResponse(
            success=True,
            content=mock_content,
            latency_ms=100.0,
            raw_response={"mock": True}
        )
