"""
DeQoG LLM Client Module

Provides a unified interface for interacting with various LLM providers.
"""

import os
import json
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .logger import get_logger

logger = get_logger("llm_client")


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Optional[Any] = None


class BaseLLMClient(ABC):
    """
    Base class for LLM clients.
    
    Provides a unified interface for different LLM providers.
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 60,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        self._client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the underlying client."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            system_prompt: System prompt
            **kwargs: Additional arguments
            
        Returns:
            LLMResponse object
        """
        pass
    
    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Async version of generate.
        """
        pass
    
    def generate_with_retry(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate with retry logic.
        """
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                return self.generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    **kwargs
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM generation attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise RuntimeError(
            f"LLM generation failed after {self.retry_attempts} attempts: {last_error}"
        )


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client.
    
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    """
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout
            )
            logger.info(f"Initialized OpenAI client with model: {self.model_name}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are an expert programmer and software engineer."
            })
        
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            finish_reason=response.choices[0].finish_reason,
            raw_response=response
        )
    
    async def agenerate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Async generate using OpenAI API."""
        from openai import AsyncOpenAI
        
        async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are an expert programmer and software engineer."
            })
        
        messages.append({"role": "user", "content": prompt})
        
        response = await async_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            finish_reason=response.choices[0].finish_reason,
            raw_response=response
        )


class AnthropicClient(BaseLLMClient):
    """
    Anthropic API client.
    
    Supports Claude models.
    """
    
    def _initialize_client(self):
        """Initialize the Anthropic client."""
        try:
            from anthropic import Anthropic
            
            self._client = Anthropic(api_key=self.api_key)
            logger.info(f"Initialized Anthropic client with model: {self.model_name}")
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        system = system_prompt or "You are an expert programmer and software engineer."
        
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.temperature,
            **kwargs
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            finish_reason=response.stop_reason,
            raw_response=response
        )
    
    async def agenerate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Async generate using Anthropic API."""
        from anthropic import AsyncAnthropic
        
        async_client = AsyncAnthropic(api_key=self.api_key)
        system = system_prompt or "You are an expert programmer and software engineer."
        
        response = await async_client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.temperature,
            **kwargs
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            finish_reason=response.stop_reason,
            raw_response=response
        )


class LLMClientFactory:
    """
    Factory for creating LLM clients.
    """
    
    _providers = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
    }
    
    @classmethod
    def create(
        cls,
        provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLMClient:
        """
        Create an LLM client.
        
        Args:
            provider: Provider name ("openai", "anthropic")
            model_name: Model name
            api_key: API key (falls back to environment variable)
            **kwargs: Additional arguments passed to client
            
        Returns:
            LLM client instance
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(cls._providers.keys())}")
        
        # Get API key from environment if not provided
        if api_key is None:
            env_vars = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
            }
            api_key = os.environ.get(env_vars.get(provider, ""))
        
        if not api_key:
            raise ValueError(f"API key not provided for {provider}")
        
        # Set default model if not provided
        if model_name is None:
            default_models = {
                "openai": "gpt-4",
                "anthropic": "claude-3-opus-20240229",
            }
            model_name = default_models.get(provider, "gpt-4")
        
        return cls._providers[provider](
            model_name=model_name,
            api_key=api_key,
            **kwargs
        )
    
    @classmethod
    def register_provider(cls, name: str, client_class: type):
        """
        Register a custom LLM provider.
        
        Args:
            name: Provider name
            client_class: Client class (must inherit from BaseLLMClient)
        """
        if not issubclass(client_class, BaseLLMClient):
            raise TypeError("Client class must inherit from BaseLLMClient")
        cls._providers[name.lower()] = client_class


# Convenience function
def LLMClient(
    provider: str = "openai",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """
    Create an LLM client (convenience function).
    
    Args:
        provider: Provider name
        model_name: Model name
        api_key: API key
        **kwargs: Additional arguments
        
    Returns:
        LLM client instance
    """
    return LLMClientFactory.create(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )

