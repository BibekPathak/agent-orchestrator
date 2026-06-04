from .anthropic_llm import AnthropicLLM
from .base import LLM, LLMConfig
from .huggingface_llm import HuggingFaceLLM
from .openai_llm import OpenAILLM
from .openrouter_llm import OpenRouterLLM

__all__ = [
    "AnthropicLLM",
    "HuggingFaceLLM",
    "LLM",
    "LLMConfig",
    "OpenAILLM",
    "OpenRouterLLM",
]


def create_llm(config: LLMConfig | None = None) -> LLM:
    config = config or LLMConfig()
    if config.provider == "openai":
        return OpenAILLM(config)
    elif config.provider == "anthropic":
        return AnthropicLLM(config)
    elif config.provider == "huggingface":
        return HuggingFaceLLM(config)
    elif config.provider == "openrouter":
        return OpenRouterLLM(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
