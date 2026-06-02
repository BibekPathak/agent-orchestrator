from .anthropic_llm import AnthropicLLM
from .base import LLM, LLMConfig
from .huggingface_llm import HuggingFaceLLM
from .openai_llm import OpenAILLM

__all__ = [
    "AnthropicLLM",
    "HuggingFaceLLM",
    "LLM",
    "LLMConfig",
    "OpenAILLM",
]


def create_llm(config: LLMConfig | None = None) -> LLM:
    config = config or LLMConfig()
    if config.provider == "openai":
        return OpenAILLM(config)
    elif config.provider == "anthropic":
        return AnthropicLLM(config)
    elif config.provider == "huggingface":
        return HuggingFaceLLM(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
