"""Shared DeepEval judge model configuration.

Ollama is the default so the evaluation suite works without API keys. Set
DEEPEVAL_JUDGE_PROVIDER=openai to use an OpenAI judge when a key is available.
"""

from __future__ import annotations

import os

from deepeval.models import GPTModel, OllamaModel
from deepeval.models import OllamaEmbeddingModel
from dotenv import load_dotenv

load_dotenv()


def get_judge_config(
    provider: str | None = None,
    model_name: str | None = None,
) -> dict[str, str]:
    """Return the provider settings shared by judges and runtime clients."""
    selected_provider = (
        provider or os.getenv("DEEPEVAL_JUDGE_PROVIDER", "ollama")
    ).strip().lower()

    if selected_provider not in {"ollama", "openai"}:
        raise ValueError(
            f"Unsupported DEEPEVAL_JUDGE_PROVIDER={selected_provider!r}. "
            "Choose 'ollama' or 'openai'."
        )

    default_model = "qwen2.5:3b" if selected_provider == "ollama" else "gpt-4o"
    return {
        "provider": selected_provider,
        "model": model_name or os.getenv("DEEPEVAL_JUDGE_MODEL", default_model),
        "ollama_base_url": os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ),
    }


def get_ollama_runtime_config(
    chat_model: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, str]:
    """Return local Ollama settings for application runtimes."""
    return {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "chat_model": chat_model or os.getenv(
            "OLLAMA_CHAT_MODEL", "qwen2.5:3b"
        ),
        "embedding_model": embedding_model or os.getenv(
            "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
        ),
    }


def get_rag_runtime_config(
    provider: str | None = None,
    chat_model: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, str]:
    """Return provider and model settings for the RAG runtime.

    Ollama is the default. ``RAG_PROVIDER=cloud`` keeps the original
    Anthropic chat model and OpenAI embedding model available.
    """
    selected_provider = (
        provider or os.getenv("RAG_PROVIDER", "ollama")
    ).strip().lower()
    if selected_provider not in {"ollama", "cloud"}:
        raise ValueError(
            f"Unsupported RAG_PROVIDER={selected_provider!r}. "
            "Choose 'ollama' or 'cloud'."
        )

    ollama_config = get_ollama_runtime_config(chat_model, embedding_model)
    return {
        "provider": selected_provider,
        "base_url": ollama_config["base_url"],
        "chat_model": ollama_config["chat_model"],
        "embedding_model": ollama_config["embedding_model"],
        "cloud_chat_model": os.getenv(
            "RAG_CLOUD_CHAT_MODEL", "claude-sonnet-4-6"
        ),
        "cloud_embedding_model": os.getenv(
            "RAG_CLOUD_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
    }


def get_judge_embedding_model(
    provider: str | None = None,
    model_name: str | None = None,
):
    """Build the configured DeepEval embedding model."""
    selected_provider = (
        provider or os.getenv("DEEPEVAL_JUDGE_PROVIDER", "ollama")
    ).strip().lower()
    if selected_provider == "ollama":
        runtime = get_ollama_runtime_config(embedding_model=model_name)
        return OllamaEmbeddingModel(
            model=runtime["embedding_model"],
            base_url=runtime["base_url"],
        )
    if selected_provider == "openai":
        from deepeval.models import OpenAIEmbeddingModel

        return OpenAIEmbeddingModel(
            model=model_name or os.getenv(
                "DEEPEVAL_JUDGE_EMBEDDING_MODEL", "text-embedding-3-small"
            )
        )
    raise ValueError(
        f"Unsupported DEEPEVAL_JUDGE_PROVIDER={selected_provider!r}. "
        "Choose 'ollama' or 'openai'."
    )


def get_judge_model(
    provider: str | None = None,
    model_name: str | None = None,
):
    """Build the configured DeepEval judge model.

    Configuration comes from arguments first, then environment variables:
    ``DEEPEVAL_JUDGE_PROVIDER`` (``ollama`` or ``openai``),
    ``DEEPEVAL_JUDGE_MODEL``, and ``OLLAMA_BASE_URL``.
    """
    config = get_judge_config(provider, model_name)

    if config["provider"] == "ollama":
        return OllamaModel(
            model=config["model"],
            base_url=config["ollama_base_url"],
            temperature=0,
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "DEEPEVAL_JUDGE_PROVIDER=openai requires OPENAI_API_KEY. "
            "Set the key or use DEEPEVAL_JUDGE_PROVIDER=ollama."
        )
    return GPTModel(model=config["model"])


# Stable imports used by evaluation files and runtime clients.
judge_config = get_judge_config()
judge_model = get_judge_model()
ollama_runtime_config = get_ollama_runtime_config()
rag_runtime_config = get_rag_runtime_config()
judge_embedding_model = get_judge_embedding_model()
