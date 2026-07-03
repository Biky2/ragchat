"""Unified async LLM interface: Ollama primary, HuggingFace Inference API fallback."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

_ollama_available: bool | None = None
_hf_available: bool | None = None


async def check_ollama_health() -> bool:
    """Return True if Ollama is reachable and the configured model is listed."""
    global _ollama_available
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            if response.status_code != 200:
                _ollama_available = False
                return False
            data = response.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            model_base = settings.llm_model.split(":")[0]
            _ollama_available = model_base in models or any(model_base in m for m in models)
            if not _ollama_available:
                _ollama_available = True
            return _ollama_available
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)
        _ollama_available = False
        return False


async def check_huggingface_health() -> bool:
    """Return True if HuggingFace API key is configured."""
    global _hf_available
    settings = get_settings()
    _hf_available = bool(settings.huggingface_api_key.strip())
    return _hf_available


async def _call_ollama(prompt: str, system_prompt: str) -> str:
    settings = get_settings()
    payload = {
        "model": settings.llm_model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 2048},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()


async def _stream_ollama(prompt: str, system_prompt: str) -> AsyncIterator[str]:
    settings = get_settings()
    payload = {
        "model": settings.llm_model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {"temperature": 0.2, "num_predict": 2048},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    match = re.search(r'"response"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', line)
                    if match:
                        yield _unescape_json_string(match.group(1))
                    continue
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break


# async def _call_huggingface(prompt: str, system_prompt: str) -> str:
#     settings = get_settings()
#     full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
#     headers = {"Authorization": f"Bearer {settings.huggingface_api_key}"}
#     payload = {
#         "inputs": full_prompt,
#         "parameters": {
#             "max_new_tokens": 2048,
#             "temperature": 0.2,
#             "return_full_text": False,
#         },
#     }
#     url = f"https://api-inference.huggingface.co/models/{settings.huggingface_model}"
#     async with httpx.AsyncClient(timeout=120.0) as client:
#         response = await client.post(url, headers=headers, json=payload)
#         response.raise_for_status()
#         data = response.json()
#         if isinstance(data, list) and data:
#             return data[0].get("generated_text", "").strip()
#         if isinstance(data, dict):
#             return data.get("generated_text", data.get("generated", "")).strip()
#         return ""



async def _call_huggingface(prompt: str, system_prompt: str) -> str:
    settings = get_settings()

    headers = {
        "Authorization": f"Bearer {settings.huggingface_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.huggingface_model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        response.raise_for_status()
        data = response.json()

        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
async def _stream_huggingface(prompt: str, system_prompt: str) -> AsyncIterator[str]:
    text = await _call_huggingface(prompt, system_prompt)
    words = text.split(" ")
    for i, word in enumerate(words):
        if i == 0:
            yield word
        else:
            yield " " + word


def _unescape_json_string(value: str) -> str:
    return (
        value.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


async def get_llm_response(prompt: str, system_prompt: str) -> str:
    """
    Generate a complete LLM response using Ollama, falling back to HuggingFace.
    """
    if await check_ollama_health():
        try:
            return await _call_ollama(prompt, system_prompt)
        except Exception as exc:
            logger.warning("Ollama generation failed, trying HuggingFace: %s", exc)

    if await check_huggingface_health():
        try:
            return await _call_huggingface(prompt, system_prompt)
        except Exception as exc:
            logger.error("HuggingFace generation failed: %s", exc)
            raise RuntimeError("All LLM providers unavailable") from exc

    raise RuntimeError(
        "No LLM provider available. Start Ollama or configure HUGGINGFACE_API_KEY."
    )


async def stream_llm_response(prompt: str, system_prompt: str) -> AsyncIterator[str]:
    """
    Stream LLM tokens using Ollama, falling back to HuggingFace word-by-word.
    """
    if await check_ollama_health():
        try:
            async for token in _stream_ollama(prompt, system_prompt):
                yield token
            return
        except Exception as exc:
            logger.warning("Ollama streaming failed, trying HuggingFace: %s", exc)

    if await check_huggingface_health():
        async for token in _stream_huggingface(prompt, system_prompt):
            yield token
        return

    raise RuntimeError(
        "No LLM provider available. Start Ollama or configure HUGGINGFACE_API_KEY."
    )


async def get_llm_status() -> dict[str, Any]:
    ollama_ok = await check_ollama_health()
    hf_ok = await check_huggingface_health()
    return {
        "ollama": "ok" if ollama_ok else "unavailable",
        "hf_fallback": hf_ok,
    }
