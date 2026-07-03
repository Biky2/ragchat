"""Builds RAG prompts from retrieved context chunks."""

from __future__ import annotations

from retrieval.hybrid import HybridResult

SYSTEM_PROMPT = (
    "You are a precise enterprise knowledge assistant. Answer the user's question "
    "using ONLY the provided context chunks. For every claim you make, cite the "
    "source using [Source N] notation where N is the chunk number. If the answer "
    "cannot be found in the context, say exactly: 'I could not find this information "
    "in the provided documents.' Never hallucinate or use outside knowledge."
)

FALLBACK_ANSWER = "I could not find this information in the provided documents."


def build_prompt(
    query: str,
    chunks: list[HybridResult],
    conversation_context: str = "",
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) tuple for LLM call.
    """
    if not chunks:
        return system_prompt, f"QUESTION: {query}\nANSWER:\n{FALLBACK_ANSWER}"

    context_lines: list[str] = ["CONTEXT:"]
    for i, chunk in enumerate(chunks, start=1):
        source_file = chunk.metadata.get("source_file", "unknown")
        context_lines.append(
            f"[Source {i}]: {chunk.text} (from: {source_file})"
        )

    parts: list[str] = context_lines
    if conversation_context.strip():
        parts.insert(0, f"PREVIOUS CONVERSATION:\n{conversation_context.strip()}")
    parts.append(f"QUESTION: {query}")
    parts.append("ANSWER:")

    user_prompt = "\n".join(parts)
    return system_prompt, user_prompt


def build_streaming_prompt(
    query: str,
    chunks: list[HybridResult],
    conversation_context: str = "",
) -> tuple[str, str]:
    return build_prompt(query, chunks, conversation_context)
