"""Model-family behavior shared by the benchmark clients."""

from __future__ import annotations

from typing import Any

MODEL_FAMILIES = ("auto", "qwen", "internvl", "llava-next", "generic")


def resolve_model_family(model: str, requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in MODEL_FAMILIES:
            raise ValueError(f"unsupported model family: {requested}")
        return requested
    if model.startswith("Qwen/Qwen3-VL-"):
        return "qwen"
    if model == "OpenGVLab/InternVL3-8B-hf":
        return "internvl"
    if model == "llava-hf/llava-v1.6-mistral-7b-hf":
        return "llava-next"
    return "generic"


def chat_template_extra_body(
    model_family: str, enable_thinking: bool
) -> dict[str, Any] | None:
    if enable_thinking and model_family != "qwen":
        raise ValueError("thinking mode is supported only by the Qwen model profile")
    if model_family == "qwen" and not enable_thinking:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def adapt_messages(messages: list[dict[str, Any]], model_family: str) -> list[dict[str, Any]]:
    """Make official messages compatible without changing media-part order."""

    if model_family != "llava-next" or not messages or messages[0].get("role") != "system":
        return messages

    system_text = str(messages[0].get("content") or "").strip()
    adapted = [{**message} for message in messages[1:]]
    user_index = next(
        (index for index, message in enumerate(adapted) if message.get("role") == "user"),
        None,
    )
    if user_index is None:
        raise ValueError("cannot fold the system prompt without a user message")

    content = adapted[user_index].get("content")
    prefix = f"{system_text}\n\n" if system_text else ""
    if isinstance(content, list):
        adapted[user_index]["content"] = [
            {"type": "text", "text": prefix},
            *content,
        ]
    else:
        adapted[user_index]["content"] = f"{prefix}{content or ''}"
    return adapted
