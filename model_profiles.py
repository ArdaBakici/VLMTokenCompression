"""Model-family behavior shared by the benchmark clients."""

from __future__ import annotations

from typing import Any

MODEL_FAMILIES = ("auto", "qwen", "internvl", "llava-next", "generic")

# `llava-hf/llava-v1.6-mistral-7b-hf` AnyRes geometry. These reproduce the
# Transformers processor exactly, so a prompt's visual length can be computed
# from image dimensions alone instead of being bounded by a worst-case guess.
LLAVA_NEXT_GRID_PINPOINTS = (
    (336, 672),
    (672, 336),
    (672, 672),
    (1008, 336),
    (336, 1008),
)
LLAVA_NEXT_PROCESSED_SIZE = 336
LLAVA_NEXT_PATCH_SIZE = 14


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


def select_best_resolution(
    original_size: tuple[int, int],
    possible_resolutions: tuple[tuple[int, int], ...] = LLAVA_NEXT_GRID_PINPOINTS,
) -> tuple[int, int]:
    """Port of `transformers.image_processing_utils.select_best_resolution`."""

    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")
    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width = int(original_width * scale)
        downscaled_height = int(original_height * scale)
        effective = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted = (width * height) - effective
        if effective > max_effective_resolution or (
            effective == max_effective_resolution and wasted < min_wasted_resolution
        ):
            max_effective_resolution = effective
            min_wasted_resolution = wasted
            best_fit = (height, width)
    if best_fit is None:
        raise ValueError("no candidate resolutions were provided")
    return best_fit


def llava_next_image_tokens(height: int, width: int) -> int:
    """Prompt tokens one image expands into, matching `LlavaNextProcessor`.

    Mirrors `_get_number_of_features`: unpadded AnyRes crop features, one
    newline per retained crop row, the base image, and the `default`
    vision-feature strategy that drops the class token.
    """

    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {height}x{width}")

    best_height, best_width = select_best_resolution((height, width))
    scale_height = best_height // LLAVA_NEXT_PROCESSED_SIZE
    scale_width = best_width // LLAVA_NEXT_PROCESSED_SIZE
    patches = LLAVA_NEXT_PROCESSED_SIZE // LLAVA_NEXT_PATCH_SIZE

    current_height = patches * scale_height
    current_width = patches * scale_width
    if (width / height) > (current_width / current_height):
        new_height = int(round(height * (current_width / width), 7))
        padding = (current_height - new_height) // 2
        current_height -= padding * 2
    else:
        new_width = int(round(width * (current_height / height), 7))
        padding = (current_width - new_width) // 2
        current_width -= padding * 2

    unpadded_features = current_height * current_width
    newline_features = current_height
    base_features = patches * patches + 1
    return unpadded_features + newline_features + base_features - 1


def adapt_messages(
    messages: list[dict[str, Any]], model_family: str
) -> list[dict[str, Any]]:
    """Make official messages compatible without changing media-part order."""

    if (
        model_family != "llava-next"
        or not messages
        or messages[0].get("role") != "system"
    ):
        return messages

    system_text = str(messages[0].get("content") or "").strip()
    adapted = [{**message} for message in messages[1:]]
    user_index = next(
        (
            index
            for index, message in enumerate(adapted)
            if message.get("role") == "user"
        ),
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
