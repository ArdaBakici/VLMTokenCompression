"""Pinned official visual-token-compression reproduction profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionProfile:
    method: str
    repository: str
    commit: str
    adapter: str
    model: str
    model_revision: str | None
    defaults: dict[str, float | int]
    license: str


PROFILES = {
    "visionzip": CompressionProfile(
        method="visionzip",
        repository="https://github.com/JIA-Lab-research/VisionZip.git",
        commit="8f86b55c6f000eb033e6912538af2dd7dcb30502",
        adapter="native-llava",
        model="liuhaotian/llava-v1.5-7b",
        model_revision="4481d270cc22fd5c4d1bb5df129622006ccd9234",
        defaults={"dominant_tokens": 54, "contextual_tokens": 10},
        license="Apache-2.0",
    ),
    "hiprune": CompressionProfile(
        method="hiprune",
        repository="https://github.com/Danielement321/HiPrune.git",
        commit="82781005a7e72a6be9ede58fd77473efa72b5e4f",
        adapter="native-llava",
        model="liuhaotian/llava-v1.5-7b",
        model_revision="4481d270cc22fd5c4d1bb5df129622006ccd9234",
        defaults={"retained_tokens": 192, "alpha": 0.1, "object_layer": 9},
        license="Apache-2.0",
    ),
    "cdpruner": CompressionProfile(
        method="cdpruner",
        repository="https://github.com/Theia-4869/CDPruner.git",
        commit="9541616c40fcd5625de1cdb8ea6c33c129eb7864",
        adapter="native-llava",
        model="liuhaotian/llava-v1.5-7b",
        model_revision="4481d270cc22fd5c4d1bb5df129622006ccd9234",
        defaults={"retained_tokens": 64},
        license="Apache-2.0",
    ),
    "divprune": CompressionProfile(
        method="divprune",
        repository="https://github.com/vbdi/divprune.git",
        commit="799e2d950aa01ba7860907f5a6d86061f885dca6",
        adapter="native-llava",
        model="liuhaotian/llava-v1.5-7b",
        model_revision="4481d270cc22fd5c4d1bb5df129622006ccd9234",
        defaults={"retained_ratio": 0.098, "layer": 0},
        license="CC-BY-NC-4.0",
    ),
    "fastv": CompressionProfile(
        method="fastv",
        repository="https://github.com/pkunlp-icler/FastV.git",
        commit="d1659729b5bf1be225e99ee15783deeea80f63b1",
        adapter="hf-fastv",
        model="llava-hf/llava-1.5-7b-hf",
        model_revision="a272c74",
        defaults={"pruning_layer": 3, "pruning_fraction": 0.75},
        license="Apache-2.0",
    ),
    # The two profiles below wire up each method's *own* released Qwen2.5-VL
    # fork (not this project's code) so MMIU rows with more than one image can
    # be evaluated. Qwen2.5-VL's NaViT vision encoder emits a variable number
    # of visual tokens per image (there is no fixed 576-token boundary like
    # LLaVA-1.5), so unlike the profiles above, visual token counts for these
    # two methods cannot be known from the profile alone -- see
    # `visual_token_counts` and `official_compression_server.py`, which
    # computes them per request from each request's `image_grid_thw`.
    "hiprune-qwen": CompressionProfile(
        method="hiprune-qwen",
        repository="https://github.com/Danielement321/HiPrune.git",
        commit="82781005a7e72a6be9ede58fd77473efa72b5e4f",
        adapter="qwen-multi-image",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        model_revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
        # Defaults follow HiPrune's README hyperparameter table for Qwen2.5-VL
        # (HIPRUNE_QWEN_RETENTION, HIPRUNE_ALPHA, HIPRUNE_OBJECT_LAYER).
        defaults={"retained_ratio": 0.223, "alpha": 0.1, "object_layer": 16},
        license="MIT",
    ),
    "visionzip-qwen": CompressionProfile(
        method="visionzip-qwen",
        repository="https://github.com/JIA-Lab-research/VisionZip.git",
        commit="8f86b55c6f000eb033e6912538af2dd7dcb30502",
        adapter="qwen-multi-image",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        model_revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
        # VisionZip's Qwen2.5-VL fork (Qwen2_5_VL/qwen2_5vl_visionzip.py)
        # hardcodes its dominant/contextual token ratios (0.65 / 0.05) as
        # inline literals in the forward pass -- there is no parameter, env
        # var, or config attribute exposed to change them without editing the
        # vendored file. This profile therefore has no configurable
        # parameters; any override is rejected by `validated_parameters`.
        defaults={},
        license="Apache-2.0",
    ),
}

# Methods whose released code targets Qwen2.5-VL instead of LLaVA-1.5. These
# support more than one image per MMIU row; see `visual_token_counts` below
# for why they cannot report static before/after token counts.
QWEN_MULTI_IMAGE_METHODS = frozenset({"hiprune-qwen", "visionzip-qwen"})


def get_profile(method: str) -> CompressionProfile:
    try:
        return PROFILES[method]
    except KeyError as exc:
        supported = ", ".join(PROFILES)
        raise ValueError(
            f"unknown compression method {method!r}; choose {supported}"
        ) from exc


def validated_parameters(
    method: str, overrides: dict[str, float | int]
) -> dict[str, float | int]:
    profile = get_profile(method)
    unknown = set(overrides) - set(profile.defaults)
    if unknown:
        raise ValueError(
            f"unsupported {method} parameters: {', '.join(sorted(unknown))}"
        )
    parameters = {**profile.defaults, **overrides}
    if method == "visionzip":
        if parameters["dominant_tokens"] < 2 or parameters["contextual_tokens"] < 1:
            raise ValueError("VisionZip token counts must be positive")
        if parameters["dominant_tokens"] + parameters["contextual_tokens"] > 576:
            raise ValueError("VisionZip cannot retain more than 576 LLaVA-1.5 tokens")
    elif method in {"hiprune", "cdpruner"}:
        if not 1 <= parameters["retained_tokens"] <= 576:
            raise ValueError(f"{method} retained_tokens must be between 1 and 576")
        if method == "hiprune":
            if not 0 <= parameters["alpha"] <= 1:
                raise ValueError("HiPrune alpha must be between 0 and 1")
            if not 1 <= parameters["object_layer"] <= 24:
                raise ValueError("HiPrune object_layer must be between 1 and 24")
    elif method == "divprune":
        if not 0 < parameters["retained_ratio"] <= 1:
            raise ValueError(
                "DivPrune retained_ratio must be greater than 0 and at most 1"
            )
        if parameters["layer"] != 0:
            raise ValueError("the released DivPrune code only implements layer 0")
    elif method == "fastv":
        if not 1 <= parameters["pruning_layer"] < 32:
            raise ValueError("FastV pruning_layer must select a decoder layer")
        if not 0 <= parameters["pruning_fraction"] < 1:
            raise ValueError("FastV pruning_fraction must be in [0, 1)")
    elif method == "hiprune-qwen":
        if not 0 < parameters["retained_ratio"] <= 1:
            raise ValueError(
                "hiprune-qwen retained_ratio must be greater than 0 and at most 1"
            )
        if not 0 <= parameters["alpha"] <= 1:
            raise ValueError("hiprune-qwen alpha must be between 0 and 1")
        if not 1 <= parameters["object_layer"] <= 32:
            raise ValueError("hiprune-qwen object_layer must select a vision layer")
    return parameters


def visual_token_counts(
    method: str, parameters: dict[str, float | int]
) -> tuple[int, int]:
    if method in QWEN_MULTI_IMAGE_METHODS:
        raise ValueError(
            f"{method} has no static visual token count: Qwen2.5-VL's NaViT "
            "vision encoder emits a variable number of tokens per image "
            "(no fixed 576-token boundary), and the released fork prunes a "
            "ratio of whatever that count is. Counts must be computed per "
            "request from that request's image_grid_thw instead."
        )
    before = 576
    if method == "visionzip":
        after = int(parameters["dominant_tokens"] + parameters["contextual_tokens"])
    elif method in {"hiprune", "cdpruner"}:
        after = int(parameters["retained_tokens"])
    elif method == "divprune":
        after = round(before * float(parameters["retained_ratio"]))
    elif method == "fastv":
        after = round(before * (1 - float(parameters["pruning_fraction"])))
    else:
        raise ValueError(f"unknown compression method: {method}")
    return before, max(1, after)
