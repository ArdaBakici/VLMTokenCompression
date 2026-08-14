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
}


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
    return parameters


def visual_token_counts(
    method: str, parameters: dict[str, float | int]
) -> tuple[int, int]:
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
