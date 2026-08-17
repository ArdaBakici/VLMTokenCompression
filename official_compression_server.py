"""OpenAI-compatible server for pinned official method code.

Every method except `hiprune-qwen`/`visionzip-qwen` wraps a LLaVA-1.5-based
fork and only ever handles exactly one image per request (LLaVA-1.5 has a
single `<image>` placeholder per conversation turn). `hiprune-qwen` and
`visionzip-qwen` instead wrap each method's own released Qwen2.5-VL fork,
which is natively multi-image, so those two accept any number of images.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from compression_profiles import (
    QWEN_MULTI_IMAGE_METHODS,
    get_profile,
    validated_parameters,
    visual_token_counts,
)

# Files inside each method's own Qwen2.5-VL fork, relative to the pinned
# repository checkout. Both are complete standalone copies of transformers'
# modeling_qwen2_5_vl.py (only Qwen2_5_VLConfig/Qwen2_5_VLVisionConfig are
# imported from transformers itself), so loading them only requires
# executing the file -- no package `__init__.py` or sys.path changes needed,
# since their internal imports are all fully qualified (e.g.
# `from transformers.models.qwen2_5_vl...`), not relative to the checkout.
QWEN_MODEL_FILES = {
    "hiprune-qwen": Path("Qwen2_5_VL") / "qwen2_5_vl_HiPrune.py",
    "visionzip-qwen": Path("Qwen2_5_VL") / "qwen2_5vl_visionzip.py",
}


def _load_module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_data_url(url: str) -> bytes:
    match = re.fullmatch(r"data:[^;,]+;base64,(.*)", url, re.DOTALL)
    if not match:
        raise ValueError("only base64 data URLs are supported")
    return base64.b64decode(match.group(1), validate=True)


def request_content(messages: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    text = []
    images = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            text.append(content)
            continue
        for item in content:
            if item.get("type") == "text":
                text.append(str(item.get("text") or ""))
            elif item.get("type") == "image_url":
                # Pillow is only needed for requests that actually carry media,
                # so a text-only probe works before the model stack is present.
                from PIL import Image

                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    data = parse_data_url(url)
                elif url.startswith("file:"):
                    data = Path(unquote(urlparse(url).path)).read_bytes()
                else:
                    with urlopen(url, timeout=30) as response:
                        data = response.read()
                images.append(Image.open(BytesIO(data)).convert("RGB"))
    return "\n".join(part for part in text if part), images


class OfficialBackend:
    def __init__(
        self,
        method: str,
        model: str,
        revision: str | None,
        repository: Path,
        llava_repository: Path | None,
        parameters: dict[str, float | int],
    ) -> None:
        self.method = method
        self.model_id = model
        self.parameters = parameters
        self.lock = threading.Lock()
        if method in QWEN_MULTI_IMAGE_METHODS:
            # Qwen2.5-VL's NaViT vision encoder emits a variable number of
            # visual tokens per image, so there is no fixed before/after
            # boundary like LLaVA-1.5's 576 patch tokens. Real counts are
            # computed per request in _qwen_inputs() from that request's
            # image_grid_thw and stashed in `state` for _usage() to report;
            # these are placeholders only used before the first request.
            self.visual_tokens_before = 0
            self.visual_tokens_after = 0
            self._load_qwen(method, model, revision, repository)
        elif method == "fastv":
            # LLaVA-1.5 encodes every image as exactly 576 patch tokens, so the
            # configured budget is the retained count rather than an estimate.
            # It is still the method's own boundary: FastV keeps all 576
            # tokens until its pruning layer, so its prompt usage stays
            # uncompressed.
            self.visual_tokens_before, self.visual_tokens_after = visual_token_counts(
                method, parameters
            )
            self._load_fastv(model, revision, repository)
        else:
            self.visual_tokens_before, self.visual_tokens_after = visual_token_counts(
                method, parameters
            )
            self._load_native_llava(
                method, model, revision, repository, llava_repository
            )

    def _load_native_llava(
        self,
        method: str,
        model: str,
        revision: str | None,
        repository: Path,
        llava_repository: Path | None,
    ) -> None:
        if method == "hiprune":
            os.environ["HIPRUNE_RETENTION"] = str(self.parameters["retained_tokens"])
            os.environ["HIPRUNE_ALPHA"] = str(self.parameters["alpha"])
            os.environ["HIPRUNE_OBJECT_LAYER"] = str(self.parameters["object_layer"])
            source = repository / "LLaVA"
        elif method == "divprune":
            os.environ["SUBSET_RATIO"] = str(self.parameters["retained_ratio"])
            os.environ["LAYER_INDEX"] = "0"
            source = repository / "LLaVA"
        elif method == "cdpruner":
            source = repository
        else:
            if llava_repository is None:
                raise ValueError("VisionZip requires --llava-repository")
            source = llava_repository
        sys.path.insert(0, str(source))
        sys.path.insert(0, str(repository))

        import torch
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
        if revision is not None:
            kwargs["revision"] = revision
        if method == "cdpruner":
            kwargs["visual_token_num"] = int(self.parameters["retained_tokens"])
        name = get_model_name_from_path(model)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model, None, name, **kwargs
        )
        if method == "visionzip":
            from visionzip import visionzip

            self.model = visionzip(
                self.model,
                dominant=int(self.parameters["dominant_tokens"]),
                contextual=int(self.parameters["contextual_tokens"]),
            )
        self.model.eval()
        self.torch = torch
        self.adapter = "native-llava"

    def _load_fastv(self, model: str, revision: str | None, repository: Path) -> None:
        source = repository / "src" / "FastV" / "llava-hf" / "transformers" / "src"
        sys.path.insert(0, str(source))
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        config = {
            "use_fastv": True,
            "fastv_k": int(self.parameters["pruning_layer"]),
            "fastv_r": float(self.parameters["pruning_fraction"]),
            "image_token_start_index": 5,
            "image_token_length": 576,
        }
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model,
            revision=revision,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            fastv_config=config,
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(model, revision=revision)
        self.model.eval()
        self.torch = torch
        self.adapter = "hf-fastv"

    def _load_qwen(
        self, method: str, model: str, revision: str | None, repository: Path
    ) -> None:
        if method == "hiprune-qwen":
            # qwen2_5_vl_HiPrune.py reads these three at *import* time
            # (module-level globals), so they must be set before the module
            # is executed, not just before generation.
            os.environ["HIPRUNE_QWEN_RETENTION"] = str(self.parameters["retained_ratio"])
            os.environ["HIPRUNE_ALPHA"] = str(self.parameters["alpha"])
            os.environ["HIPRUNE_OBJECT_LAYER"] = str(self.parameters["object_layer"])

        import torch
        from transformers import AutoProcessor

        source = repository / QWEN_MODEL_FILES[method]
        module_name = f"vlm_token_compression_vendor_{method.replace('-', '_')}"
        module = _load_module_from_path(module_name, source)
        model_class = module.Qwen2_5_VLForConditionalGeneration

        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            # HiPrune's Qwen2.5-VL fork asserts attn_implementation != "sdpa"
            # (it needs raw attention weights to rank tokens by attention,
            # which sdpa's fused kernel does not expose). VisionZip's fork has
            # no such restriction but "eager" works for it too, so the same
            # value is used for both rather than adding a per-method knob.
            "attn_implementation": "eager",
        }
        if revision is not None:
            kwargs["revision"] = revision
        self.model = model_class.from_pretrained(model, **kwargs).to(0)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model, revision=revision, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
        self.spatial_merge_size = int(self.model.config.vision_config.spatial_merge_size)
        self.torch = torch
        self.adapter = "qwen-multi-image"

    def _qwen_inputs(
        self, text: str, images: list[Any]
    ) -> tuple[dict[str, Any], int, int]:
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=images, return_tensors="pt"
        ).to(self.model.device)
        # image_grid_thw is (num_images, 3): (temporal, height, width) in
        # patches. Dividing by spatial_merge_size**2 gives the number of
        # visual tokens the vision tower emits per image *before* either
        # method's pruning runs -- the same quantity each vendored fork calls
        # n_image_tokens internally, computed here without touching their
        # forward-pass internals.
        grid = inputs["image_grid_thw"]
        before = int(
            (grid.prod(dim=-1) // (self.spatial_merge_size**2)).sum().item()
        )
        if self.method == "hiprune-qwen":
            # Matches qwen2_5_vl_HiPrune.py's own
            # `visual_token_num = round(n_image_tokens * RETAIN)`.
            after = round(before * float(self.parameters["retained_ratio"]))
        else:
            # Matches qwen2_5vl_visionzip.py's own hardcoded
            # `dominant_num = int(0.65 * n) ; contextual_num = max(int(0.05 * n), 1)`.
            after = int(0.65 * before) + max(int(0.05 * before), 1)
        return inputs, before, max(1, after)

    def _native_inputs(self, text: str, image: Any) -> tuple[Any, dict[str, Any]]:
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{text}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [
                item.to(self.model.device, dtype=self.torch.float16)
                for item in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=self.torch.float16)
        return input_ids, {"images": image_tensor, "image_sizes": [image.size]}

    def start_generation(
        self, text: str, images: list[Any], max_tokens: int
    ) -> tuple[Any, dict[str, Any]]:
        if self.adapter == "qwen-multi-image":
            if not images:
                raise ValueError(
                    f"official {self.method} integration requires at least one image; got 0"
                )
        elif len(images) != 1:
            raise ValueError(
                f"official {self.method} integration requires exactly one image; got {len(images)}"
            )
        self.lock.acquire()
        try:
            from transformers import TextIteratorStreamer

            visual_tokens_before = self.visual_tokens_before
            visual_tokens_after = self.visual_tokens_after
            if self.adapter == "hf-fastv":
                prompt = f"USER: <image>\n{text}\nASSISTANT:"
                inputs = self.processor(prompt, images[0], return_tensors="pt").to(
                    0, self.torch.float16
                )
                tokenizer = self.processor.tokenizer
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": max_tokens,
                    "do_sample": False,
                    "use_cache": False,
                }
                prompt_tokens = int(inputs["input_ids"].shape[1])
            elif self.adapter == "qwen-multi-image":
                inputs, visual_tokens_before, visual_tokens_after = self._qwen_inputs(
                    text, images
                )
                tokenizer = self.processor.tokenizer
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": max_tokens,
                    "do_sample": False,
                    "use_cache": True,
                }
                image_token_id = self.model.config.image_token_id
                image_placeholders = int(
                    (inputs["input_ids"] == image_token_id).sum().item()
                )
                prompt_tokens = (
                    int(inputs["input_ids"].shape[1])
                    - image_placeholders
                    + visual_tokens_after
                )
            else:
                input_ids, multimodal = self._native_inputs(text, images[0])
                tokenizer = self.tokenizer
                generate_kwargs = {
                    "inputs": input_ids,
                    **multimodal,
                    "do_sample": False,
                    "max_new_tokens": max_tokens,
                    "use_cache": True,
                }
                if self.method == "cdpruner":
                    generate_kwargs["texts"] = text
                image_placeholders = int((input_ids == -200).sum().item())
                prompt_tokens = (
                    int(input_ids.shape[1])
                    - image_placeholders
                    + self.visual_tokens_after
                )

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=600,
            )
            generate_kwargs["streamer"] = streamer
            state: dict[str, Any] = {
                "error": None,
                "prompt_tokens": prompt_tokens,
                "tokenizer": tokenizer,
                "visual_tokens_before": visual_tokens_before,
                "visual_tokens_after": visual_tokens_after,
            }

            def generate() -> None:
                try:
                    with self.torch.inference_mode():
                        self.model.generate(**generate_kwargs)
                except Exception as exc:  # noqa: BLE001 - propagate from model thread.
                    state["error"] = exc
                    streamer.end()

            thread = threading.Thread(target=generate, daemon=True)
            state["thread"] = thread
            thread.start()
            return streamer, state
        except Exception:
            self.lock.release()
            raise

    def finish_generation(
        self, state: dict[str, Any], text: str
    ) -> tuple[int, int, int, int]:
        try:
            state["thread"].join()
            if state["error"] is not None:
                raise state["error"]
            # The streamer yields text rather than token IDs, so the completion
            # count is recovered by re-encoding. It can differ from the sampled
            # count by a token when detokenization is not perfectly invertible.
            completion_tokens = len(
                state["tokenizer"].encode(text, add_special_tokens=False)
            )
            return (
                int(state["prompt_tokens"]),
                completion_tokens,
                int(state["visual_tokens_before"]),
                int(state["visual_tokens_after"]),
            )
        finally:
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    backend: OfficialBackend

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def _json(self, status: int, value: Any) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [{"id": self.backend.model_id, "object": "model"}],
                },
            )
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        response_started = False
        state = None
        generation_finished = False
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if request.get("model") != self.backend.model_id:
                raise ValueError(
                    f"requested model {request.get('model')!r}, server provides "
                    f"{self.backend.model_id!r}"
                )
            text, images = request_content(request.get("messages", []))
            started = time.perf_counter()
            streamer, state = self.backend.start_generation(
                text, images, int(request.get("max_tokens", 16))
            )
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                response_started = True
                parts = []
                for part in streamer:
                    if not part:
                        continue
                    parts.append(part)
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": part},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                result = "".join(parts)
                try:
                    (
                        prompt_tokens,
                        completion_tokens,
                        visual_tokens_before,
                        visual_tokens_after,
                    ) = self.backend.finish_generation(state, result)
                finally:
                    generation_finished = True
                usage = self._usage(
                    prompt_tokens, completion_tokens, visual_tokens_before, visual_tokens_after
                )
                for chunk in (
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    },
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "choices": [],
                        "usage": usage,
                    },
                ):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                return
            parts = [part for part in streamer if part]
            result = "".join(parts)
            try:
                (
                    prompt_tokens,
                    completion_tokens,
                    visual_tokens_before,
                    visual_tokens_after,
                ) = self.backend.finish_generation(state, result)
            finally:
                generation_finished = True
            usage = self._usage(
                prompt_tokens, completion_tokens, visual_tokens_before, visual_tokens_after
            )
            self._json(
                200,
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                    "latency_seconds": time.perf_counter() - started,
                },
            )
        except Exception as exc:  # noqa: BLE001 - API boundary returns model failures.
            if state is not None and not generation_finished:
                try:
                    self.backend.finish_generation(state, "")
                except Exception as cleanup_error:  # noqa: BLE001 - original error wins.
                    sys.stderr.write(f"generation cleanup failed: {cleanup_error}\n")
            if response_started:
                self.close_connection = True
            else:
                self._json(400, {"error": {"message": f"{type(exc).__name__}: {exc}"}})

    def _usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        visual_tokens_before: int,
        visual_tokens_after: int,
    ) -> dict[str, int]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "visual_tokens_before": visual_tokens_before,
            "visual_tokens_after": visual_tokens_after,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--llava-repository", type=Path)
    parser.add_argument("--parameters", default="{}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    profile = get_profile(args.method)
    parameters = validated_parameters(args.method, json.loads(args.parameters))
    Handler.backend = OfficialBackend(
        args.method,
        args.model or profile.model,
        args.revision or profile.model_revision,
        args.repository,
        args.llava_repository,
        parameters,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {args.method} at http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
