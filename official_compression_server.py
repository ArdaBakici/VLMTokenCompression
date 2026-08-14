"""OpenAI-compatible single-image server for pinned official method code."""

from __future__ import annotations

import argparse
import base64
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

from compression_profiles import get_profile, validated_parameters, visual_token_counts


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
        # LLaVA-1.5 encodes every image as exactly 576 patch tokens, so the
        # configured budget is the retained count rather than an estimate. It is
        # still the method's own boundary: FastV keeps all 576 tokens until its
        # pruning layer, so its prompt usage stays uncompressed.
        self.visual_tokens_before, self.visual_tokens_after = visual_token_counts(
            method, parameters
        )
        self.lock = threading.Lock()
        if method == "fastv":
            self._load_fastv(model, revision, repository)
        else:
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
        if len(images) != 1:
            raise ValueError(
                f"official {self.method} integration requires exactly one image; got {len(images)}"
            )
        self.lock.acquire()
        try:
            from transformers import TextIteratorStreamer

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

    def finish_generation(self, state: dict[str, Any], text: str) -> tuple[int, int]:
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
            return int(state["prompt_tokens"]), completion_tokens
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
                    prompt_tokens, completion_tokens = self.backend.finish_generation(
                        state, result
                    )
                finally:
                    generation_finished = True
                usage = self._usage(prompt_tokens, completion_tokens)
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
                prompt_tokens, completion_tokens = self.backend.finish_generation(
                    state, result
                )
            finally:
                generation_finished = True
            usage = self._usage(prompt_tokens, completion_tokens)
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

    def _usage(self, prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "visual_tokens_before": self.backend.visual_tokens_before,
            "visual_tokens_after": self.backend.visual_tokens_after,
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
