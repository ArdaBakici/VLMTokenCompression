import json
import os
import subprocess
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from compression_profiles import (
    PROFILES,
    get_profile,
    validated_parameters,
    visual_token_counts,
)
from official_compression_server import Handler

ROOT = Path(__file__).parent
LAUNCHER = ROOT / "scripts" / "run_mmiu_compression.sh"


class CompressionProfileTest(unittest.TestCase):
    def test_profiles_pin_every_official_repository(self):
        self.assertEqual(
            set(PROFILES), {"visionzip", "hiprune", "cdpruner", "divprune", "fastv"}
        )
        for profile in PROFILES.values():
            with self.subTest(method=profile.method):
                self.assertRegex(profile.commit, r"^[0-9a-f]{40}$")
                self.assertIn("llava", profile.model.lower())
                self.assertIsNotNone(profile.model_revision)

    def test_validates_method_specific_parameters(self):
        self.assertEqual(
            validated_parameters("visionzip", {"dominant_tokens": 54}),
            {"dominant_tokens": 54, "contextual_tokens": 10},
        )
        invalid = (
            ("visionzip", {"dominant_tokens": 576}),
            ("hiprune", {"alpha": 2.0}),
            ("cdpruner", {"retained_tokens": 0}),
            ("divprune", {"retained_ratio": 0}),
            ("fastv", {"pruning_fraction": 1}),
        )
        for method, parameters in invalid:
            with self.subTest(method=method), self.assertRaises(ValueError):
                validated_parameters(method, parameters)

    def test_reports_explicit_visual_token_counts(self):
        expected = {
            "visionzip": 64,
            "hiprune": 192,
            "cdpruner": 64,
            "divprune": 56,
            "fastv": 144,
        }
        for method, after in expected.items():
            parameters = validated_parameters(method, {})
            self.assertEqual(visual_token_counts(method, parameters), (576, after))

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            get_profile("unknown")


class CompressionLauncherTest(unittest.TestCase):
    def test_prints_profiles_without_creating_an_environment(self):
        for method in PROFILES:
            environment = os.environ.copy()
            environment["PRINT_COMPRESSION_PROFILE"] = "1"
            result = subprocess.run(
                ["bash", str(LAUNCHER), method],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            profile = dict(line.split("=", 1) for line in result.stdout.splitlines())
            with self.subTest(method=method):
                self.assertEqual(profile["method"], method)
                self.assertEqual(profile["max_images_per_example"], "1")
                json.loads(profile["parameters"])


class FakeBackend:
    model_id = "test-model"
    visual_tokens_before = 576
    visual_tokens_after = 64

    def start_generation(self, text, images, max_tokens):
        return iter(["B"]), {}

    def finish_generation(self, state, text):
        return 100, 1


class AdapterProtocolTest(unittest.TestCase):
    def test_streams_openai_chunks_and_visual_usage(self):
        Handler.backend = FakeBackend()
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=json.dumps(
                    {
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "question"}],
                        "max_tokens": 16,
                        "stream": True,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                body = response.read().decode()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertIn('"content": "B"', body)
        self.assertIn('"finish_reason": "stop"', body)
        self.assertIn('"visual_tokens_before": 576', body)
        self.assertIn('"visual_tokens_after": 64', body)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))


if __name__ == "__main__":
    unittest.main()
