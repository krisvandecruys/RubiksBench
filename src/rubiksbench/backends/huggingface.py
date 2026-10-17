"""Hugging Face backend: Inference Providers via the official ``huggingface_hub`` client.

One token (``HF_TOKEN`` in the environment, or ``token=``) reaches every
provider Hugging Face routes to (fal-ai, replicate, together, hf-inference,
...). Models are Hub repo ids, e.g. ``black-forest-labs/FLUX.1-dev``; with
``provider="auto"`` Hugging Face picks the first available provider.
"""

import io
import os
import time
from typing import Any, Self

from huggingface_hub import AsyncInferenceClient, HfApi
from huggingface_hub.errors import HfHubHTTPError

from .base import (
    GenerationError,
    GenerationRequest,
    GenerationResult,
    ImageBackend,
    ModelInfo,
    register_backend,
    utc_now,
)


@register_backend
class HuggingFaceBackend(ImageBackend):
    name = "huggingface"

    def __init__(self, token: str | None = None, provider: str = "auto"):
        self._token = token or os.environ.get("HF_TOKEN")
        self._provider = provider
        self._client: AsyncInferenceClient | None = None

    async def __aenter__(self) -> Self:
        self._client = AsyncInferenceClient(provider=self._provider, token=self._token)  # type: ignore[arg-type]
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def build_arguments(self, request: GenerationRequest) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "model": request.model,
            "width": request.width,
            "height": request.height,
            "seed": request.seed,
            "negative_prompt": request.negative_prompt,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance,
        }
        arguments = {k: v for k, v in arguments.items() if v is not None}
        if request.extra:
            arguments["extra_body"] = request.extra
        return arguments

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._client is None:
            raise RuntimeError("use HuggingFaceBackend as an async context manager")
        arguments = self.build_arguments(request)
        result = GenerationResult(
            request, self.name, utc_now(), 0.0,
            provider_request={"provider": self._provider, "prompt": request.prompt, **arguments},
        )
        start = time.monotonic()
        try:
            image = await self._client.text_to_image(request.prompt, **arguments)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            result.image, result.image_format = buffer.getvalue(), "png"
            result.provider_response = {"size": list(image.size), "mode": image.mode}
        except HfHubHTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            result.error = GenerationError(f"http_{status}" if status else "http", str(exc), status in (429, 500, 502, 503))
        except Exception as exc:  # recorded as a failed sample rather than aborting the run
            result.error = GenerationError("unknown", f"{type(exc).__name__}: {exc}")
        result.latency_s = time.monotonic() - start
        return result

    async def search_models(self, query: str, *, text_to_image_only: bool = True) -> list[ModelInfo]:
        """Hub models matching ``query`` that an Inference Provider currently serves."""
        api = HfApi(token=self._token)
        models = api.list_models(
            search=query,
            pipeline_tag="text-to-image" if text_to_image_only else None,
            inference_provider="all",
            sort="downloads",
            limit=100,
            expand=["pipeline_tag", "downloads", "inferenceProviderMapping"],
        )
        found = []
        for m in models:
            providers = sorted(p.provider for p in (m.inference_provider_mapping or []))
            found.append(
                ModelInfo(
                    id=m.id,
                    name=m.id,
                    architecture=m.pipeline_tag,
                    source=", ".join(providers) or None,
                    details={"downloads": m.downloads, "providers": providers},
                )
            )
        return found
