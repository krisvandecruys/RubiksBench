"""Provider-independent image generation interface.

A backend turns a ``GenerationRequest`` into a ``GenerationResult``. Provider
or model failures are returned as results with ``error`` set, never retried
or regenerated silently, so they can be reported separately.
"""

import abc
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Self


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    prompt: str
    seed: int | None
    width: int = 1024
    height: int = 1024
    negative_prompt: str | None = None
    steps: int | None = None
    guidance: float | None = None
    # Provider-specific parameters, passed through unchanged.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationError:
    code: str
    message: str
    retryable: bool = False


@dataclass
class GenerationResult:
    request: GenerationRequest
    backend: str
    started_at: str
    latency_s: float
    image: bytes | None = None
    image_format: str | None = None  # file extension, e.g. "png"
    cost: float | None = None
    error: GenerationError | None = None
    # False when the model does not accept a seed and the request was sent without one.
    seed_supported: bool = True
    # Request fields the provider made us change, e.g. {"size": {"requested": [1024, 1024], "used": [2048, 2048]}}.
    adjustments: dict[str, Any] = field(default_factory=dict)
    # Exactly what was sent to the provider and what came back (minus image bytes).
    provider_request: dict[str, Any] = field(default_factory=dict)
    provider_response: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.image is not None

    def metadata(self) -> dict[str, Any]:
        """JSON-serialisable record of everything except the image bytes."""
        out = asdict(self)
        del out["image"]
        return out


@dataclass
class ModelInfo:
    id: str  # what to pass as GenerationRequest.model
    name: str
    version: str | None = None
    architecture: str | None = None
    source: str | None = None
    # Everything else the provider returned.
    details: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ImageBackend(abc.ABC):
    """Base class for generation providers. Use as an async context manager."""

    name: ClassVar[str]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    @abc.abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one image. Must not raise for provider-side failures."""

    async def search_models(self, query: str, *, text_to_image_only: bool = True) -> list[ModelInfo]:
        """Find models whose name matches ``query``. Optional for backends."""
        raise NotImplementedError(f"backend {self.name!r} does not support model search")


_REGISTRY: dict[str, type[ImageBackend]] = {}


def register_backend[B: type[ImageBackend]](cls: B) -> B:
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str, **options: Any) -> ImageBackend:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(_REGISTRY)}") from None
    return cls(**options)


def available_backends() -> list[str]:
    return sorted(_REGISTRY)
