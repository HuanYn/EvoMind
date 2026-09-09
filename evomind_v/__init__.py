"""EvoMind-V's deterministic multi-view extension to official MiniMind-V.

The package keeps model imports lazy so data preparation can run with only PIL
and torch, without importing the transformer model stack.
"""

from .views import VIEW_VERSION, make_image_views, num_views, preprocess_views
from .cache import VisionFeatureCache, fingerprint_vision

__all__ = [
    "EvoMindVLM", "VLMConfig", "VIEW_VERSION", "make_image_views",
    "num_views", "preprocess_views", "VisionFeatureCache", "fingerprint_vision",
]


def __getattr__(name):
    if name in {"EvoMindVLM", "VLMConfig"}:
        from .model import EvoMindVLM, VLMConfig
        return {"EvoMindVLM": EvoMindVLM, "VLMConfig": VLMConfig}[name]
    raise AttributeError(name)
