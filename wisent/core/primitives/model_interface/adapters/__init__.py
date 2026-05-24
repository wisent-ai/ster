"""
Multi-modal adapters for Wisent contrastive steering.

Adapters provide a unified interface for different modalities (text, audio, video, robotics)
while keeping the core steering logic modality-agnostic.
"""
from wisent.core.primitives.model_interface.adapters.base import BaseAdapter, AdapterError
from wisent.core.primitives.model_interface.adapters.text import TextAdapter
from wisent.core.primitives.model_interface.adapters.modalities import (
    AudioAdapter,
    ImageAdapter,
    VideoAdapter,
    RoboticsAdapter,
    MultimodalAdapter,
)

__all__ = [
    "BaseAdapter",
    "AdapterError",
    "TextAdapter",
    "AudioAdapter",
    "ImageAdapter",
    "VideoAdapter",
    "RoboticsAdapter",
    "MultimodalAdapter",
]
