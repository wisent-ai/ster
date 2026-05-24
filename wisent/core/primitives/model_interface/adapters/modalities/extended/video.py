"""
Video adapter for video understanding and generation steering.

Supports models like VideoMAE, TimeSformer, and video generation models.
Enables contrastive steering for:
- Video content safety
- Action/behavior steering in generated video
- Style and motion control

Implementation split into _video_helpers/video_core.py and
_video_helpers/video_ops.py to keep files under 300 lines.
"""
from __future__ import annotations

from wisent.core.primitives.model_interface.adapters.modalities.extended._video_helpers.video_core import (
    VideoSteeringConfig,
    VideoAdapterCore,
)
from wisent.core.primitives.model_interface.adapters.modalities.extended._video_helpers.video_ops import (
    VideoOpsMixin,
)

__all__ = ["VideoAdapter", "VideoSteeringConfig"]


class VideoAdapter(VideoOpsMixin, VideoAdapterCore):
    """
    Adapter for video model steering.

    Supports various video models:
    - VideoMAE (video encoding): Steer video representations
    - TimeSformer (video classification): Steer classification behavior
    - Video generation models: Steer synthesis style/content

    Example:
        >>> adapter = VideoAdapter(model_name="MCG-NJU/videomae-base")
        >>> video = VideoContent.from_file("action.mp4")
        >>> activations = adapter.extract_activations(video)
        >>> # Steer toward safe actions
        >>> output = adapter.generate(video, steering_vectors=safe_vectors)
    """
    pass
