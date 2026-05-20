"""
Unified Wisent interface for multi-modal contrastive steering.

This module provides a high-level API that works across all modalities
while maintaining backward compatibility with the text-only API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, TypeVar
from dataclasses import dataclass, field
import logging

import torch

from wisent.core.primitives.model_interface.adapters.base import BaseAdapter, SteeringConfig, AdapterError
from wisent.core.primitives.model_interface.adapters.text import TextAdapter
from wisent.core.primitives.model_interface.adapters import AudioAdapter
from wisent.core.primitives.model_interface.adapters import VideoAdapter
from wisent.core.primitives.model_interface.adapters import RoboticsAdapter
from wisent.core.primitives.model_interface.adapters import MultimodalAdapter
# ImageAdapter ships separately and is absent from some published wheels
# (0.11.36, 0.11.46). Guard the import so `core.wisent.Wisent` is still
# constructible for text/audio/video/robotics; image-modality construction
# below will raise a clear error at call time instead of crashing every
# `import wisent` consumer through this transitive import.
try:
    from wisent.core.primitives.model_interface.adapters import ImageAdapter
except ImportError:
    ImageAdapter = None
from wisent.core.utils.infra_tools.errors import UnknownTypeError, NoTrainedVectorsError
from wisent.core.primitives.models.modalities import (
    Modality,
    ModalityContent,
    ContentType,
    TextContent,
    AudioContent,
    VideoContent,
    ImageContent,
    RobotState,
    RobotTrajectory,
    MultimodalContent,
    wrap_content,
)
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
from wisent.core.control.steering_methods.core.atoms import BaseSteeringMethod

__all__ = [
    "Wisent",
    "TraitConfig",
]

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT", bound=ModalityContent)
OutputT = TypeVar("OutputT")

from wisent.core.primitives.model_interface.entry_points._wisent_pairs import WisentPairsMixin
from wisent.core.primitives.model_interface.entry_points._wisent_training import WisentTrainingMixin
from wisent.core.primitives.model_interface.entry_points._wisent_io import WisentIOMixin

@dataclass
class TraitConfig:
    """
    Configuration for a steering trait.

    Attributes:
        name: Unique identifier for the trait
        description: Human-readable description
        steering_vectors: Per-layer steering vectors
        default_scale: Default steering strength
        layers: Which layers to apply to (None = all recommended)
    """
    name: str
    description: str = ""
    steering_vectors: LayerActivations | None = None
    default_scale: float = None
    layers: List[str] | None = None


class Wisent(WisentPairsMixin, WisentTrainingMixin, WisentIOMixin):
    """
    Unified interface for multi-modal contrastive steering.

    Wisent provides a simple, consistent API for steering AI models across
    different modalities (text, audio, video, robotics). The core workflow is:

    1. Create a Wisent instance with an appropriate adapter
    2. Add contrastive pairs (positive/negative examples)
    3. Train steering vectors from the pairs
    4. Generate outputs with steering applied

    Example (Text):
        >>> wisent = Wisent.for_text("meta-llama/Llama-3-8B-Instruct")
        >>> wisent.add_pair(
        ...     positive="I'd be happy to help you with that.",
        ...     negative="I refuse to help with that.",
        ...     trait="helpfulness"
        ... )
        >>> wisent.train()
        >>> response = wisent.generate("How do I cook pasta?", steer={"helpfulness": 1.5})

    Example (Audio):
        >>> wisent = Wisent.for_audio("openai/whisper-large-v3")
        >>> wisent.add_pair(
        ...     positive=AudioContent.from_file("calm_speech.wav"),
        ...     negative=AudioContent.from_file("angry_speech.wav"),
        ...     trait="calmness"
        ... )
        >>> wisent.train()
        >>> transcript = wisent.generate(audio_input, steer={"calmness": 1.2})

    Example (Video):
        >>> wisent = Wisent.for_video("MCG-NJU/videomae-base")
        >>> wisent.add_pair(
        ...     positive=VideoContent.from_file("safe_action.mp4"),
        ...     negative=VideoContent.from_file("unsafe_action.mp4"),
        ...     trait="safety"
        ... )
        >>> wisent.train()
        >>> embedding = wisent.generate(video_input, steer={"safety": 2.0})

    Example (Robotics):
        >>> wisent = Wisent.for_robotics(model=my_policy_network)
        >>> wisent.add_pair(
        ...     positive=gentle_trajectory,
        ...     negative=forceful_trajectory,
        ...     trait="gentleness"
        ... )
        >>> wisent.train()
        >>> action = wisent.act(robot_state, steer={"gentleness": 1.5})
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        steering_method: Optional[str] = None,
        device: str | None = None,
    ):
        """
        Initialize Wisent with an adapter.

        Args:
            adapter: The modality-specific adapter to use
            steering_method: Method for computing steering vectors ("caa", etc.)
            device: Override device (uses adapter's device by default)
        """
        self.adapter = adapter
        self.steering_method_name = steering_method
        self.device = device or adapter.device

        # Storage for pairs and trained vectors
        self._pairs: Dict[str, List[tuple[ModalityContent, ModalityContent]]] = {}
        self._traits: Dict[str, TraitConfig] = {}
        self._trained = False

    # ==================== Factory Methods ====================

    @classmethod
    def for_text(
        cls,
        model_name: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> "Wisent":
        """
        Create a Wisent instance for text/LLM steering.

        Args:
            model_name: HuggingFace model identifier
            device: Target device
            **kwargs: Additional adapter arguments

        Returns:
            Configured Wisent instance
        """
        adapter = TextAdapter(model_name=model_name, device=device, **kwargs)
        return cls(adapter)

    @classmethod
    def for_audio(
        cls,
        model_name: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> "Wisent":
        """
        Create a Wisent instance for audio steering.

        Args:
            model_name: HuggingFace model identifier
            device: Target device
            **kwargs: Additional adapter arguments

        Returns:
            Configured Wisent instance
        """
        adapter = AudioAdapter(model_name=model_name, device=device, **kwargs)
        return cls(adapter)

    @classmethod
    def for_video(
        cls,
        model_name: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> "Wisent":
        """
        Create a Wisent instance for video steering.

        Args:
            model_name: HuggingFace model identifier
            device: Target device
            **kwargs: Additional adapter arguments

        Returns:
            Configured Wisent instance
        """
        adapter = VideoAdapter(model_name=model_name, device=device, **kwargs)
        return cls(adapter)

    @classmethod
    def for_robotics(
        cls,
        model: torch.nn.Module,
        device: str | None = None,
        **kwargs: Any,
    ) -> "Wisent":
        """
        Create a Wisent instance for robotics policy steering.

        Args:
            model: Policy network
            device: Target device
            **kwargs: Additional adapter arguments

        Returns:
            Configured Wisent instance
        """
        adapter = RoboticsAdapter(model=model, device=device, **kwargs)
        return cls(adapter)

    @classmethod
    def for_multimodal(
        cls,
        model_name: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> "Wisent":
        """
        Create a Wisent instance for multimodal (VLM, etc.) steering.

        Args:
            model_name: HuggingFace model identifier
            device: Target device
            **kwargs: Additional adapter arguments

        Returns:
            Configured Wisent instance
        """
        adapter = MultimodalAdapter(model_name=model_name, device=device, **kwargs)
        return cls(adapter)

    @classmethod
    def for_image(
        cls,
        model_name: str,
        device: str | None = None,
        **kwargs: Any,
    ) -> "Wisent":
        """
        Create a Wisent instance for text-to-image diffusion model steering.

        Args:
            model_name: HuggingFace diffusers pipeline identifier
                (e.g. "Tongyi-MAI/Z-Image-Turbo", "stabilityai/stable-diffusion-xl-base-1.0")
            device: Target device
            **kwargs: Additional adapter arguments forwarded to DiffusionPipeline.from_pretrained

        Returns:
            Configured Wisent instance
        """
        if ImageAdapter is None:
            raise ImportError(
                "ImageAdapter is not available in the installed wisent wheel. "
                "Install a wisent build that ships "
                "wisent.core.primitives.model_interface.adapters.modalities.extended.image "
                "to use image modality."
            )
        adapter = ImageAdapter(model_name=model_name, device=device, **kwargs)
        return cls(adapter)

    # ==================== Pair Management ====================

