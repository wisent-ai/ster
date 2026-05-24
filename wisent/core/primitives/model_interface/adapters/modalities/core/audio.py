"""
Audio adapter for speech and audio model steering.

Supports models like Whisper, Wav2Vec2, and audio generation models.
Enables contrastive steering for:
- Speech emotion/tone control
- Audio content moderation
- Voice style steering

Implementation split into _helpers/audio_core.py and
_helpers/audio_ops.py to keep files under 300 lines.
"""
from __future__ import annotations

from wisent.core.primitives.model_interface.adapters.modalities.extended._helpers.audio_core import (
    AudioAdapterCore,
)
from wisent.core.primitives.model_interface.adapters.modalities.extended._helpers.audio_ops import (
    AudioOpsMixin,
)

__all__ = ["AudioAdapter"]


class AudioAdapter(AudioOpsMixin, AudioAdapterCore):
    """
    Adapter for audio model steering.

    Supports various audio models:
    - Whisper (speech-to-text): Steer transcription behavior
    - Wav2Vec2 (audio encoding): Steer audio representations
    - Audio generation models: Steer synthesis style/tone

    Example:
        >>> adapter = AudioAdapter(model_name="openai/whisper-large-v3")
        >>> audio = AudioContent.from_file("speech.wav")
        >>> activations = adapter.extract_activations(audio)
        >>> # Steer toward calm speech patterns
        >>> output = adapter.generate(audio, steering_vectors=calm_vectors)
    """
    pass
