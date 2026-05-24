"""
Robotics adapter for policy network steering.

Enables contrastive steering for robot behavior:
- Safe/gentle manipulation
- Goal-directed behavior steering
- Motion style control
- Human-robot interaction patterns

Implementation split into _helpers/robotics_core.py and
_helpers/robotics_ops.py to keep files under 300 lines.
"""
from __future__ import annotations

from wisent.core.primitives.model_interface.adapters.modalities.extended._helpers.robotics_core import (
    RoboticsSteeringConfig,
    InputType,
    RoboticsAdapterCore,
)
from wisent.core.primitives.model_interface.adapters.modalities.extended._helpers.robotics_ops import (
    RoboticsOpsMixin,
)

__all__ = ["RoboticsAdapter", "RoboticsSteeringConfig"]


class RoboticsAdapter(RoboticsOpsMixin, RoboticsAdapterCore):
    """
    Adapter for robot policy network steering.

    Supports various policy architectures:
    - MLP policies (standard RL)
    - Transformer policies (decision transformer, etc.)
    - Diffusion policies
    - Vision-language-action models

    The adapter hooks into the policy network to steer the action
    distribution toward desired behaviors (e.g., gentleness, safety).

    Example:
        >>> adapter = RoboticsAdapter(model=my_policy_network)
        >>> state = RobotState(joint_positions=np.array([...]))
        >>> # Steer toward gentle manipulation
        >>> action = adapter.act(state, steering_vectors=gentle_vectors)
    """
    pass
