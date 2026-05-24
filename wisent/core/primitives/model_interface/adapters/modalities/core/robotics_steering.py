"""Robotics adapter: steering, rollout, and trajectory methods."""
from __future__ import annotations
from typing import Any, Callable, Dict, List
import torch
import torch.nn as nn
from wisent.core.primitives.model_interface.adapters.base import SteeringConfig
from wisent.core.primitives.models.modalities import RobotState, RobotAction, RobotTrajectory
from wisent.core.utils.infra_tools.errors import UnknownTypeError
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
from wisent.core.utils.config_tools.constants import TEMPORAL_RAMP_MAX

__all__ = [
    "forward_with_steering_robotics", "generate_unsteered_robotics",
    "act_robotics", "rollout_robotics", "compute_trajectory_steering_vector",
]


def forward_with_steering_robotics(adapter, content, steering_vectors, config=None) -> RobotAction:
    """Get action with steering applied."""
    from wisent.core.primitives.model_interface.adapters.modalities.extended.robotics import RoboticsSteeringConfig
    config = config or RoboticsSteeringConfig()
    with adapter._steering_hooks(steering_vectors, config):
        action_tensor = adapter.forward_policy(content)
    if isinstance(config, RoboticsSteeringConfig) and config.safety_clamp:
        if config.action_bounds:
            action_tensor = torch.clamp(action_tensor, config.action_bounds[0], config.action_bounds[1])
    import numpy as np
    action_np = action_tensor.detach().cpu().numpy()
    if action_np.ndim > 1:
        action_np = action_np.squeeze(0)
    return RobotAction(raw_action=action_np)


def generate_unsteered_robotics(adapter, content, **kwargs: Any) -> RobotAction:
    """Get action without steering."""
    import numpy as np
    action_tensor = adapter.forward_policy(content)
    action_np = action_tensor.detach().cpu().numpy()
    if action_np.ndim > 1:
        action_np = action_np.squeeze(0)
    return RobotAction(raw_action=action_np)


def act_robotics(adapter, state: RobotState, steering_vectors=None, config=None) -> RobotAction:
    """Convenience method to get an action from a state."""
    if steering_vectors is not None:
        return forward_with_steering_robotics(adapter, state, steering_vectors, config)
    return generate_unsteered_robotics(adapter, state)


def rollout_robotics(
    adapter, initial_state: RobotState, env_step_fn: Callable[[RobotAction], RobotState],
    num_steps: int, steering_vectors=None, config=None,
) -> RobotTrajectory:
    """Execute a trajectory with optional steering."""
    states = [initial_state]
    actions = []
    state = initial_state
    for _ in range(num_steps):
        action = act_robotics(adapter, state, steering_vectors, config)
        actions.append(action)
        state = env_step_fn(action)
        states.append(state)
    return RobotTrajectory(states=tuple(states), actions=tuple(actions))


def compute_trajectory_steering_vector(
    adapter, positive_trajectory: RobotTrajectory, negative_trajectory: RobotTrajectory,
    layer: str, aggregation: str, *, temporal_ramp_min: float,
) -> torch.Tensor:
    """Compute steering vector from trajectory pairs."""
    pos_activations = []
    neg_activations = []
    for state in positive_trajectory.states:
        acts = adapter.extract_activations(state, [layer])
        pos_activations.append(acts[layer])
    for state in negative_trajectory.states:
        acts = adapter.extract_activations(state, [layer])
        neg_activations.append(acts[layer])
    pos_tensor = torch.stack(pos_activations)
    neg_tensor = torch.stack(neg_activations)
    if aggregation == "mean":
        pos_agg = pos_tensor.mean(dim=0)
        neg_agg = neg_tensor.mean(dim=0)
    elif aggregation == "last":
        pos_agg = pos_tensor[-1]
        neg_agg = neg_tensor[-1]
    elif aggregation == "weighted":
        weights = torch.linspace(temporal_ramp_min, TEMPORAL_RAMP_MAX, len(pos_activations))
        weights = weights / weights.sum()
        pos_agg = (pos_tensor * weights.view(-1, 1, 1)).sum(dim=0)
        weights = torch.linspace(temporal_ramp_min, TEMPORAL_RAMP_MAX, len(neg_activations))
        weights = weights / weights.sum()
        neg_agg = (neg_tensor * weights.view(-1, 1, 1)).sum(dim=0)
    else:
        raise UnknownTypeError(entity_type="aggregation", value=aggregation,
                               valid_values=["mean", "max", "min", "last", "weighted"])
    return pos_agg.squeeze() - neg_agg.squeeze()
