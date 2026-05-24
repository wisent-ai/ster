"""
Robotics adapter operations: encode, decode, steering, rollout, trajectory.

Extracted from robotics.py to keep files under 300 lines.
"""
from __future__ import annotations

from typing import Any, Dict, List, Callable

import torch
import torch.nn as nn
import numpy as np

from wisent.core.primitives.model_interface.adapters.base import (
    AdapterError,
    InterventionPoint,
    SteeringConfig,
)
from wisent.core.primitives.models.modalities import (
    RobotState,
    RobotAction,
    RobotTrajectory,
)
from wisent.core.utils.infra_tools.errors import UnknownTypeError
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations
from wisent.core.utils.config_tools.constants import TEMPORAL_RAMP_MAX
from wisent.core.primitives.model_interface.adapters.modalities.extended._helpers.robotics_core import (
    RoboticsSteeringConfig,
    InputType,
)


class RoboticsOpsMixin:
    """Mixin with encode, decode, steering, rollout, and trajectory methods."""

    def encode(self, content: InputType) -> torch.Tensor:
        """Encode robot state/trajectory to latent representation."""
        if isinstance(content, RobotTrajectory):
            states = torch.stack([s.to_tensor() for s in content.states])
            if states.dim() == 2:
                states = states.unsqueeze(0)
        else:
            states = content.to_tensor()
            if states.dim() == 1:
                states = states.unsqueeze(0)
        states = states.float().to(self.device)
        if self._observation_encoder is not None:
            with torch.no_grad():
                return self._observation_encoder(states)
        policy_type = self._detect_policy_type()
        if policy_type == self.POLICY_TYPE_MLP:
            with torch.no_grad():
                x = states
                layers = self._resolve_policy_layers()
                n_encoder = len(layers) // 2
                for layer in layers[:n_encoder]:
                    if isinstance(layer, nn.Linear):
                        x = layer(x)
                        x = torch.relu(x)
                return x
        elif policy_type == self.POLICY_TYPE_TRANSFORMER:
            with torch.no_grad():
                if hasattr(self.model, "encode"):
                    return self.model.encode(states)
                outputs = self.model(states, output_hidden_states=True)
                if hasattr(outputs, "last_hidden_state"):
                    return outputs.last_hidden_state
                return outputs.hidden_states[-1]
        return states

    def decode(self, latent: torch.Tensor) -> RobotAction:
        """Decode latent representation to action."""
        if self._action_decoder is not None:
            with torch.no_grad():
                action_tensor = self._action_decoder(latent)
        else:
            policy_type = self._detect_policy_type()
            if policy_type == self.POLICY_TYPE_MLP:
                layers = self._resolve_policy_layers()
                n_encoder = len(layers) // 2
                x = latent
                for i, layer in enumerate(layers[n_encoder:]):
                    if isinstance(layer, nn.Linear):
                        x = layer(x)
                        if i < len(layers[n_encoder:]) - 1:
                            x = torch.relu(x)
                action_tensor = x
            else:
                if hasattr(self.model, "action_head"):
                    action_tensor = self.model.action_head(latent)
                else:
                    action_tensor = latent[..., :self.action_dim]
        action_np = action_tensor.detach().cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.squeeze(0)
        return RobotAction(raw_action=action_np)

    def get_intervention_points(self) -> List[InterventionPoint]:
        """Get available intervention points in the policy network."""
        points = []
        layers = self._resolve_policy_layers()
        policy_type = self._detect_policy_type()
        if policy_type == self.POLICY_TYPE_MLP:
            for i, name in enumerate(self._layer_names):
                recommended = i >= len(layers) // 2
                points.append(
                    InterventionPoint(
                        name=f"layer.{i}",
                        module_path=name,
                        description=f"Policy layer {i} ({name})",
                        recommended=recommended,
                    )
                )
        else:
            for i in range(len(layers)):
                recommended = i >= len(layers) // 2
                points.append(
                    InterventionPoint(
                        name=f"layer.{i}",
                        module_path=f"{self._layer_path}.{i}" if hasattr(self, "_layer_path") else f"layers.{i}",
                        description=f"Policy layer {i}",
                        recommended=recommended,
                    )
                )
        return points

    def extract_activations(
        self,
        content: InputType,
        layers: List[str] | None = None,
    ) -> LayerActivations:
        """Extract activations from policy layers."""
        all_points = {ip.name: ip for ip in self.get_intervention_points()}
        target_layers = layers if layers else list(all_points.keys())
        activations: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations[layer_name] = output.detach().cpu()
            return hook

        try:
            for layer_name in target_layers:
                if layer_name not in all_points:
                    continue
                ip = all_points[layer_name]
                module = self._get_module_by_path(ip.module_path)
                if module is not None:
                    handle = module.register_forward_hook(make_hook(layer_name))
                    hooks.append(handle)
            _ = self.forward_policy(content)
        finally:
            for handle in hooks:
                handle.remove()
        return LayerActivations(activations)

    def forward_policy(self, content: InputType) -> torch.Tensor:
        """Forward pass through the policy network."""
        if isinstance(content, RobotTrajectory):
            states = torch.stack([s.to_tensor() for s in content.states])
        else:
            states = content.to_tensor()
        if states.dim() == 1:
            states = states.unsqueeze(0)
        states = states.float().to(self.device)
        with torch.no_grad():
            return self.model(states)

    def forward_with_steering(
        self,
        content: InputType,
        steering_vectors: LayerActivations,
        config: SteeringConfig | RoboticsSteeringConfig | None = None,
    ) -> RobotAction:
        """Get action with steering applied."""
        config = config or RoboticsSteeringConfig()
        with self._steering_hooks(steering_vectors, config):
            action_tensor = self.forward_policy(content)
        if isinstance(config, RoboticsSteeringConfig) and config.safety_clamp:
            if config.action_bounds:
                action_tensor = torch.clamp(
                    action_tensor,
                    config.action_bounds[0],
                    config.action_bounds[1],
                )
        action_np = action_tensor.detach().cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.squeeze(0)
        return RobotAction(raw_action=action_np)

    def _generate_unsteered(self, content: InputType, **kwargs: Any) -> RobotAction:
        """Get action without steering."""
        action_tensor = self.forward_policy(content)
        action_np = action_tensor.detach().cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.squeeze(0)
        return RobotAction(raw_action=action_np)

    def act(
        self,
        state: RobotState,
        steering_vectors: LayerActivations | None = None,
        config: RoboticsSteeringConfig | None = None,
    ) -> RobotAction:
        """Convenience method to get an action from a state."""
        if steering_vectors is not None:
            return self.forward_with_steering(state, steering_vectors, config)
        return self._generate_unsteered(state)

    def rollout(
        self,
        initial_state: RobotState,
        env_step_fn: Callable[[RobotAction], RobotState],
        num_steps: int,
        steering_vectors: LayerActivations | None = None,
        config: RoboticsSteeringConfig | None = None,
    ) -> RobotTrajectory:
        """Execute a trajectory with optional steering."""
        states = [initial_state]
        actions = []
        state = initial_state
        for _ in range(num_steps):
            action = self.act(state, steering_vectors, config)
            actions.append(action)
            state = env_step_fn(action)
            states.append(state)
        return RobotTrajectory(states=tuple(states), actions=tuple(actions))

    def compute_trajectory_steering_vector(
        self,
        positive_trajectory: RobotTrajectory,
        negative_trajectory: RobotTrajectory,
        layer: str,
        aggregation: str,
        *,
        temporal_ramp_min: float,
    ) -> torch.Tensor:
        """Compute steering vector from trajectory pairs."""
        pos_activations = []
        neg_activations = []
        for state in positive_trajectory.states:
            acts = self.extract_activations(state, [layer])
            pos_activations.append(acts[layer])
        for state in negative_trajectory.states:
            acts = self.extract_activations(state, [layer])
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
            raise UnknownTypeError(
                entity_type="aggregation",
                value=aggregation,
                valid_values=["mean", "max", "min", "last", "weighted"],
            )
        return pos_agg.squeeze() - neg_agg.squeeze()
