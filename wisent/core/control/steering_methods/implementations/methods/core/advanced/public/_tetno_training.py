"""TETNO behavior and condition vector training."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations, RawActivationMap, LayerName
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.control.steering_methods.methods.advanced._tetno_types import TETNOConfig, TETNOResult
from wisent.core.utils.config_tools.constants import RECURSION_INITIAL_DEPTH
from wisent.core.utils.infra_tools.errors import InsufficientDataError

class TETNOTrainingMixin:
    """Mixin: behavior and condition vector training."""

    def _train_behavior_vectors(
        self,
        pair_set: ContrastivePairSet,
    ) -> Dict[LayerName, torch.Tensor]:
        """Train one CAA behavior vector for every configured steering layer."""
        buckets = self._collect_from_set(pair_set)
        by_index: Dict[int, LayerName] = {}
        for layer_name in buckets:
            try:
                layer_index = int(str(layer_name).split("_")[-1])
            except (ValueError, IndexError) as exc:
                raise InsufficientDataError(
                    reason=f"Unparseable activation layer name: {layer_name!r}"
                ) from exc
            if layer_index in by_index:
                raise InsufficientDataError(
                    reason=f"Multiple activation names resolve to layer {layer_index}"
                )
            by_index[layer_index] = layer_name

        behavior_vectors: Dict[LayerName, torch.Tensor] = {}
        for layer_index in self.config.steering_layers:
            if layer_index not in by_index:
                raise InsufficientDataError(
                    reason=f"Missing configured steering layer {layer_index}"
                )
            layer_name = by_index[layer_index]
            pos_list, neg_list = buckets[layer_name]
            if not pos_list or not neg_list:
                raise InsufficientDataError(
                    reason=f"Empty activations at steering layer {layer_index}"
                )

            pos_tensor = torch.stack(
                [tensor.detach().float().reshape(-1) for tensor in pos_list], dim=0,
            )
            neg_tensor = torch.stack(
                [tensor.detach().float().reshape(-1) for tensor in neg_list], dim=0,
            )
            behavior_vec = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
            if self.config.normalize:
                behavior_vec = F.normalize(behavior_vec, p=2, dim=0)
            behavior_vectors[layer_name] = behavior_vec
            self._training_logs.append({
                "phase": "behavior",
                "layer": str(layer_name),
                "pos_samples": len(pos_list),
                "neg_samples": len(neg_list),
                "vector_norm": behavior_vec.norm().item(),
            })
        return behavior_vectors

    
    def _train_condition_vector(
        self,
        pair_set: ContrastivePairSet,
        log_interval: int,
        *,
        condition_margin: float,
    ) -> torch.Tensor:
        """
        Train condition vector at sensor layer.
        
        The condition vector detects when the behavior (e.g., refusal) is present.
        High similarity with condition vector → steering should activate.
        """
        buckets = self._collect_from_set(pair_set)
        
        sensor_layer_name = None
        for layer_name in buckets:
            try:
                layer_index = int(str(layer_name).split("_")[-1])
            except (ValueError, IndexError):
                continue
            if layer_index == self.config.sensor_layer:
                if sensor_layer_name is not None:
                    raise InsufficientDataError(
                        reason=f"Multiple activation names resolve to sensor layer {layer_index}"
                    )
                sensor_layer_name = layer_name
        if sensor_layer_name is None:
            raise InsufficientDataError(
                reason=f"Missing configured sensor layer {self.config.sensor_layer}"
            )
        
        pos_list, neg_list = buckets[sensor_layer_name]
        if not pos_list or not neg_list:
            raise InsufficientDataError(reason="Empty activations at sensor layer")
        
        # Stack activations
        pos_tensor = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg_tensor = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)
        
        if self.config.use_caa_init:
            # Initialize with CAA direction
            condition_vec = pos_tensor.mean(dim=0) - neg_tensor.mean(dim=0)
        else:
            condition_vec = torch.randn(pos_tensor.shape[1])
        
        # Optimize condition vector to maximize separation
        condition_vec = self._optimize_condition_vector(
            condition_vec, pos_tensor, neg_tensor, log_interval=log_interval,
            condition_margin=condition_margin,
        )
        
        if self.config.normalize:
            condition_vec = F.normalize(condition_vec, p=2, dim=0)
        
        self._training_logs.append({
            "phase": "condition",
            "sensor_layer": str(sensor_layer_name),
            "pos_samples": len(pos_list),
            "neg_samples": len(neg_list),
            "vector_norm": condition_vec.norm().item(),
        })
        
        return condition_vec
    
    def _optimize_condition_vector(
        self,
        init_vec: torch.Tensor,
        pos_tensor: torch.Tensor,
        neg_tensor: torch.Tensor,
        log_interval: int,
        *,
        condition_margin: float,
    ) -> torch.Tensor:
        """
        Optimize condition vector to maximize separation between pos and neg.
        
        Goal: pos examples should have high similarity, neg examples low similarity.
        """
        condition_vec = init_vec.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([condition_vec], lr=self.config.learning_rate)
        
        best_vec = condition_vec.detach().clone()
        best_separation = -float('inf')
        
        for step in range(self.config.optimization_steps):
            optimizer.zero_grad()
            
            # Normalize for cosine similarity
            c_norm = F.normalize(condition_vec, p=2, dim=0)
            pos_norm = F.normalize(pos_tensor, p=2, dim=1)
            neg_norm = F.normalize(neg_tensor, p=2, dim=1)
            
            # Cosine similarities
            pos_sim = (pos_norm * c_norm).sum(dim=1)  # [N_pos]
            neg_sim = (neg_norm * c_norm).sum(dim=1)  # [N_neg]
            
            # Loss: maximize pos_sim, minimize neg_sim
            # Equivalent to minimizing -pos_sim + neg_sim
            margin = condition_margin
            pos_loss = F.relu(margin - pos_sim).mean()
            neg_loss = F.relu(neg_sim + margin).mean()
            
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
            # Track best
            separation = pos_sim.mean() - neg_sim.mean()
            if separation.item() > best_separation:
                best_separation = separation.item()
                best_vec = condition_vec.detach().clone()
            
            if step % log_interval == RECURSION_INITIAL_DEPTH:
                self._training_logs.append({
                    "phase": "condition_opt",
                    "step": step,
                    "loss": loss.item(),
                    "pos_sim_mean": pos_sim.mean().item(),
                    "neg_sim_mean": neg_sim.mean().item(),
                    "separation": separation.item(),
                })
        
        return best_vec
    
