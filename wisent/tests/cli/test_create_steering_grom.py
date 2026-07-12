from types import SimpleNamespace

import pytest
import torch

from wisent.core.utils.cli.commands.steering.core.creation import create_steering_grom


def test_final_metrics_use_restored_nonaliased_best_network_state_under_no_grad(monkeypatch, capsys):
    forward_observations = []

    class RecordingGate(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, inputs, temperature):
            forward_observations.append((self.weight.detach().item(), torch.is_grad_enabled()))
            return inputs[:, 0] * 0 + self.weight

    class MutableIntensity(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(10.0))

    class IncrementingOptimizer:
        def __init__(self, params, *, lr, weight_decay):
            self.params = list(params)

        def zero_grad(self):
            for parameter in self.params:
                parameter.grad = None

        def step(self):
            with torch.no_grad():
                for parameter in self.params:
                    parameter.add_(1.0)

    def loss_that_worsens_after_first_update(
        gate_network,
        sensor_pos,
        sensor_neg,
        gate_temperature,
        gate_threshold,
        layer_order,
        directions,
        direction_weights,
        all_pos,
        all_neg,
        retain_weight,
    ):
        return gate_network.weight.square()

    object_module = __import__(
        "wisent.core.control.steering_methods._steering_object_grom",
        fromlist=["GROMGateNetwork"],
    )
    monkeypatch.setattr(object_module, "GROMGateNetwork", RecordingGate)
    monkeypatch.setattr(object_module, "GROMIntensityNetwork", MutableIntensity)
    monkeypatch.setattr(
        object_module,
        "GROMSteeringObject",
        lambda **attributes: SimpleNamespace(**attributes),
    )
    monkeypatch.setattr(create_steering_grom.torch.optim, "AdamW", IncrementingOptimizer)
    monkeypatch.setattr(create_steering_grom, "_grom_training_step", loss_that_worsens_after_first_update)

    args = SimpleNamespace(
        grom_num_directions=1,
        grom_max_alpha=2.0,
        grom_learning_rate=0.1,
        grom_weight_decay=0.0,
        grom_retain_weight=0.0,
        grom_max_grad_norm=1.0,
        grom_gate_temperature=1.0,
        grom_optimization_steps=2,
        grom_sensor_layer=0,
    )
    activations = {
        "0": {
            "positive": [torch.tensor([1.0, 0.0])],
            "negative": [torch.tensor([-1.0, 0.0])],
        }
    }

    result = create_steering_grom._create_grom_steering_object(
        metadata=SimpleNamespace(hidden_dim=2),
        layer_activations=activations,
        available_layers=["0"],
        args=args,
        log_interval=10,
        gate_dim_min=1,
        gate_dim_max=2,
        gate_dim_divisor=1,
        gate_shrink_factor=1,
        intensity_dim_min=1,
        intensity_dim_max=2,
        intensity_dim_divisor=1,
        create_noise_scale=0.0,
        create_gate_threshold=0.5,
    )

    assert result.gate_network.weight.item() == pytest.approx(1.0)
    assert result.intensity_network.weight.item() == pytest.approx(11.0)
    assert forward_observations == [(1.0, False), (1.0, False)]
    assert "Final gate accuracy: pos=1.000, neg=1.000" in capsys.readouterr().out
