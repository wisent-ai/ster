"""GROM model loading functionality."""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.utils.infra_tools.errors import InsufficientDataError
from wisent.core.weight_modification.export._generic import load_steered_model

_LOG = setup_logger(__name__)

def load_grom_model(
    model_path: str | Path,
    device_map: Optional[str] = None,
    torch_dtype=None,
    install_hooks: bool = True,
):
    """
    Load a GROM-steered model with optional runtime hooks.
    
    This loads the model and optionally installs GROM runtime hooks
    for dynamic steering based on input content.
    
    Args:
        model_path: Path to saved GROM model
        device_map: Device map for model loading
        torch_dtype: Torch dtype for model weights
        install_hooks: Whether to install runtime hooks (default: True)
        
    Returns:
        Tuple of (model, tokenizer, hooks) where hooks is None if install_hooks=False
        
    Example:
        >>> model, tokenizer, hooks = load_grom_model("./my_grom_model")
        >>> 
        >>> # Generate with dynamic steering
        >>> output = model.generate(...)
        >>> 
        >>> # Check gate value
        >>> print(f"Gate: {hooks.get_current_gate()}")
        >>> 
        >>> # Remove hooks when done
        >>> hooks.remove()
    """
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download, HfFileSystem
    
    model_path_str = str(model_path)
    log = bind(_LOG, model_path=model_path_str)
    
    # Check if this is a local path or HuggingFace repo
    is_local = Path(model_path_str).exists()
    
    if is_local:
        model_path = Path(model_path_str)
        grom_config_path = model_path / "grom_config.json"
        grom_data_path = model_path / "grom_steering.pt"
        config_exists = grom_config_path.exists()
    else:
        # HuggingFace Hub repo - check if grom files exist
        try:
            fs = HfFileSystem()
            files = fs.ls(model_path_str, detail=False)
            file_names = [f.split("/")[-1] for f in files]
            config_exists = "grom_config.json" in file_names
        except Exception:
            config_exists = False
    
    if not config_exists:
        log.warning("No grom_config.json found - loading as regular model")
        if is_local:
            return load_steered_model(model_path_str, device_map, torch_dtype) + (None,)
        else:
            # Load from HF without GROM
            model = AutoModelForCausalLM.from_pretrained(model_path_str, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path_str)
            return model, tokenizer, None
    
    # Load GROM config
    if is_local:
        with open(grom_config_path) as f:
            grom_config = json.load(f)
    else:
        # Download from HF Hub
        config_file = hf_hub_download(repo_id=model_path_str, filename="grom_config.json")
        with open(config_file) as f:
            grom_config = json.load(f)
    
    mode = grom_config.get("mode", "hybrid")
    log.info(f"Loading GROM model (mode={mode})")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path_str,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)
    
    # Load GROM data
    hooks = None
    
    # Get grom_steering.pt path (local or download from HF)
    if is_local:
        grom_data_path = Path(model_path_str) / "grom_steering.pt"
        grom_data_exists = grom_data_path.exists()
    else:
        try:
            grom_data_path = hf_hub_download(repo_id=model_path_str, filename="grom_steering.pt")
            grom_data_exists = True
        except Exception:
            grom_data_exists = False
    
    if install_hooks and mode in ("dynamic", "hybrid") and grom_data_exists:
        grom_data = torch.load(grom_data_path, map_location="cpu", weights_only=False)
        
        # Reconstruct GROMResult-like object for hooks
        from wisent.core.weight_modification.directional import GROMRuntimeHooks
        from types import SimpleNamespace
        
        # Create a minimal grom_result object with required attributes
        grom_result = SimpleNamespace()
        grom_result.layer_order = grom_data["layer_order"]
        grom_result.directions = {k: v.to(model.device) for k, v in grom_data["directions"].items()}
        grom_result.direction_weights = {k: v.to(model.device) for k, v in grom_data["direction_weights"].items()}
        if "gate_temperature" not in grom_data:
            raise InsufficientDataError(
                reason="gate_temperature missing from saved GROM data. Re-bake the model with current wisent version."
            )
        grom_result.gate_temperature = grom_data["gate_temperature"]
        if "max_alpha" not in grom_data:
            raise InsufficientDataError(
                reason="max_alpha missing from saved GROM data. Re-bake the model with current wisent version."
            )
        grom_result.max_alpha = grom_data["max_alpha"]
        
        # Reconstruct gate network
        if "gate_network_state" in grom_data:
            from wisent.core.control.steering_methods.methods.grom import GatingNetwork
            config = grom_data["gate_network_config"]
            if "hidden_dim" not in config:
                raise InsufficientDataError(
                    reason="hidden_dim missing from gate_network_config. Re-bake the model."
                )
            if "shrink_factor" not in config:
                raise InsufficientDataError(
                    reason="shrink_factor missing from gate_network_config. Re-bake the model."
                )
            grom_result.gate_network = GatingNetwork(
                config["input_dim"],
                config["hidden_dim"],
                shrink_factor=config["shrink_factor"],
            )
            grom_result.gate_network.load_state_dict(grom_data["gate_network_state"])
            grom_result.gate_network = grom_result.gate_network.to(model.device)
        else:
            grom_result.gate_network = None
        
        # Reconstruct intensity network
        if "intensity_network_state" in grom_data:
            from wisent.core.control.steering_methods.methods.grom import IntensityNetwork
            config = grom_data["intensity_network_config"]
            if "hidden_dim" not in config:
                raise InsufficientDataError(
                    reason="hidden_dim missing from intensity_network_config. Re-bake the model."
                )
            grom_result.intensity_network = IntensityNetwork(
                config["input_dim"],
                config["num_layers"],
                config["hidden_dim"],
            )
            grom_result.intensity_network.load_state_dict(grom_data["intensity_network_state"])
            grom_result.intensity_network = grom_result.intensity_network.to(model.device)
        
        # Add required methods
        def get_effective_direction(layer_name):
            dirs = grom_result.directions.get(layer_name)
            weights = grom_result.direction_weights.get(layer_name)
            if dirs is None:
                return None
            if weights is not None:
                return (dirs * weights.unsqueeze(-1)).sum(0)
            return dirs.mean(0)
        
        def predict_gate(hidden_state):
            if grom_result.gate_network is not None:
                return grom_result.gate_network(hidden_state.float(), grom_result.gate_temperature).to(hidden_state.dtype)
            return torch.ones(hidden_state.shape[0], device=hidden_state.device, dtype=hidden_state.dtype)
        
        def predict_intensity(hidden_state):
            if grom_result.intensity_network is not None:
                intensities = grom_result.intensity_network(hidden_state.float())
                return {layer: intensities[:, i] for i, layer in enumerate(grom_result.layer_order)}
            return {layer: torch.ones(1, device=hidden_state.device) for layer in grom_result.layer_order}
        
        grom_result.get_effective_direction = get_effective_direction
        grom_result.predict_gate = predict_gate
        grom_result.predict_intensity = predict_intensity
        
        # Sensor identity is independent from direction-bearing layer_order.
        if "sensor_layer" not in grom_data:
            raise InsufficientDataError(
                reason="sensor_layer missing from saved GROM data. Re-export the model."
            )
        sensor_layer = grom_data["sensor_layer"]
        grom_result.metadata = {"sensor_layer": sensor_layer}
        grom_result.sensor_layer = sensor_layer
        
        # Install hooks
        hooks = GROMRuntimeHooks(
            model=model,
            grom_result=grom_result,
            base_strength=1.0,
            use_soft_gating=True,
        )
        hooks.install()
        log.info(f"Installed GROM runtime hooks (sensor_layer={grom_result.sensor_layer})")
    
    return model, tokenizer, hooks


