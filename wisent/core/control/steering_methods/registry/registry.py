"""
Centralized Steering Method Registry.

Single source of truth for all steering method definitions.
See _definitions_*.py for method-specific definitions.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from wisent.core.control.steering_methods.core.atoms import BaseSteeringMethod

logger = logging.getLogger(__name__)

# Sentinel for "no default provided"
_UNSET = object()


class SteeringMethodType(Enum):
    """Enumeration of all supported steering methods."""
    CAA = "caa"
    OSTRZE = "ostrze"
    MLP = "mlp"
    TECZA = "tecza"
    TETNO = "tetno"
    GROM = "grom"
    NURT = "nurt"
    SZLAK = "szlak"
    WICHER = "wicher"
    PRZELOM = "przelom"


@dataclass
class SteeringMethodParameter:
    """Definition of a single parameter for a steering method."""
    name: str
    type: Type
    help: str
    default: Any = _UNSET
    required: bool = True
    cli_flag: Optional[str] = None
    cli_dest: Optional[str] = None
    choices: Optional[List[Any]] = None
    action: Optional[str] = None

    @property
    def has_default(self) -> bool:
        """Whether this parameter has an explicit default value."""
        return self.default is not _UNSET

    def get_cli_flag(self, method_name: str) -> str:
        """Get the CLI flag name for this parameter."""
        if self.cli_flag:
            return self.cli_flag
        return f"--{method_name}-{self.name.replace('_', '-')}"

    def get_cli_dest(self, method_name: str) -> str:
        """Get the argparse namespace destination for this parameter."""
        if self.cli_dest:
            return self.cli_dest
        return self.get_cli_flag(method_name).lstrip("-").replace("-", "_")


@dataclass
class SteeringMethodDefinition:
    """Complete definition of a steering method."""
    name: str
    method_type: SteeringMethodType
    description: str
    method_class_path: str
    parameters: List[SteeringMethodParameter] = field(default_factory=list)
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    default_strength: Optional[float] = None
    strength_range: Optional[tuple] = None

    def get_method_class(self) -> Type[BaseSteeringMethod]:
        """Dynamically import and return the method class."""
        module_path, class_name = self.method_class_path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get parameter defaults defined directly on parameters."""
        return {p.name: p.default for p in self.parameters if p.has_default}

    def get_required_param_names(self) -> List[str]:
        """Get names of required params that lack inline defaults."""
        return [p.name for p in self.parameters if p.required and not p.has_default]

    def add_cli_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add this method's CLI arguments to a parser."""
        for param in self.parameters:
            flag = param.get_cli_flag(self.name)
            kwargs: Dict[str, Any] = {
                "default": param.default if param.has_default else None,
                "help": f"[{self.name.upper()}] {param.help}",
            }
            if param.cli_dest:
                kwargs["dest"] = param.cli_dest
            if param.action:
                kwargs["action"] = param.action
            else:
                kwargs["type"] = param.type
            if param.choices:
                kwargs["choices"] = param.choices
            parser.add_argument(flag, **kwargs)

    def extract_params_from_args(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract this method's parameters from parsed CLI args."""
        params = {}
        for param in self.parameters:
            attr_name = param.get_cli_dest(self.name)
            if hasattr(args, attr_name):
                val = getattr(args, attr_name)
                if val is not None:
                    params[param.name] = val
            elif param.has_default:
                params[param.name] = param.default
        return params


# =============================================================================
# STEERING METHOD DEFINITIONS (imported from sibling modules)
# =============================================================================
from wisent.core.control.steering_methods._definitions_caa_ostrze_tecza import (
    CAA_DEFINITION,
    OSTRZE_DEFINITION,
    TECZA_DEFINITION,
)
from wisent.core.control.steering_methods._definitions_tetno_grom import (
    TETNO_DEFINITION,
    GROM_DEFINITION,
    PRZELOM_DEFINITION,
)
from wisent.core.control.steering_methods._definitions_mlp_nurt_szlak_wicher import (
    MLP_DEFINITION,
    NURT_DEFINITION,
    SZLAK_DEFINITION,
    WICHER_DEFINITION,
)


# =============================================================================
# REGISTRY CLASS
# =============================================================================

class SteeringMethodRegistry:
    """
    Central registry for all steering methods.
    
    Usage:
        # Get all registered methods
        methods = SteeringMethodRegistry.list_methods()
        
        # Get a specific method definition
        caa = SteeringMethodRegistry.get("caa")
        
        # Add CLI arguments for all methods
        SteeringMethodRegistry.add_all_cli_arguments(parser)
        
        # Get method class
        method_class = SteeringMethodRegistry.get_method_class("caa")
    """
    
    _REGISTRY: Dict[str, SteeringMethodDefinition] = {
        "caa": CAA_DEFINITION,
        "ostrze": OSTRZE_DEFINITION,
        "mlp": MLP_DEFINITION,
        "tecza": TECZA_DEFINITION,
        "tetno": TETNO_DEFINITION,
        "grom": GROM_DEFINITION,
        "nurt": NURT_DEFINITION,
        "szlak": SZLAK_DEFINITION,
        "wicher": WICHER_DEFINITION,
        "przelom": PRZELOM_DEFINITION,
    }
    
    @classmethod
    def register(cls, definition: SteeringMethodDefinition) -> None:
        """Register a new steering method definition."""
        cls._REGISTRY[definition.name] = definition
    
    @classmethod
    def get(cls, name: str) -> SteeringMethodDefinition:
        """Get a steering method definition by name."""
        name_lower = name.lower()
        if name_lower not in cls._REGISTRY:
            available = ", ".join(cls._REGISTRY.keys())
            raise ValueError(f"Unknown steering method: {name}. Available: {available}")
        return cls._REGISTRY[name_lower]
    
    @classmethod
    def list_methods(cls) -> List[str]:
        """List all registered method names."""
        return list(cls._REGISTRY.keys())
    
    @classmethod
    def list_definitions(cls) -> List[SteeringMethodDefinition]:
        """List all registered method definitions."""
        return list(cls._REGISTRY.values())
    
    @classmethod
    def get_method_class(cls, name: str) -> Type[BaseSteeringMethod]:
        """Get the method class for a given method name."""
        return cls.get(name).get_method_class()
    
    @classmethod
    def get_default_method(cls) -> str:
        """Get the default steering method name."""
        return "caa"
    
    @classmethod
    def add_all_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI arguments for all registered steering methods."""
        # Add method selection argument
        parser.add_argument(
            "--steering-method",
            type=str,
            default=cls.get_default_method(),
            choices=[m.upper() for m in cls.list_methods()] + [m.lower() for m in cls.list_methods()],
            help=f"Steering method to use (default: {cls.get_default_method().upper()})",
        )
        
        # Add method-specific arguments
        for definition in cls.list_definitions():
            definition.add_cli_arguments(parser)
    
    @classmethod
    def extract_method_params(cls, method_name: str, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract method-specific parameters from parsed CLI args."""
        return cls.get(method_name).extract_params_from_args(args)
    
    @classmethod
    def create_method_instance(
        cls, name: str, model_name: Optional[str] = None,
        task_name: Optional[str] = None, **kwargs,
    ) -> BaseSteeringMethod:
        """Create a method instance using three-layer config merge."""
        from wisent.core.control.steering_methods.registry.registry_instantiation import (
            build_merged_params,
        )
        definition = cls.get(name)
        method_class = definition.get_method_class()
        params = build_merged_params(definition, model_name, task_name, kwargs)
        return method_class(**params)
    
    @classmethod
    def get_optimization_config(cls, name: str) -> Dict[str, Any]:
        """Get optimization configuration for a method."""
        return cls.get(name).optimization_config
    
    @classmethod
    def get_strength_range(cls, name: str) -> tuple:
        """Get the strength search range for a method."""
        defn = cls.get(name)
        if defn.strength_range is None:
            raise ValueError(
                f"No strength_range defined for method '{name}'. "
                f"Specify --strength-range via CLI."
            )
        return defn.strength_range
    
    @classmethod
    def get_default_strength(cls, name: str) -> float:
        """Get the default strength for a method."""
        return cls.get(name).default_strength
    
    @classmethod
    def validate_method(cls, name: str) -> bool:
        """Check if a method name is valid."""
        return name.lower() in cls._REGISTRY
    
    @classmethod
    def get_method_info(cls) -> List[Dict[str, Any]]:
        """Get detailed info about all methods for documentation/help."""
        from wisent.core.control.steering_methods.registry.registry_instantiation import (
            get_method_info_impl,
        )
        return get_method_info_impl(cls.list_definitions())


def get_steering_method(
    name: str, model_name: Optional[str] = None,
    task_name: Optional[str] = None, **kwargs,
) -> BaseSteeringMethod:
    """Create a steering method instance by name."""
    return SteeringMethodRegistry.create_method_instance(
        name, model_name=model_name, task_name=task_name, **kwargs,
    )


def list_steering_methods() -> List[str]:
    return SteeringMethodRegistry.list_methods()


def is_valid_steering_method(name: str) -> bool:
    return SteeringMethodRegistry.validate_method(name)
