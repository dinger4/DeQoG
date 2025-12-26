"""
DeQoG Configuration Module

Handles loading and managing configuration for the DeQoG framework.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class LLMConfig:
    """LLM configuration."""
    model_name: str = "gpt-4"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class HILEConfig:
    """HILE algorithm configuration."""
    num_thoughts: int = 5
    num_solutions: int = 3
    num_implementations: int = 2


@dataclass
class IRQNConfig:
    """IRQN algorithm configuration."""
    p_qn1: float = 0.7
    p_qn2: float = 0.3
    max_iterations: int = 5
    theta_diff: float = 0.3
    theta_ident: float = 0.7


@dataclass
class DiversityConfig:
    """Diversity configuration."""
    threshold: float = 0.6
    hile: HILEConfig = field(default_factory=HILEConfig)
    irqn: IRQNConfig = field(default_factory=IRQNConfig)


@dataclass
class QualityConfig:
    """Quality assurance configuration."""
    threshold: float = 0.9
    max_refinement_iterations: int = 5
    test_timeout: int = 5


@dataclass
class FSMConfig:
    """FSM configuration."""
    max_retries: int = 3
    enable_rollback: bool = True
    decision_temperature: float = 0.1


@dataclass
class NVersionConfig:
    """N-version configuration."""
    default_n: int = 5
    min_n: int = 3
    max_n: int = 10


class Config:
    """
    Configuration manager for DeQoG.
    
    Handles loading configuration from YAML files and environment variables,
    with support for nested configuration access.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict or {}
        self._resolve_env_vars()
        self._init_sub_configs()
    
    def _resolve_env_vars(self):
        """Resolve environment variables in configuration values."""
        def resolve(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.environ.get(env_var, "")
            elif isinstance(value, dict):
                return {k: resolve(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve(v) for v in value]
            return value
        
        self._config = resolve(self._config)
    
    def _init_sub_configs(self):
        """Initialize sub-configuration objects."""
        # LLM Config
        llm_dict = self._config.get("llm", {})
        self.llm = LLMConfig(**{k: v for k, v in llm_dict.items() 
                                if k in LLMConfig.__dataclass_fields__})
        
        # Diversity Config
        div_dict = self._config.get("diversity", {})
        hile_dict = div_dict.get("hile", {})
        irqn_dict = div_dict.get("irqn", {})
        
        self.diversity = DiversityConfig(
            threshold=div_dict.get("threshold", 0.6),
            hile=HILEConfig(**{k: v for k, v in hile_dict.items() 
                              if k in HILEConfig.__dataclass_fields__}),
            irqn=IRQNConfig(**{k: v for k, v in irqn_dict.items() 
                              if k in IRQNConfig.__dataclass_fields__})
        )
        
        # Quality Config
        qual_dict = self._config.get("quality", {})
        self.quality = QualityConfig(**{k: v for k, v in qual_dict.items() 
                                        if k in QualityConfig.__dataclass_fields__})
        
        # FSM Config
        fsm_dict = self._config.get("fsm", {})
        self.fsm = FSMConfig(**{k: v for k, v in fsm_dict.items() 
                                if k in FSMConfig.__dataclass_fields__})
        
        # N-Version Config
        nv_dict = self._config.get("n_versions", {})
        self.n_versions = NVersionConfig(**{k: v for k, v in nv_dict.items() 
                                            if k in NVersionConfig.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            Config instance
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        return cls(deepcopy(config_dict))
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Supports dot notation for nested keys (e.g., "llm.model_name").
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._init_sub_configs()  # Reinitialize sub-configs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return deepcopy(self._config)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Set configuration value using bracket notation."""
        self.set(key, value)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            Path("configs/default_config.yaml"),
            Path("config.yaml"),
            Path.home() / ".deqog" / "config.yaml",
        ]
        
        for path in default_paths:
            if path.exists():
                return Config.from_yaml(path)
        
        # Return empty config with defaults
        return Config({})
    
    return Config.from_yaml(config_path)

