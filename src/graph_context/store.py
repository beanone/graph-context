"""
Graph store factory.

This module provides a factory for creating store instances based on configuration.
"""

import importlib
import json
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Type

from .interfaces.store import GraphStore
from .stores.memory_store import InMemoryGraphStore


@dataclass
class StoreConfig:
    """Internal configuration for graph store."""

    type: str
    config: Dict[str, Any]


class GraphStoreFactory:
    """Factory for creating GraphStore instances from configuration."""

    _store_types: ClassVar[Dict[str, Type[GraphStore]]] = {"memory": InMemoryGraphStore}
    _CONFIG_ENV_VAR: ClassVar[str] = "GRAPH_STORE_CONFIG"
    _CONFIG_FILE_PATH: ClassVar[str] = "graph_store_config.json"

    @classmethod
    def register_store_type(
        cls, store_type: str, store_class: Type[GraphStore]
    ) -> None:
        """Register a new store type."""
        cls._store_types[store_type] = store_class

    @classmethod
    def create(cls) -> GraphStore:
        """Create a GraphStore instance based on internal configuration."""
        config = cls._load_config()

        # Check if store type is registered
        if config.type not in cls._store_types:
            # Try to load dynamically if it looks like a module path
            if "." in config.type:
                try:
                    module_path, class_name = config.type.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    store_class = getattr(module, class_name)
                    if not issubclass(store_class, GraphStore):
                        raise ValueError(
                            f"{class_name} is not a valid GraphStore implementation"
                        )
                    # Register the dynamically loaded class
                    cls.register_store_type(config.type, store_class)
                except (ImportError, AttributeError) as err:
                    raise ValueError(f"Unknown store type: {config.type}") from err
            else:
                raise ValueError(f"Unknown store type: {config.type}")

        return cls._store_types[config.type](config.config)

    @classmethod
    def _load_config(cls) -> StoreConfig:
        """
        Load store configuration from environment/config files.

        Configuration is loaded in the following order (first found wins):
        1. Environment variable GRAPH_STORE_CONFIG (JSON string)
        2. Configuration file graph_store_config.json
        3. Default configuration (memory store)
        """
        # Try environment variable
        config_str = os.getenv(cls._CONFIG_ENV_VAR)
        if config_str:
            try:
                config_dict = json.loads(config_str)
                return StoreConfig(**config_dict)
            except (json.JSONDecodeError, TypeError) as err:
                raise ValueError(f"Invalid environment configuration: {err}") from err

        # Try configuration file
        if os.path.exists(cls._CONFIG_FILE_PATH):
            try:
                with open(cls._CONFIG_FILE_PATH, "r") as f:
                    config_dict = json.load(f)
                return StoreConfig(**config_dict)
            except (json.JSONDecodeError, TypeError, OSError) as err:
                raise ValueError(f"Invalid configuration file: {err}") from err

        # Default to memory store
        return StoreConfig(type="memory", config={})
