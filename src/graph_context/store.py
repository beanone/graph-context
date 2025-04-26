"""
Graph store factory.

This module provides a factory for creating store instances based on configuration.
"""
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Type

from .interfaces.store import GraphStore
from .stores.memory_store import InMemoryGraphStore


@dataclass
class StoreConfig:
    """Internal configuration for graph store."""
    type: str
    config: Dict[str, Any]


class GraphStoreFactory:
    """Factory for creating GraphStore instances from configuration."""

    _store_types: Dict[str, Type[GraphStore]] = {
        "memory": InMemoryGraphStore
    }
    _CONFIG_ENV_VAR = "GRAPH_STORE_CONFIG"
    _CONFIG_FILE_PATH = "graph_store_config.json"

    @classmethod
    def register_store_type(cls, store_type: str, store_class: Type[GraphStore]) -> None:
        """Register a new store type."""
        cls._store_types[store_type] = store_class

    @classmethod
    def create(cls) -> GraphStore:
        """Create a GraphStore instance based on internal configuration."""
        config = cls._load_config()
        if config.type not in cls._store_types:
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
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"Invalid environment configuration: {e}")

        # Try configuration file
        if os.path.exists(cls._CONFIG_FILE_PATH):
            try:
                with open(cls._CONFIG_FILE_PATH, 'r') as f:
                    config_dict = json.load(f)
                return StoreConfig(**config_dict)
            except (json.JSONDecodeError, TypeError, OSError) as e:
                raise ValueError(f"Invalid configuration file: {e}")

        # Default to memory store
        return StoreConfig(
            type="memory",
            config={}
        )