"""Configuration module for the cache system.

This module provides configuration and metrics classes for the cache system.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CacheMetrics:
    """Cache metrics tracking."""

    hits: int = 0
    misses: int = 0
    total_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary.

        Returns:
            Dictionary containing the metrics
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        avg_time = self.total_time / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": hit_rate,
            "total_time": self.total_time,
            "avg_time": avg_time,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CacheConfig:
    """Configuration for the cache system."""
    # Default cache settings (for backward compatibility)
    max_size: int = 1000
    default_ttl: float = 300.0  # 5 minutes in seconds

    # Entity cache settings
    entity_cache_size: int = field(default=1000)
    entity_cache_ttl: float = field(default=300.0)  # 5 minutes

    # Relation cache settings
    relation_cache_size: int = field(default=1000)
    relation_cache_ttl: float = field(default=300.0)  # 5 minutes

    # Query cache settings
    query_cache_size: int = field(default=500)
    query_cache_ttl: float = field(default=60.0)  # 1 minute

    # Traversal cache settings
    traversal_cache_size: int = field(default=500)
    traversal_cache_ttl: float = field(default=60.0)  # 1 minute

    # Feature flags
    enable_metrics: bool = field(default=True)

    def get_ttl_for_type(self, type_name: str) -> Optional[int]:
        """Get the TTL for a specific type.

        Args:
            type_name: The type name to get TTL for

        Returns:
            TTL in seconds or None if not set
        """
        if type_name == "entity":
            return self.entity_cache_ttl
        elif type_name == "relation":
            return self.relation_cache_ttl
        elif type_name == "query":
            return self.query_cache_ttl
        elif type_name == "traversal":
            return self.traversal_cache_ttl
        else:
            return self.default_ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Dictionary containing the configuration
        """
        return {
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
            "enable_metrics": self.enable_metrics,
            "entity_cache_size": self.entity_cache_size,
            "entity_cache_ttl": self.entity_cache_ttl,
            "relation_cache_size": self.relation_cache_size,
            "relation_cache_ttl": self.relation_cache_ttl,
            "query_cache_size": self.query_cache_size,
            "query_cache_ttl": self.query_cache_ttl,
            "traversal_cache_size": self.traversal_cache_size,
            "traversal_cache_ttl": self.traversal_cache_ttl,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "CacheConfig":
        """Create a configuration from a dictionary.

        Args:
            config: Dictionary containing configuration values

        Returns:
            New CacheConfig instance
        """
        return cls(
            max_size=config.get("max_size", 1000),
            default_ttl=config.get("default_ttl", 300.0),
            enable_metrics=config.get("enable_metrics", True),
            entity_cache_size=config.get("entity_cache_size", 1000),
            entity_cache_ttl=config.get("entity_cache_ttl", 300.0),
            relation_cache_size=config.get("relation_cache_size", 1000),
            relation_cache_ttl=config.get("relation_cache_ttl", 300.0),
            query_cache_size=config.get("query_cache_size", 500),
            query_cache_ttl=config.get("query_cache_ttl", 60.0),
            traversal_cache_size=config.get("traversal_cache_size", 500),
            traversal_cache_ttl=config.get("traversal_cache_ttl", 60.0),
        )