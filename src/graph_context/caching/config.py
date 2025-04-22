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


class CacheConfig:
    """Configuration for the cache system."""

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: Optional[int] = None,
        enable_metrics: bool = True,
        type_ttls: Optional[Dict[str, int]] = None,
    ):
        """Initialize cache configuration.

        Args:
            max_size: Maximum number of entries in the cache
            default_ttl: Default time-to-live in seconds for cache entries
            enable_metrics: Whether to enable metrics tracking
            type_ttls: Optional mapping of type names to TTL values
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.enable_metrics = enable_metrics
        self.type_ttls = type_ttls or {}

        # Set default TTLs for different operation types
        if not self.type_ttls:
            self.type_ttls = {
                # Entities and relations have a longer TTL since they change less frequently
                "entity": 3600,  # 1 hour
                "relation": 3600,  # 1 hour
                # Queries and traversals have a shorter TTL since they depend on multiple items
                "query": 300,  # 5 minutes
                "traversal": 300,  # 5 minutes
            }

    def get_ttl_for_type(self, type_name: str) -> Optional[int]:
        """Get the TTL for a specific type.

        Args:
            type_name: The type name to get TTL for

        Returns:
            TTL in seconds or None if not set
        """
        return self.type_ttls.get(type_name, self.default_ttl)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Dictionary containing the configuration
        """
        return {
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
            "enable_metrics": self.enable_metrics,
            "type_ttls": self.type_ttls,
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
            max_size=config.get("max_size", 10000),
            default_ttl=config.get("default_ttl"),
            enable_metrics=config.get("enable_metrics", True),
            type_ttls=config.get("type_ttls"),
        )