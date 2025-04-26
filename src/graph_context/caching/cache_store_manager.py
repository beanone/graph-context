"""Cache store manager for managing different types of caches.

This module provides the CacheStoreManager class that manages separate cache stores
for different types of data (entities, relations, queries, traversals).
"""

from .cache_store import CacheStore
from .config import CacheConfig


class CacheStoreManager:
    """Manages separate cache stores for different types of data."""

    def __init__(self, config: CacheConfig):
        """Initialize the cache store manager.

        Args:
            config: Cache configuration containing sizes and TTLs for different stores
        """
        self.entity_store = CacheStore(maxsize=config.entity_cache_size, ttl=config.entity_cache_ttl)
        self.relation_store = CacheStore(maxsize=config.relation_cache_size, ttl=config.relation_cache_ttl)
        self.query_store = CacheStore(maxsize=config.query_cache_size, ttl=config.query_cache_ttl)
        self.traversal_store = CacheStore(maxsize=config.traversal_cache_size, ttl=config.traversal_cache_ttl)

    def get_entity_store(self) -> CacheStore:
        """Get the entity cache store.

        Returns:
            The entity cache store
        """
        return self.entity_store

    def get_relation_store(self) -> CacheStore:
        """Get the relation cache store.

        Returns:
            The relation cache store
        """
        return self.relation_store

    def get_query_store(self) -> CacheStore:
        """Get the query cache store.

        Returns:
            The query cache store
        """
        return self.query_store

    def get_traversal_store(self) -> CacheStore:
        """Get the traversal cache store.

        Returns:
            The traversal cache store
        """
        return self.traversal_store

    async def clear_all(self) -> None:
        """Clear all cache stores."""
        await self.entity_store.clear()
        await self.relation_store.clear()
        await self.query_store.clear()
        await self.traversal_store.clear()
