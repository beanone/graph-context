"""Cache store implementation for graph operations.

This module provides the core caching functionality for the graph context,
including the cache entry model and storage interface.
"""

from typing import (TypeVar, Generic, Optional,
                    AsyncIterator, Tuple,
                    Set, Dict)
from datetime import datetime, UTC
from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4
from cachetools import TTLCache
from collections import defaultdict

T = TypeVar('T', bound=BaseModel)

class CacheEntry(BaseModel, Generic[T]):
    """Cache entry with metadata.

    Attributes:
        value: The cached value
        created_at: When the entry was created
        entity_type: Type name for entity entries
        relation_type: Type name for relation entries
        operation_id: Unique identifier for the operation that created this entry
        query_hash: Hash of the query that produced this result (for query results)
        dependencies: Set of entity/relation IDs this entry depends on
    """
    value: T
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    entity_type: Optional[str] = None
    relation_type: Optional[str] = None
    operation_id: str = Field(default_factory=lambda: str(uuid4()))
    query_hash: Optional[str] = None
    dependencies: Set[str] = Field(default_factory=set)

    model_config = ConfigDict(frozen=True)

class CacheStore:
    """Cache store implementation with type awareness and TTL support."""

    def __init__(
        self,
        maxsize: int = 10000,
        ttl: Optional[int] = 300  # 5 minutes default TTL
    ):
        """Initialize the cache store.

        Args:
            maxsize: Maximum number of entries to store
            ttl: Time-to-live in seconds for cache entries (None for no TTL)
        """
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl) if ttl else {}
        self._type_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._query_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve a cache entry by key.

        Args:
            key: The cache key to retrieve

        Returns:
            The cache entry if found and not expired, None otherwise
        """
        try:
            return self._cache[key]
        except KeyError:
            return None

    async def set(
        self,
        key: str,
        entry: CacheEntry,
        dependencies: Optional[Set[str]] = None
    ) -> None:
        """Store a cache entry.

        Args:
            key: The cache key
            entry: The entry to store
            dependencies: Optional set of keys this entry depends on
        """
        self._cache[key] = entry

        # Track type dependencies
        if entry.entity_type:
            self._type_dependencies[entry.entity_type].add(key)
        if entry.relation_type:
            self._type_dependencies[entry.relation_type].add(key)

        # Track query dependencies
        if entry.query_hash:
            self._query_dependencies[entry.query_hash].add(key)

        # Track reverse dependencies
        if dependencies:
            for dep in dependencies:
                self._reverse_dependencies[dep].add(key)

    async def delete(self, key: str) -> None:
        """Delete a cache entry.

        Args:
            key: The cache key to delete
        """
        try:
            entry = self._cache.pop(key)

            # Clean up type dependencies
            if entry.entity_type:
                self._type_dependencies[entry.entity_type].discard(key)
            if entry.relation_type:
                self._type_dependencies[entry.relation_type].discard(key)

            # Clean up query dependencies
            if entry.query_hash:
                self._query_dependencies[entry.query_hash].discard(key)

            # Clean up reverse dependencies
            for dep_key in self._reverse_dependencies:
                self._reverse_dependencies[dep_key].discard(key)

        except KeyError:
            pass

    async def delete_many(self, keys: Set[str]) -> None:
        """Delete multiple cache entries efficiently.

        Args:
            keys: Set of cache keys to delete
        """
        # Create a copy of the keys to avoid mutation during iteration
        keys_to_delete = set(keys)
        for key in keys_to_delete:
            await self.delete(key)

    async def scan(self) -> AsyncIterator[Tuple[str, CacheEntry]]:
        """Iterate over all cache entries.

        Yields:
            Tuples of (key, entry) for each cache entry
        """
        for key, entry in self._cache.items():
            yield key, entry

    async def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._type_dependencies.clear()
        self._query_dependencies.clear()
        self._reverse_dependencies.clear()

    async def invalidate_type(self, type_name: str) -> None:
        """Invalidate all cache entries for a type.

        Args:
            type_name: The type name to invalidate
        """
        keys = self._type_dependencies.get(type_name, set())
        await self.delete_many(keys)
        self._type_dependencies[type_name].clear()

    async def invalidate_query(self, query_hash: str) -> None:
        """Invalidate all cache entries for a query.

        Args:
            query_hash: The query hash to invalidate
        """
        keys = self._query_dependencies.get(query_hash, set())
        await self.delete_many(keys)
        self._query_dependencies[query_hash].clear()

    async def invalidate_dependencies(self, key: str) -> None:
        """Invalidate all cache entries that depend on a key.

        Args:
            key: The key whose dependents should be invalidated
        """
        dependent_keys = self._reverse_dependencies.get(key, set())
        await self.delete_many(dependent_keys)
        self._reverse_dependencies[key].clear()