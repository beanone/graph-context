"""Cache store implementation for graph operations.

This module provides the core caching functionality for the graph context,
including the cache entry model and storage interface.
"""

from datetime import UTC, datetime
from typing import AsyncIterator, Generic, Optional, Set, Tuple, TypeVar
from uuid import uuid4

from cachetools import TTLCache
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")  # Changed from BaseModel to Any


class CacheEntry(BaseModel, Generic[T]):
    """Cache entry with metadata.

    Attributes:
        value: The cached value (any JSON-serializable value)
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
    """Cache store implementation with TTL support.

    A simple read-only cache that stores computed results with optional TTL.
    """

    def __init__(
        self,
        maxsize: int = 10000,
        ttl: Optional[int] = 300,  # 5 minutes default TTL
    ):
        """Initialize the cache store.

        Args:
            maxsize: Maximum number of entries to store
            ttl: Time-to-live in seconds for cache entries (None for no TTL)
        """
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl) if ttl else {}

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

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Store a cache entry.

        Args:
            key: The cache key
            entry: The entry to store
        """
        self._cache[key] = entry

    async def delete(self, key: str) -> None:
        """Delete a cache entry.

        Args:
            key: The cache key to delete
        """
        try:
            self._cache.pop(key)
        except KeyError:
            pass

    async def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    async def scan(self) -> AsyncIterator[Tuple[str, CacheEntry]]:
        """Iterate over all cache entries.

        Yields:
            Tuples of (key, entry) for each cache entry
        """
        for key, entry in self._cache.items():
            yield key, entry


class DisabledCacheStore(CacheStore):
    """A cache store implementation that does nothing.

    This is used when caching is disabled to avoid conditional logic
    spread throughout the codebase.
    """

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key.

        Always returns None when caching is disabled.
        """
        return None

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Set a cache entry.

        Does nothing when caching is disabled.
        """
        pass

    async def delete(self, key: str) -> None:
        """Delete a cache entry.

        Does nothing when caching is disabled.
        """
        pass

    async def clear(self) -> None:
        """Clear all entries in the cache.

        Does nothing when caching is disabled.
        """
        pass
