"""Cache store implementation for graph operations.

This module provides the core caching functionality for the graph context,
including the cache entry model and storage interface.
"""

from typing import (TypeVar, Generic, Optional,
                    AsyncIterator, Tuple,
                    Set, Dict, Any)
from datetime import datetime, UTC
from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4
from cachetools import TTLCache
from collections import defaultdict

T = TypeVar('T')  # Changed from BaseModel to Any

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
        self._entity_relations: Dict[str, Set[str]] = defaultdict(set)
        self._relation_entities: Dict[str, Set[str]] = defaultdict(set)

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

        # For queries and traversals, track dependencies on affected types
        if dependencies and (key.startswith("query:") or key.startswith("traversal:")):
            for type_name in dependencies:
                self._type_dependencies[type_name].add(key)

        # Track query dependencies
        if entry.query_hash:
            self._query_dependencies[entry.query_hash].add(key)

        # Track reverse dependencies
        if dependencies:
            for dep in dependencies:
                self._reverse_dependencies[dep].add(key)

        # Track entity-relation dependencies
        if key.startswith("relation:"):
            # Extract entity IDs from relation value if available
            if hasattr(entry.value, "from_entity"):
                from_entity = entry.value.from_entity
                self._entity_relations[from_entity].add(key)
                self._relation_entities[key].add(from_entity)
            if hasattr(entry.value, "to_entity"):
                to_entity = entry.value.to_entity
                self._entity_relations[to_entity].add(key)
                self._relation_entities[key].add(to_entity)

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

            # Clean up entity-relation dependencies
            if key.startswith("relation:"):
                for entity_id in self._relation_entities.get(key, set()):
                    self._entity_relations[entity_id].discard(key)
                self._relation_entities.pop(key, None)
            elif key.startswith("entity:"):
                # When deleting an entity, also delete its relations
                for relation_key in self._entity_relations.get(key, set()):
                    await self.delete(relation_key)
                self._entity_relations.pop(key, None)

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
        self._entity_relations.clear()
        self._relation_entities.clear()

    async def invalidate_type(self, type_name: str) -> None:
        """Invalidate all cache entries for a type.

        Args:
            type_name: The type name to invalidate
        """
        # Get all keys to invalidate
        keys = self._type_dependencies.get(type_name, set())
        if not keys:
            return

        # Create a copy of the keys to avoid mutation during iteration
        keys_to_delete = set(keys)

        # Delete all affected entries
        for key in keys_to_delete:
            try:
                # Get the entry before deleting it
                entry = self._cache[key]

                # Delete from main cache
                del self._cache[key]

                # Clean up query dependencies
                if entry.query_hash:
                    self._query_dependencies[entry.query_hash].discard(key)

                # Clean up reverse dependencies
                for dep_key in self._reverse_dependencies:
                    self._reverse_dependencies[dep_key].discard(key)

                # Clean up entity-relation dependencies
                if key.startswith("relation:"):
                    for entity_id in self._relation_entities.get(key, set()):
                        self._entity_relations[entity_id].discard(key)
                    self._relation_entities.pop(key, None)
                elif key.startswith("entity:"):
                    # When deleting an entity, also delete its relations
                    for relation_key in self._entity_relations.get(key, set()):
                        await self.delete(relation_key)
                    self._entity_relations.pop(key, None)

                # If this is a query or traversal entry, also invalidate any entries that depend on it
                if key.startswith(("query:", "traversal:")):
                    await self.invalidate_dependencies(key)

            except KeyError:
                pass

        # Clear the type dependencies
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

        This includes:
        1. Direct dependencies (entries that listed this key as a dependency)
        2. Related entries (e.g. relations when invalidating an entity)
        3. Queries that depend on the affected types

        Args:
            key: The key whose dependents should be invalidated
        """
        # Get all keys that need to be invalidated
        keys_to_invalidate = set()

        # Add direct dependencies
        keys_to_invalidate.update(self._reverse_dependencies.get(key, set()))

        # Add related entries
        if key.startswith("entity:"):
            # When invalidating an entity, also invalidate its relations
            keys_to_invalidate.update(self._entity_relations.get(key, set()))
        elif key.startswith("relation:"):
            # When invalidating a relation, also invalidate related entity caches
            for entity_id in self._relation_entities.get(key, set()):
                keys_to_invalidate.update(self._reverse_dependencies.get(entity_id, set()))

        # Add affected query results
        entry = await self.get(key)
        if entry:
            if entry.entity_type:
                keys_to_invalidate.update(self._type_dependencies.get(entry.entity_type, set()))
            if entry.relation_type:
                keys_to_invalidate.update(self._type_dependencies.get(entry.relation_type, set()))

        # Delete all affected entries
        await self.delete_many(keys_to_invalidate)

        # Clean up dependency tracking
        self._reverse_dependencies[key].clear()
        if key.startswith("entity:"):
            self._entity_relations.pop(key, None)
        elif key.startswith("relation:"):
            self._relation_entities.pop(key, None)