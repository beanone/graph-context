"""Cache manager for the graph context.

This module provides the CacheManager class that handles caching operations
and event handling for the graph context.
"""

import time
from typing import Any, Dict, Optional, Set
from datetime import datetime

from graph_context.event_system import GraphEvent
from .cache_store import CacheStore, CacheEntry
from .config import CacheConfig, CacheMetrics


class CacheManager:
    """Manages cache operations and event handling for the graph context."""

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
    ):
        """Initialize the cache manager.

        Args:
            config: Optional cache configuration
        """
        self.config = config or CacheConfig()
        self.store = CacheStore(maxsize=self.config.max_size, ttl=self.config.default_ttl)
        self.metrics = CacheMetrics() if self.config.enable_metrics else None
        self._enabled = True

    def _track_cache_access(self, hit: bool, duration: float) -> None:
        """Track a cache access in metrics.

        Args:
            hit: Whether the access was a cache hit
            duration: Time taken for the operation in seconds
        """
        if not self.metrics:
            return

        if hit:
            self.metrics.hits += 1
        else:
            self.metrics.misses += 1
        self.metrics.total_time += duration

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current cache metrics.

        Returns:
            Dictionary of metrics if enabled, None otherwise
        """
        return self.metrics.to_dict() if self.metrics else None

    async def handle_event(self, event: GraphEvent, **kwargs) -> None:
        """Handle a graph event.

        Args:
            event: The graph event to handle
            **kwargs: Additional event data
        """
        if not self._enabled:
            return

        start_time = time.time()
        try:
            match event:
                case GraphEvent.ENTITY_READ:
                    await self._handle_entity_read(**kwargs)
                case GraphEvent.ENTITY_WRITE:
                    await self._handle_entity_write(**kwargs)
                case GraphEvent.ENTITY_DELETE:
                    await self._handle_entity_delete(**kwargs)
                case GraphEvent.RELATION_READ:
                    await self._handle_relation_read(**kwargs)
                case GraphEvent.RELATION_WRITE:
                    await self._handle_relation_write(**kwargs)
                case GraphEvent.RELATION_DELETE:
                    await self._handle_relation_delete(**kwargs)
                case GraphEvent.QUERY_EXECUTED:
                    await self._handle_query_executed(**kwargs)
                case GraphEvent.TRAVERSAL_EXECUTED:
                    await self._handle_traversal_executed(**kwargs)
                case GraphEvent.SCHEMA_MODIFIED:
                    await self._handle_schema_modified(**kwargs)
        finally:
            if self.metrics:
                duration = time.time() - start_time
                self.metrics.total_time += duration

    async def _handle_entity_read(
        self,
        entity_id: str,
        entity_type: str,
        result: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        """Handle an entity read event.

        Args:
            entity_id: ID of the entity
            entity_type: Type of the entity
            result: Optional entity data if found
            **_: Additional unused kwargs
        """
        key = f"entity:{entity_type}:{entity_id}"
        start_time = time.time()

        cached = await self.store.get(key)
        if cached:
            self._track_cache_access(True, time.time() - start_time)
            return cached.value

        self._track_cache_access(False, time.time() - start_time)

        if result:
            entry = CacheEntry(
                value=result,
                entity_type=entity_type,
            )
            await self.store.set(key, entry)

        return result

    async def _handle_entity_write(
        self,
        entity_id: str,
        entity_type: str,
        **_: Any,
    ) -> None:
        """Handle an entity write event.

        Args:
            entity_id: ID of the entity
            entity_type: Type of the entity
            **_: Additional unused kwargs
        """
        key = f"entity:{entity_type}:{entity_id}"
        await self.store.delete(key)
        # Invalidate queries that depend on this entity type
        await self.store.invalidate_type(entity_type)

    async def _handle_entity_delete(
        self,
        entity_id: str,
        entity_type: str,
        **_: Any,
    ) -> None:
        """Handle an entity delete event.

        Args:
            entity_id: ID of the entity
            entity_type: Type of the entity
            **_: Additional unused kwargs
        """
        await self._handle_entity_write(entity_id, entity_type)

    async def _handle_relation_read(
        self,
        relation_id: str,
        relation_type: str,
        result: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        """Handle a relation read event.

        Args:
            relation_id: ID of the relation
            relation_type: Type of the relation
            result: Optional relation data if found
            **_: Additional unused kwargs
        """
        key = f"relation:{relation_type}:{relation_id}"
        start_time = time.time()

        cached = await self.store.get(key)
        if cached:
            self._track_cache_access(True, time.time() - start_time)
            return cached.value

        self._track_cache_access(False, time.time() - start_time)

        if result:
            entry = CacheEntry(
                value=result,
                relation_type=relation_type,
            )
            await self.store.set(key, entry)

        return result

    async def _handle_relation_write(
        self,
        relation_id: str,
        relation_type: str,
        **_: Any,
    ) -> None:
        """Handle a relation write event.

        Args:
            relation_id: ID of the relation
            relation_type: Type of the relation
            **_: Additional unused kwargs
        """
        key = f"relation:{relation_type}:{relation_id}"
        await self.store.delete(key)
        # Invalidate queries that depend on this relation type
        await self.store.invalidate_type(relation_type)

    async def _handle_relation_delete(
        self,
        relation_id: str,
        relation_type: str,
        **_: Any,
    ) -> None:
        """Handle a relation delete event.

        Args:
            relation_id: ID of the relation
            relation_type: Type of the relation
            **_: Additional unused kwargs
        """
        await self._handle_relation_write(relation_id, relation_type)

    async def _handle_query_executed(
        self,
        query_hash: str,
        result: Any,
        dependencies: Set[str],
        **_: Any,
    ) -> None:
        """Handle a query execution event.

        Args:
            query_hash: Hash of the query
            result: Query result
            dependencies: Set of type names the query depends on
            **_: Additional unused kwargs
        """
        key = f"query:{query_hash}"
        entry = CacheEntry(
            value={"result": result},  # Wrap in dict to satisfy BaseModel requirement
            query_hash=query_hash,
            dependencies=dependencies,
        )
        await self.store.set(key, entry)

    async def _handle_traversal_executed(
        self,
        traversal_hash: str,
        result: Any,
        dependencies: Set[str],
        **_: Any,
    ) -> None:
        """Handle a traversal execution event.

        Args:
            traversal_hash: Hash of the traversal
            result: Traversal result
            dependencies: Set of type names the traversal depends on
            **_: Additional unused kwargs
        """
        key = f"traversal:{traversal_hash}"
        entry = CacheEntry(
            value={"result": result},  # Wrap in dict to satisfy BaseModel requirement
            query_hash=traversal_hash,  # Reuse query_hash field
            dependencies=dependencies,
        )
        await self.store.set(key, entry)

    async def _handle_schema_modified(
        self,
        modified_types: Set[str],
        **_: Any,
    ) -> None:
        """Handle a schema modification event.

        Args:
            modified_types: Set of type names that were modified
            **_: Additional unused kwargs
        """
        start_time = time.time()
        for type_name in modified_types:
            await self.store.invalidate_type(type_name)
            # Track cache miss for each invalidated type
            self._track_cache_access(False, time.time() - start_time)

    def enable(self) -> None:
        """Enable caching."""
        self._enabled = True

    def disable(self) -> None:
        """Disable caching."""
        self._enabled = False

    async def clear(self) -> None:
        """Clear all cache entries."""
        await self.store.clear()