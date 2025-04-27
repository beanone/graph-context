"""Cache manager for the graph context.

This module provides the CacheManager class that handles caching operations
and event handling for the graph context.
"""

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from graph_context.event_system import EventContext, EventSystem, GraphEvent

from .cache_store import CacheEntry, CacheStore, DisabledCacheStore
from .cache_store_manager import CacheStoreManager
from .config import CacheConfig, CacheMetrics

# Setup module logger
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages cache operations and event handling for the graph context."""

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        event_system: Optional[EventSystem] = None,
    ):
        """Initialize the cache manager.

        Args:
            config: Optional cache configuration
            event_system: Optional event system to subscribe to
        """
        self.config = config or CacheConfig()
        self.store_manager = CacheStoreManager(self.config)
        self.metrics = CacheMetrics() if self.config.enable_metrics else None

        # Subscribe to events if event system is provided
        if event_system:
            self._subscribe_to_events(event_system)

    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        # Caching is disabled if we're using DisabledCacheStore
        return not isinstance(self.store_manager.entity_store, DisabledCacheStore)

    def _subscribe_to_events(self, event_system: EventSystem) -> None:
        """Subscribe to relevant graph events."""
        events = [
            GraphEvent.ENTITY_READ,
            GraphEvent.ENTITY_WRITE,
            GraphEvent.ENTITY_DELETE,
            GraphEvent.ENTITY_BULK_WRITE,
            GraphEvent.ENTITY_BULK_DELETE,
            GraphEvent.RELATION_READ,
            GraphEvent.RELATION_WRITE,
            GraphEvent.RELATION_DELETE,
            GraphEvent.RELATION_BULK_WRITE,
            GraphEvent.RELATION_BULK_DELETE,
            GraphEvent.QUERY_EXECUTED,
            GraphEvent.TRAVERSAL_EXECUTED,
            GraphEvent.SCHEMA_MODIFIED,
            GraphEvent.TYPE_MODIFIED,
        ]
        for event in events:
            event_system.subscribe(event, self.handle_event)

    def _track_cache_access(self, hit: bool, duration: float) -> None:
        """Track a cache access in metrics."""
        if not self.metrics:
            return

        if hit:
            self.metrics.hits += 1
            logger.debug("Cache hit (duration: %.3fs)", duration)
        else:
            self.metrics.misses += 1
            logger.debug("Cache miss (duration: %.3fs)", duration)
        self.metrics.total_time += duration

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current cache metrics."""
        return self.metrics.to_dict() if self.metrics else None

    def is_read_only(self, event: GraphEvent) -> bool:
        """Return True if the event is a read-only cache event."""
        return event in [
            GraphEvent.ENTITY_READ,
            GraphEvent.RELATION_READ,
            GraphEvent.QUERY_EXECUTED,
            GraphEvent.TRAVERSAL_EXECUTED,
        ]

    async def handle_event(self, context: EventContext) -> None:
        """Handle a graph event."""
        start_time = time.time()
        try:
            if not self.is_enabled():
                # When cache is disabled, treat all reads as misses
                if self.is_read_only(context.event):
                    self._track_cache_access(False, time.time() - start_time)
                return

            # Map of event types to handler coroutines
            event_handlers = {
                GraphEvent.ENTITY_READ: self._handle_entity_read,
                GraphEvent.RELATION_READ: self._handle_relation_read,
                GraphEvent.QUERY_EXECUTED: self._handle_query_executed,
                GraphEvent.TRAVERSAL_EXECUTED: self._handle_traversal_executed,
                GraphEvent.ENTITY_WRITE: self._handle_entity_write,
                GraphEvent.ENTITY_BULK_WRITE: self._handle_entity_write,
                GraphEvent.ENTITY_DELETE: self._handle_entity_write,
                GraphEvent.ENTITY_BULK_DELETE: self._handle_entity_write,
                GraphEvent.RELATION_WRITE: self._handle_relation_write,
                GraphEvent.RELATION_BULK_WRITE: self._handle_relation_write,
                GraphEvent.RELATION_DELETE: self._handle_relation_write,
                GraphEvent.RELATION_BULK_DELETE: self._handle_relation_write,
                GraphEvent.SCHEMA_MODIFIED: self._handle_schema_modified,
                GraphEvent.TYPE_MODIFIED: self._handle_schema_modified,
            }
            handler = event_handlers.get(context.event)
            if handler:
                await handler(context)
        finally:
            if self.metrics:
                duration = time.time() - start_time
                self.metrics.total_time += duration

    async def _handle_entity_read(self, context: EventContext) -> None:
        """Handle an entity read event."""
        entity_id = context.data["entity_id"]
        result = context.data.get("result")

        start_time = time.time()
        logger.debug("Handling entity read: %s", entity_id)

        # Check cache
        store = self.store_manager.get_entity_store()
        cached_entry = await store.get(entity_id)
        if cached_entry:
            logger.debug("Found in cache: %s", entity_id)
            self._track_cache_access(True, time.time() - start_time)
            return cached_entry.value

        logger.debug("Not found in cache: %s", entity_id)
        self._track_cache_access(False, time.time() - start_time)

        # Cache the result if we have one
        if result:
            logger.debug("Storing in cache: %s", entity_id)
            entry = CacheEntry(
                value=result,
                created_at=datetime.now(UTC),
                entity_type=context.metadata.entity_type,
            )
            await store.set(entity_id, entry)

        return result

    async def _handle_entity_write(self, context: EventContext) -> None:
        """Handle an entity write/delete event."""
        entity_id = context.data["entity_id"]
        logger.debug("Handling entity write/delete: %s", entity_id)

        # Simply invalidate the cache entry
        store = self.store_manager.get_entity_store()
        await store.delete(entity_id)

    async def _handle_relation_read(self, context: EventContext) -> None:
        """Handle a relation read event."""
        relation_id = context.data["relation_id"]
        result = context.data.get("result")

        start_time = time.time()
        logger.debug("Handling relation read: %s", relation_id)

        # Check cache
        store = self.store_manager.get_relation_store()
        cached_entry = await store.get(relation_id)
        if cached_entry:
            logger.debug("Found in cache: %s", relation_id)
            self._track_cache_access(True, time.time() - start_time)
            return cached_entry.value

        logger.debug("Not found in cache: %s", relation_id)
        self._track_cache_access(False, time.time() - start_time)

        # Cache the result if we have one
        if result:
            logger.debug("Storing in cache: %s", relation_id)
            entry = CacheEntry(
                value=result,
                created_at=datetime.now(UTC),
                relation_type=context.metadata.relation_type,
            )
            await store.set(relation_id, entry)

        return result

    async def _handle_relation_write(self, context: EventContext) -> None:
        """Handle a relation write/delete event."""
        relation_id = context.data["relation_id"]
        logger.debug("Handling relation write/delete: %s", relation_id)

        # Simply invalidate the cache entry
        store = self.store_manager.get_relation_store()
        await store.delete(relation_id)

    async def _handle_query_executed(self, context: EventContext) -> None:
        """Handle a query execution event."""
        query_hash = context.data["query_hash"]
        result = context.data.get("result")

        start_time = time.time()
        logger.debug("Handling query execution: %s", query_hash)

        # Check cache
        store = self.store_manager.get_query_store()
        cached_entry = await store.get(query_hash)
        if cached_entry:
            logger.debug("Found in cache: %s", query_hash)
            self._track_cache_access(True, time.time() - start_time)
            return cached_entry.value

        logger.debug("Not found in cache: %s", query_hash)
        self._track_cache_access(False, time.time() - start_time)

        # Cache the result if we have one
        if result:
            logger.debug("Storing in cache: %s", query_hash)
            entry = CacheEntry(
                value=result, created_at=datetime.now(UTC), query_hash=query_hash
            )
            await store.set(query_hash, entry)

        return result

    async def _handle_traversal_executed(self, context: EventContext) -> None:
        """Handle a traversal execution event."""
        traversal_hash = context.data["traversal_hash"]
        result = context.data.get("result")

        start_time = time.time()
        logger.debug("Handling traversal execution: %s", traversal_hash)

        # Check cache
        store = self.store_manager.get_traversal_store()
        cached_entry = await store.get(traversal_hash)
        if cached_entry:
            logger.debug("Found in cache: %s", traversal_hash)
            self._track_cache_access(True, time.time() - start_time)
            return cached_entry.value

        logger.debug("Not found in cache: %s", traversal_hash)
        self._track_cache_access(False, time.time() - start_time)

        # Cache the result if we have one
        if result:
            logger.debug("Storing in cache: %s", traversal_hash)
            entry = CacheEntry(
                value=result,
                created_at=datetime.now(UTC),
                query_hash=traversal_hash,  # Reuse query_hash field for traversal hash
            )
            await store.set(traversal_hash, entry)

        return result

    async def _handle_schema_modified(self, context: EventContext) -> None:
        """Handle a schema modification event."""
        logger.info("Schema modification - clearing all caches")
        await self.clear()

    def enable(self) -> None:
        """Enable caching by creating new cache stores."""
        if self.is_enabled():
            return

        logger.info("Enabling cache manager")

        # Create fresh cache stores
        self.store_manager.entity_store = CacheStore(
            maxsize=self.config.entity_cache_size, ttl=self.config.entity_cache_ttl
        )

        self.store_manager.relation_store = CacheStore(
            maxsize=self.config.relation_cache_size, ttl=self.config.relation_cache_ttl
        )

        self.store_manager.query_store = CacheStore(
            maxsize=self.config.query_cache_size, ttl=self.config.query_cache_ttl
        )

        self.store_manager.traversal_store = CacheStore(
            maxsize=self.config.traversal_cache_size,
            ttl=self.config.traversal_cache_ttl,
        )

    def disable(self) -> None:
        """Disable caching by replacing stores with DisabledCacheStore."""
        if not self.is_enabled():
            return

        logger.info("Disabling cache manager")

        # Create a single disabled store and use it for all store types
        disabled_store = DisabledCacheStore()

        # Replace all stores with the disabled store
        self.store_manager.entity_store = disabled_store
        self.store_manager.relation_store = disabled_store
        self.store_manager.query_store = disabled_store
        self.store_manager.traversal_store = disabled_store

    async def clear(self) -> None:
        """Clear all caches."""
        logger.info("Clearing all caches")
        await self.store_manager.clear_all()

    def _hash_query(self, query_spec: Any) -> str:
        """Generate a hash for a query specification."""
        query_str = json.dumps(query_spec, sort_keys=True)
        return hashlib.sha256(query_str.encode()).hexdigest()
