"""Cache manager for the graph context.

This module provides the CacheManager class that handles caching operations
and event handling for the graph context.
"""

import time
import logging
from typing import Any, Dict, Optional, Set
from datetime import datetime, UTC

from graph_context.event_system import (
    EventSystem,
    GraphEvent,
    EventContext,
    EventMetadata
)
from .cache_store import CacheStore, CacheEntry
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
        self.store = CacheStore(maxsize=self.config.max_size, ttl=self.config.default_ttl)
        self.metrics = CacheMetrics() if self.config.enable_metrics else None
        self._enabled = True
        self._in_transaction = False
        self._transaction_cache: Dict[str, CacheEntry] = {}

        # Subscribe to events if event system is provided
        if event_system:
            self._subscribe_to_events(event_system)

    def _subscribe_to_events(self, event_system: EventSystem) -> None:
        """Subscribe to relevant graph events.

        Args:
            event_system: The event system to subscribe to
        """
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
            GraphEvent.TRANSACTION_BEGIN,
            GraphEvent.TRANSACTION_COMMIT,
            GraphEvent.TRANSACTION_ROLLBACK,
        ]
        for event in events:
            event_system.subscribe(event, self.handle_event)

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
            logger.debug("Cache hit (duration: %.3fs)", duration)
        else:
            self.metrics.misses += 1
            logger.debug("Cache miss (duration: %.3fs)", duration)
        self.metrics.total_time += duration

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current cache metrics.

        Returns:
            Dictionary of metrics if enabled, None otherwise
        """
        return self.metrics.to_dict() if self.metrics else None

    async def handle_event(self, context: EventContext) -> None:
        """Handle a graph event.

        Args:
            context: Event context containing event type and data
        """
        start_time = time.time()
        try:
            if not self._enabled:
                # When cache is disabled, treat all reads as misses
                if context.event in [GraphEvent.ENTITY_READ, GraphEvent.RELATION_READ,
                                   GraphEvent.QUERY_EXECUTED, GraphEvent.TRAVERSAL_EXECUTED]:
                    self._track_cache_access(False, time.time() - start_time)
                return

            match context.event:
                case GraphEvent.ENTITY_READ:
                    await self._handle_entity_read(context)
                case GraphEvent.ENTITY_WRITE | GraphEvent.ENTITY_BULK_WRITE:
                    await self._handle_entity_write(context)
                case GraphEvent.ENTITY_DELETE | GraphEvent.ENTITY_BULK_DELETE:
                    await self._handle_entity_delete(context)
                case GraphEvent.RELATION_READ:
                    await self._handle_relation_read(context)
                case GraphEvent.RELATION_WRITE | GraphEvent.RELATION_BULK_WRITE:
                    await self._handle_relation_write(context)
                case GraphEvent.RELATION_DELETE | GraphEvent.RELATION_BULK_DELETE:
                    await self._handle_relation_delete(context)
                case GraphEvent.QUERY_EXECUTED:
                    await self._handle_query_executed(context)
                case GraphEvent.TRAVERSAL_EXECUTED:
                    await self._handle_traversal_executed(context)
                case GraphEvent.SCHEMA_MODIFIED | GraphEvent.TYPE_MODIFIED:
                    await self._handle_schema_modified(context)
                case GraphEvent.TRANSACTION_BEGIN:
                    await self._handle_transaction_begin(context)
                case GraphEvent.TRANSACTION_COMMIT:
                    await self._handle_transaction_commit(context)
                case GraphEvent.TRANSACTION_ROLLBACK:
                    await self._handle_transaction_rollback(context)
        finally:
            if self.metrics:
                duration = time.time() - start_time
                self.metrics.total_time += duration

    async def _handle_entity_read(self, context: EventContext) -> None:
        """Handle an entity read event.

        Args:
            context: Event context containing entity information
        """
        entity_id = context.data["entity_id"]
        entity_type = context.metadata.entity_type
        result = context.data.get("result")

        key = f"entity:{entity_type}:{entity_id}"
        start_time = time.time()

        logger.debug("Handling entity read: %s (type: %s)", entity_id, entity_type)

        # Check transaction cache first if in transaction
        if self._in_transaction:
            if key in self._transaction_cache:
                logger.debug("Found in transaction cache: %s", key)
                self._track_cache_access(True, time.time() - start_time)
                return self._transaction_cache[key].value

        # Then check main cache
        cached = await self.store.get(key)
        if cached:
            logger.debug("Found in main cache: %s", key)
            self._track_cache_access(True, time.time() - start_time)
            return cached.value

        logger.debug("Not found in cache: %s", key)
        self._track_cache_access(False, time.time() - start_time)

        if result:
            logger.debug("Storing in cache: %s", key)
            entry = CacheEntry(
                value=result,
                entity_type=entity_type,
                operation_id=context.metadata.operation_id,
                created_at=datetime.now(UTC)
            )
            if self._in_transaction:
                self._transaction_cache[key] = entry
            else:
                await self.store.set(key, entry)

        return result

    async def _handle_entity_write(self, context: EventContext) -> None:
        """Handle an entity write event.

        Args:
            context: Event context containing entity information
        """
        entity_id = context.data["entity_id"]
        entity_type = context.metadata.entity_type

        key = f"entity:{entity_type}:{entity_id}"
        logger.debug("Handling entity write: %s (type: %s)", entity_id, entity_type)

        if self._in_transaction:
            # In transaction, just update transaction cache
            logger.debug("Removing from transaction cache: %s", key)
            self._transaction_cache.pop(key, None)
        else:
            # Outside transaction, invalidate main cache
            logger.debug("Invalidating cache for entity: %s", key)
            await self.store.delete(key)
            await self.store.invalidate_type(entity_type)
            # Also invalidate dependent queries and traversals
            await self.store.invalidate_dependencies(key)

    async def _handle_entity_delete(self, context: EventContext) -> None:
        """Handle an entity delete event.

        Args:
            context: Event context containing entity information
        """
        await self._handle_entity_write(context)

    async def _handle_relation_read(self, context: EventContext) -> None:
        """Handle a relation read event.

        Args:
            context: Event context containing relation information
        """
        relation_id = context.data["relation_id"]
        relation_type = context.metadata.relation_type
        result = context.data.get("result")

        key = f"relation:{relation_type}:{relation_id}"
        start_time = time.time()

        logger.debug("Handling relation read: %s (type: %s)", relation_id, relation_type)

        # Check transaction cache first if in transaction
        if self._in_transaction:
            if key in self._transaction_cache:
                logger.debug("Found in transaction cache: %s", key)
                self._track_cache_access(True, time.time() - start_time)
                return self._transaction_cache[key].value

        # Then check main cache
        cached = await self.store.get(key)
        if cached:
            logger.debug("Found in main cache: %s", key)
            self._track_cache_access(True, time.time() - start_time)
            return cached.value

        logger.debug("Not found in cache: %s", key)
        self._track_cache_access(False, time.time() - start_time)

        if result:
            logger.debug("Storing in cache: %s", key)
            entry = CacheEntry(
                value=result,
                relation_type=relation_type,
                operation_id=context.metadata.operation_id,
                created_at=datetime.now(UTC)
            )
            if self._in_transaction:
                self._transaction_cache[key] = entry
            else:
                await self.store.set(key, entry)

        return result

    async def _handle_relation_write(self, context: EventContext) -> None:
        """Handle a relation write event.

        Args:
            context: Event context containing relation information
        """
        relation_id = context.data["relation_id"]
        relation_type = context.metadata.relation_type

        key = f"relation:{relation_type}:{relation_id}"
        logger.debug("Handling relation write: %s (type: %s)", relation_id, relation_type)

        if self._in_transaction:
            # In transaction, just update transaction cache
            logger.debug("Removing from transaction cache: %s", key)
            self._transaction_cache.pop(key, None)
        else:
            # Outside transaction, invalidate main cache
            logger.debug("Invalidating cache for relation: %s", key)
            await self.store.delete(key)
            await self.store.invalidate_type(relation_type)
            # Also invalidate dependent queries and traversals
            await self.store.invalidate_dependencies(key)

    async def _handle_relation_delete(self, context: EventContext) -> None:
        """Handle a relation delete event.

        Args:
            context: Event context containing relation information
        """
        await self._handle_relation_write(context)

    async def _handle_query_executed(self, context: EventContext) -> None:
        """Handle a query execution event.

        Args:
            context: Event context containing query information
        """
        query_hash = context.data["query_hash"]
        key = f"query:{query_hash}"
        start_time = time.time()

        logger.debug("Handling query execution: %s", query_hash)

        # Check if this is a cache read attempt
        if "result" not in context.data:
            # Try to get from cache
            cached = await self.store.get(key)
            if cached:
                logger.debug("Found query in cache: %s", key)
                self._track_cache_access(True, time.time() - start_time)
                return cached.value
            logger.debug("Query not found in cache: %s", key)
            self._track_cache_access(False, time.time() - start_time)
            return None

        # This is a write operation - track as cache miss
        logger.debug("Storing query result in cache: %s", key)
        self._track_cache_access(False, time.time() - start_time)

        # Store the result
        result = context.data["result"]
        entry = CacheEntry(
            value=result,
            query_hash=query_hash,
            operation_id=context.metadata.operation_id,
            created_at=datetime.now(UTC)
        )

        if self._in_transaction:
            self._transaction_cache[key] = entry
        else:
            # Store with type dependencies
            await self.store.set(
                key,
                entry,
                dependencies=context.metadata.affected_types
            )
            if context.metadata.affected_types:
                logger.debug("Query dependencies: %s", context.metadata.affected_types)

        return result

    async def _handle_traversal_executed(self, context: EventContext) -> None:
        """Handle a traversal execution event.

        Args:
            context: Event context containing traversal information
        """
        traversal_hash = context.data["traversal_hash"]
        key = f"traversal:{traversal_hash}"
        start_time = time.time()

        logger.debug("Handling traversal execution: %s", traversal_hash)

        # Check if this is a cache read attempt
        if "result" not in context.data:
            # Try to get from cache
            cached = await self.store.get(key)
            if cached:
                logger.debug("Found traversal in cache: %s", key)
                self._track_cache_access(True, time.time() - start_time)
                return cached.value
            logger.debug("Traversal not found in cache: %s", key)
            self._track_cache_access(False, time.time() - start_time)
            return None

        # This is a write operation - track as cache miss
        logger.debug("Storing traversal result in cache: %s", key)
        self._track_cache_access(False, time.time() - start_time)

        # Store the result
        result = context.data["result"]
        entry = CacheEntry(
            value=result,
            query_hash=traversal_hash,  # Reuse query_hash field for traversal hash
            operation_id=context.metadata.operation_id,
            created_at=datetime.now(UTC)
        )

        if self._in_transaction:
            self._transaction_cache[key] = entry
        else:
            # Store with type dependencies
            await self.store.set(
                key,
                entry,
                dependencies=context.metadata.affected_types
            )
            if context.metadata.affected_types:
                logger.debug("Traversal dependencies: %s", context.metadata.affected_types)

        return result

    async def _handle_schema_modified(self, context: EventContext) -> None:
        """Handle a schema modification event.

        Args:
            context: Event context containing schema information
        """
        affected_types = context.metadata.affected_types
        if not affected_types:
            return

        logger.info("Schema modification affecting types: %s", affected_types)

        # Track cache miss for schema modification
        self._track_cache_access(False, 0)
        logger.debug("Tracking cache miss for schema modification")

        # Invalidate all affected types
        for type_name in affected_types:
            logger.debug("Invalidating cache for type: %s", type_name)
            await self.store.invalidate_type(type_name)

    async def _handle_transaction_begin(self, context: EventContext) -> None:
        """Handle a transaction begin event.

        Args:
            context: Event context for transaction begin
        """
        logger.info("Beginning new transaction")
        self._in_transaction = True
        self._transaction_cache.clear()

    async def _handle_transaction_commit(self, context: EventContext) -> None:
        """Handle a transaction commit event.

        Args:
            context: Event context for transaction commit
        """
        logger.info("Committing transaction with %d cached entries", len(self._transaction_cache))

        # Apply transaction cache to main cache
        for key, entry in self._transaction_cache.items():
            logger.debug("Committing cache entry: %s", key)
            await self.store.set(key, entry)

        self._in_transaction = False
        self._transaction_cache.clear()

    async def _handle_transaction_rollback(self, context: EventContext) -> None:
        """Handle a transaction rollback event.

        Args:
            context: Event context for transaction rollback
        """
        logger.info("Rolling back transaction, discarding %d cached entries", len(self._transaction_cache))
        self._in_transaction = False
        self._transaction_cache.clear()

    def enable(self) -> None:
        """Enable the cache manager."""
        logger.info("Enabling cache manager")
        self._enabled = True

    def disable(self) -> None:
        """Disable the cache manager."""
        logger.info("Disabling cache manager")
        self._enabled = False

    async def clear(self) -> None:
        """Clear all caches."""
        logger.info("Clearing all caches")
        await self.store.clear()
        self._transaction_cache.clear()