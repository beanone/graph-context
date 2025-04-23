"""Cache manager for the graph context.

This module provides the CacheManager class that handles caching operations
and event handling for the graph context.
"""

import time
import logging
import hashlib
import json
from typing import Any, Dict, Optional
from datetime import datetime, UTC

from graph_context.event_system import (
    EventSystem,
    GraphEvent,
    EventContext,
    EventMetadata
)
from .cache_store import CacheEntry
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
        self._enabled = True
        self._in_transaction = False
        self._transaction_cache: Dict[str, Dict[str, CacheEntry]] = {
            'entity': {},
            'relation': {},
            'query': {},
            'traversal': {}
        }

        # Subscribe to events if event system is provided
        if event_system:
            self._subscribe_to_events(event_system)

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
            GraphEvent.TRANSACTION_BEGIN,
            GraphEvent.TRANSACTION_COMMIT,
            GraphEvent.TRANSACTION_ROLLBACK,
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

    async def handle_event(self, context: EventContext) -> None:
        """Handle a graph event."""
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
                    await self._handle_entity_write(context)  # Same handling as write
                case GraphEvent.RELATION_READ:
                    await self._handle_relation_read(context)
                case GraphEvent.RELATION_WRITE | GraphEvent.RELATION_BULK_WRITE:
                    await self._handle_relation_write(context)
                case GraphEvent.RELATION_DELETE | GraphEvent.RELATION_BULK_DELETE:
                    await self._handle_relation_write(context)  # Same handling as write
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
        """Handle an entity read event."""
        entity_id = context.data["entity_id"]
        result = context.data.get("result")

        start_time = time.time()
        logger.debug("Handling entity read: %s", entity_id)

        # Check transaction cache first if in transaction
        if self._in_transaction:
            if entity_id in self._transaction_cache['entity']:
                logger.debug("Found in transaction cache: %s", entity_id)
                self._track_cache_access(True, time.time() - start_time)
                return self._transaction_cache['entity'][entity_id].value

            # If we have a result and we're in a transaction, store it in transaction cache
            if result:
                logger.debug("Storing in transaction cache: %s", entity_id)
                entry = CacheEntry(
                    value=result,
                    created_at=datetime.now(UTC),
                    entity_type=context.metadata.entity_type
                )
                self._transaction_cache['entity'][entity_id] = entry
                return result

        # Then check main cache
        store = self.store_manager.get_entity_store()
        cached_entry = await store.get(entity_id)
        if cached_entry:
            logger.debug("Found in main cache: %s", entity_id)
            self._track_cache_access(True, time.time() - start_time)
            return cached_entry.value

        logger.debug("Not found in cache: %s", entity_id)
        self._track_cache_access(False, time.time() - start_time)

        if result and not self._in_transaction:
            logger.debug("Storing in main cache: %s", entity_id)
            entry = CacheEntry(
                value=result,
                created_at=datetime.now(UTC),
                entity_type=context.metadata.entity_type
            )
            await store.set(entity_id, entry)

        return result

    async def _handle_entity_write(self, context: EventContext) -> None:
        """Handle an entity write/delete event."""
        entity_id = context.data["entity_id"]
        logger.debug("Handling entity write/delete: %s", entity_id)

        if self._in_transaction:
            self._transaction_cache['entity'].pop(entity_id, None)
        else:
            store = self.store_manager.get_entity_store()
            await store.delete(entity_id)

    async def _handle_relation_read(self, context: EventContext) -> None:
        """Handle a relation read event."""
        relation_id = context.data["relation_id"]
        result = context.data.get("result")

        start_time = time.time()
        logger.debug("Handling relation read: %s", relation_id)

        # Check transaction cache first if in transaction
        if self._in_transaction:
            if relation_id in self._transaction_cache['relation']:
                logger.debug("Found in transaction cache: %s", relation_id)
                self._track_cache_access(True, time.time() - start_time)
                return self._transaction_cache['relation'][relation_id].value

        # Then check main cache
        store = self.store_manager.get_relation_store()
        cached_entry = await store.get(relation_id)
        if cached_entry:
            logger.debug("Found in main cache: %s", relation_id)
            self._track_cache_access(True, time.time() - start_time)
            return cached_entry.value

        logger.debug("Not found in cache: %s", relation_id)
        self._track_cache_access(False, time.time() - start_time)

        if result:
            logger.debug("Storing in cache: %s", relation_id)
            entry = CacheEntry(
                value=result,
                created_at=datetime.now(UTC),
                relation_type=context.metadata.relation_type
            )
            if self._in_transaction:
                self._transaction_cache['relation'][relation_id] = entry
            else:
                await store.set(relation_id, entry)

        return result

    async def _handle_relation_write(self, context: EventContext) -> None:
        """Handle a relation write/delete event."""
        relation_id = context.data["relation_id"]
        logger.debug("Handling relation write/delete: %s", relation_id)

        if self._in_transaction:
            self._transaction_cache['relation'].pop(relation_id, None)
        else:
            store = self.store_manager.get_relation_store()
            await store.delete(relation_id)

    async def _handle_query_executed(self, context: EventContext) -> None:
        """Handle a query execution event."""
        query_hash = context.data["query_hash"]
        result = context.data.get("result")

        start_time = time.time()
        logger.debug("Handling query execution: %s", query_hash)

        # Check transaction cache first if in transaction
        if self._in_transaction:
            if query_hash in self._transaction_cache['query']:
                logger.debug("Found in transaction cache: %s", query_hash)
                self._track_cache_access(True, time.time() - start_time)
                return self._transaction_cache['query'][query_hash].value

        # Then check main cache
        store = self.store_manager.get_query_store()
        cached_entry = await store.get(query_hash)
        if cached_entry:
            logger.debug("Found in query cache: %s", query_hash)
            self._track_cache_access(True, time.time() - start_time)
            return cached_entry.value

        logger.debug("Not found in cache: %s", query_hash)
        self._track_cache_access(False, time.time() - start_time)

        if result:
            logger.debug("Storing in cache: %s", query_hash)
            entry = CacheEntry(
                value=result,
                created_at=datetime.now(UTC),
                query_hash=query_hash
            )
            if self._in_transaction:
                self._transaction_cache['query'][query_hash] = entry
            else:
                await store.set(query_hash, entry)

        return result

    async def _handle_traversal_executed(self, context: EventContext) -> None:
        """Handle a traversal execution event."""
        traversal_hash = context.data["traversal_hash"]
        result = context.data.get("result")

        start_time = time.time()
        logger.debug("Handling traversal execution: %s", traversal_hash)

        # Check transaction cache first if in transaction
        if self._in_transaction:
            if traversal_hash in self._transaction_cache['traversal']:
                logger.debug("Found in transaction cache: %s", traversal_hash)
                self._track_cache_access(True, time.time() - start_time)
                return self._transaction_cache['traversal'][traversal_hash].value

        # Then check main cache
        store = self.store_manager.get_traversal_store()
        cached_entry = await store.get(traversal_hash)
        if cached_entry:
            logger.debug("Found in traversal cache: %s", traversal_hash)
            self._track_cache_access(True, time.time() - start_time)
            return cached_entry.value

        logger.debug("Not found in cache: %s", traversal_hash)
        self._track_cache_access(False, time.time() - start_time)

        if result:
            logger.debug("Storing in cache: %s", traversal_hash)
            entry = CacheEntry(
                value=result,
                created_at=datetime.now(UTC),
                query_hash=traversal_hash  # Reuse query_hash field for traversal hash
            )
            if self._in_transaction:
                self._transaction_cache['traversal'][traversal_hash] = entry
            else:
                await store.set(traversal_hash, entry)

        return result

    async def _handle_schema_modified(self, context: EventContext) -> None:
        """Handle a schema modification event."""
        logger.info("Schema modification - clearing all caches")
        await self.clear()

    async def _handle_transaction_begin(self, context: EventContext) -> None:
        """Handle a transaction begin event."""
        logger.info("Beginning new transaction")
        self._in_transaction = True
        for cache in self._transaction_cache.values():
            cache.clear()

    async def _handle_transaction_commit(self, context: EventContext) -> None:
        """Handle a transaction commit event."""
        logger.info("Committing transaction")

        # Get all stores
        entity_store = self.store_manager.get_entity_store()
        relation_store = self.store_manager.get_relation_store()
        query_store = self.store_manager.get_query_store()
        traversal_store = self.store_manager.get_traversal_store()

        # Commit all cached entries to their respective stores
        for entity_id, entry in self._transaction_cache['entity'].items():
            await entity_store.set(entity_id, entry)

        for relation_id, entry in self._transaction_cache['relation'].items():
            await relation_store.set(relation_id, entry)

        for query_hash, entry in self._transaction_cache['query'].items():
            await query_store.set(query_hash, entry)

        for traversal_hash, entry in self._transaction_cache['traversal'].items():
            await traversal_store.set(traversal_hash, entry)

        self._in_transaction = False
        for cache in self._transaction_cache.values():
            cache.clear()

    async def _handle_transaction_rollback(self, context: EventContext) -> None:
        """Handle a transaction rollback event."""
        logger.info("Rolling back transaction")
        self._in_transaction = False
        for cache in self._transaction_cache.values():
            cache.clear()

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
        await self.store_manager.clear_all()
        for cache in self._transaction_cache.values():
            cache.clear()

    def _hash_query(self, query_spec: Any) -> str:
        """Generate a hash for a query specification."""
        query_str = json.dumps(query_spec, sort_keys=True)
        return hashlib.sha256(query_str.encode()).hexdigest()