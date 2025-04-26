"""Cached graph context implementation.

This module provides a cached implementation of the graph context interface,
which wraps a base context and adds caching functionality using the decorator pattern.
"""

import logging
from typing import Any, Dict, Optional

from ..event_system import EventContext, EventMetadata, GraphEvent
from ..exceptions import EntityNotFoundError, RelationNotFoundError, TransactionError
from ..interface import GraphContext
from ..types.type_base import Entity, QuerySpec, Relation, TraversalSpec
from .cache_manager import CacheManager
from .cache_store import CacheEntry

logger = logging.getLogger(__name__)


class CacheTransactionManager:
    """
    Manages transaction state and operations for cached context.

    This class encapsulates transaction-related logic to ensure proper
    transaction state management for the cached context.
    """

    def __init__(self, base_context: GraphContext, cache_manager: CacheManager):
        """
        Initialize the transaction manager.

        Args:
            base_context: The base graph context to delegate transactions to
            cache_manager: The cache manager to notify about transaction events
        """
        self._base_context = base_context
        self._cache_manager = cache_manager
        self._in_transaction = False

    def is_in_transaction(self) -> bool:
        """Return whether there is an active transaction."""
        return self._in_transaction

    def check_transaction(self, required: bool = True) -> None:
        """
        Check transaction state.

        Args:
            required: Whether a transaction is required

        Raises:
            TransactionError: If transaction state does not match requirement
        """
        if required and not self._in_transaction:
            raise TransactionError("Operation requires an active transaction")
        elif not required and self._in_transaction:
            raise TransactionError("Operation cannot be performed in a transaction")

    async def begin_transaction(self) -> None:
        """
        Begin a new transaction.

        Raises:
            TransactionError: If a transaction is already in progress
        """
        logger.debug("Beginning transaction")

        if self._in_transaction:
            logger.warning("Transaction already in progress")
            raise TransactionError("Transaction already in progress")

        # First begin the transaction in the base context
        await self._base_context.begin_transaction()

        # Set transaction flag
        self._in_transaction = True

        # Clear all caches to ensure we get fresh data during the transaction
        await self._cache_manager.store_manager.clear_all()
        logger.debug("Transaction started - all caches cleared")

        # Notify cache manager about transaction begin
        await self._cache_manager.handle_event(EventContext(
            event=GraphEvent.TRANSACTION_BEGIN,
            data={},
            metadata=EventMetadata()
        ))
        logger.debug("Transaction begin event sent to cache manager")

    async def commit_transaction(self) -> None:
        """
        Commit the current transaction.

        Raises:
            TransactionError: If no transaction is in progress
        """
        logger.debug("Committing transaction")

        if not self._in_transaction:
            logger.warning("Commit called but no active transaction")
            raise TransactionError("No transaction in progress")

        # Commit in the base context
        await self._base_context.commit_transaction()
        logger.debug("Base context transaction committed")

        # Clear transaction state
        self._in_transaction = False

        # Clear all caches to ensure we get fresh data after commit
        await self._cache_manager.store_manager.clear_all()
        logger.debug("All caches cleared after commit")

        # Notify cache manager about transaction commit
        await self._cache_manager.handle_event(EventContext(
            event=GraphEvent.TRANSACTION_COMMIT,
            data={},
            metadata=EventMetadata()
        ))
        logger.debug("Transaction commit event sent to cache manager")

    async def rollback_transaction(self) -> None:
        """
        Rollback the current transaction.

        Raises:
            TransactionError: If no transaction is in progress
        """
        logger.debug("Rolling back transaction")

        if not self._in_transaction:
            logger.warning("Rollback called but no active transaction")
            raise TransactionError("No transaction in progress")

        # Rollback the base context
        await self._base_context.rollback_transaction()
        logger.debug("Base context transaction rolled back")

        # Clear transaction state
        self._in_transaction = False

        # Clear all caches to ensure we get fresh data after rollback
        await self._cache_manager.store_manager.clear_all()
        logger.debug("All caches cleared after rollback")

        # Notify cache manager about transaction rollback
        await self._cache_manager.handle_event(EventContext(
            event=GraphEvent.TRANSACTION_ROLLBACK,
            data={},
            metadata=EventMetadata()
        ))
        logger.debug("Transaction rollback event sent to cache manager")


class CachedGraphContext(GraphContext):
    """A decorator that adds caching to any GraphContext implementation.

    This class wraps another graph context implementation and adds caching
    functionality. It uses the cache manager to handle caching of entities,
    relations, queries, and traversals.
    """

    def __init__(
        self,
        base_context: GraphContext,
        cache_manager: CacheManager
    ):
        """Initialize the cached graph context.

        Args:
            base_context: The base graph context to wrap
            cache_manager: The cache manager to use
        """
        self._base = base_context
        self._cache_manager = cache_manager
        self._initialized = False
        self._transaction = CacheTransactionManager(self._base, self._cache_manager)

    async def _initialize(self) -> None:
        """Initialize event subscriptions asynchronously."""
        if self._initialized:
            return

        # Subscribe cache manager to base context events
        if hasattr(self._base, '_events'):
            await self._base._events.subscribe(GraphEvent.ENTITY_READ, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.ENTITY_WRITE, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.ENTITY_BULK_WRITE, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.ENTITY_DELETE, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.ENTITY_BULK_DELETE, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.RELATION_READ, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.RELATION_WRITE, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.RELATION_BULK_WRITE, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.RELATION_DELETE, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.RELATION_BULK_DELETE, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.QUERY_EXECUTED, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.TRAVERSAL_EXECUTED, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.SCHEMA_MODIFIED, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.TYPE_MODIFIED, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.TRANSACTION_BEGIN, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.TRANSACTION_COMMIT, self._cache_manager.handle_event)
            await self._base._events.subscribe(GraphEvent.TRANSACTION_ROLLBACK, self._cache_manager.handle_event)

        self._initialized = True

    def enable_caching(self) -> None:
        """Enable caching."""
        self._cache_manager.enable()

    def disable_caching(self) -> None:
        """Disable caching."""
        self._cache_manager.disable()

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Rollback any active transaction
        if self._transaction.is_in_transaction():
            await self._transaction.rollback_transaction()

        await self._base.cleanup()

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID.

        Args:
            entity_id: The ID of the entity to get

        Returns:
            The entity if found, None otherwise

        Raises:
            EntityNotFoundError: If the entity does not exist
        """
        await self._initialize()
        logger.debug(f"Getting entity {entity_id}")

        # Skip cache if in transaction or caching is disabled
        if self._transaction.is_in_transaction() or not self._cache_manager.is_enabled():
            logger.debug(f"Bypassing cache for entity {entity_id} (transaction={self._transaction.is_in_transaction()}, caching_enabled={self._cache_manager.is_enabled()})")
            result = await self._base.get_entity(entity_id)
            if result is None:
                logger.debug(f"Entity {entity_id} not found in base context")
                raise EntityNotFoundError(f"Entity {entity_id} not found")
            return result

        # Try to get from cache
        cached = await self._cache_manager.store_manager.get_entity_store().get(entity_id)
        if cached is not None:
            logger.debug(f"Found entity {entity_id} in cache")
            return cached.value

        # Fall back to base context
        logger.debug(f"Getting entity {entity_id} from base context")
        result = await self._base.get_entity(entity_id)

        # Cache the result if found
        if result is not None:
            entry = CacheEntry(
                value=result,
                entity_type=result.type
            )
            await self._cache_manager.store_manager.get_entity_store().set(entity_id, entry)
            logger.debug(f"Cached entity {entity_id} from base context")
            return result
        else:
            logger.debug(f"Entity {entity_id} not found")
            raise EntityNotFoundError(f"Entity {entity_id} not found")

    async def get_relation(self, relation_id: str) -> Relation | None:
        """Get a relation by ID.

        Args:
            relation_id: The ID of the relation to get

        Returns:
            The relation if found, None otherwise

        Raises:
            RelationNotFoundError: If the relation does not exist
        """
        await self._initialize()

        # Skip cache if in transaction or caching is disabled
        if self._transaction.is_in_transaction() or not self._cache_manager.is_enabled():
            result = await self._base.get_relation(relation_id)
            if result is None:
                raise RelationNotFoundError(f"Relation {relation_id} not found")
            return result

        # Try to get from cache first
        cached = await self._cache_manager.store_manager.get_relation_store().get(relation_id)
        if cached is not None:
            return cached.value

        # Fall back to base context
        result = await self._base.get_relation(relation_id)

        # Cache the result if found
        if result is not None:
            entry = CacheEntry(
                value=result,
                relation_type=result.type
            )
            await self._cache_manager.store_manager.get_relation_store().set(relation_id, entry)
            return result
        else:
            raise RelationNotFoundError(f"Relation {relation_id} not found")

    async def query(self, query_spec: QuerySpec) -> list[Entity]:
        """Execute a query against the graph.

        Args:
            query_spec: The query specification

        Returns:
            The query results
        """
        await self._initialize()
        logger.debug(f"Executing query with spec: {query_spec}")

        # Skip cache if in transaction or caching is disabled
        if self._transaction.is_in_transaction() or not self._cache_manager.is_enabled():
            logger.debug("Bypassing cache for query")
            return await self._base.query(query_spec) or []

        # Try to get from cache first
        query_hash = self._cache_manager._hash_query(query_spec)
        cached = await self._cache_manager.store_manager.get_query_store().get(query_hash)
        if cached is not None:
            logger.debug(f"Query cache hit for hash {query_hash}")
            return cached.value

        # Fall back to base context
        logger.debug("Query cache miss, executing on base context")
        result = await self._base.query(query_spec)

        # Cache the result
        if result is not None:
            entry = CacheEntry(
                value=result,
                query_hash=query_hash
            )
            await self._cache_manager.store_manager.get_query_store().set(query_hash, entry)
            logger.debug(f"Cached query results with hash {query_hash}")

            # Notify cache manager about the query
            await self._cache_manager.handle_event(EventContext(
                event=GraphEvent.QUERY_EXECUTED,
                data={"query_spec": query_spec, "query_hash": query_hash},
                metadata=EventMetadata()
            ))

        return result or []  # Ensure we always return a list

    async def traverse(self, start_entity: str, traversal_spec: TraversalSpec) -> list[Entity]:
        """Execute a traversal in the graph.

        Args:
            start_entity: ID of the entity to start traversal from
            traversal_spec: The traversal specification

        Returns:
            The traversal results
        """
        await self._initialize()

        # Skip cache if in transaction or caching is disabled
        if self._transaction.is_in_transaction() or not self._cache_manager.is_enabled():
            return await self._base.traverse(start_entity, traversal_spec) or []

        # Try to get from cache first
        traversal_hash = self._cache_manager._hash_query(traversal_spec)
        cached = await self._cache_manager.store_manager.get_traversal_store().get(traversal_hash)
        if cached is not None:
            return cached.value

        # Fall back to base context
        result = await self._base.traverse(start_entity, traversal_spec)

        # Cache the result
        if result is not None:
            entry = CacheEntry(
                value=result,
                query_hash=traversal_hash
            )
            await self._cache_manager.store_manager.get_traversal_store().set(traversal_hash, entry)

        return result or []  # Ensure we always return a list

    async def create_entity(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """Create a new entity."""
        await self._initialize()
        entity_id = await self._base.create_entity(entity_type, properties)

        # Cache the newly created entity if not in transaction
        if not self._transaction.is_in_transaction():
            entity = await self._base.get_entity(entity_id)
            if entity is not None:
                entry = CacheEntry(
                    value=entity,
                    entity_type=entity_type
                )
                await self._cache_manager.store_manager.get_entity_store().set(entity_id, entry)

        # Notify cache manager about the write
        await self._cache_manager.handle_event(EventContext(
            event=GraphEvent.ENTITY_WRITE,
            data={"entity_id": entity_id},
            metadata=EventMetadata(entity_type=entity_type)
        ))

        return entity_id

    async def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Update an existing entity."""
        await self._initialize()
        logger.debug(f"Updating entity {entity_id} with properties {properties}")

        success = await self._base.update_entity(entity_id, properties)

        if success:
            logger.debug(f"Entity {entity_id} updated successfully")

            # Clear from main cache
            await self._cache_manager.store_manager.get_entity_store().delete(entity_id)
            logger.debug(f"Cleared entity {entity_id} from cache")

            # Notify cache manager about the write
            await self._cache_manager.handle_event(EventContext(
                event=GraphEvent.ENTITY_WRITE,
                data={"entity_id": entity_id},
                metadata=EventMetadata()
            ))
            logger.debug(f"Entity write event sent for {entity_id}")
        else:
            logger.debug(f"Failed to update entity {entity_id}")

        return success

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        await self._initialize()
        success = await self._base.delete_entity(entity_id)

        if success:
            # Invalidate cache
            await self._cache_manager.store_manager.get_entity_store().delete(entity_id)

            # Notify cache manager about the delete
            await self._cache_manager.handle_event(EventContext(
                event=GraphEvent.ENTITY_DELETE,
                data={"entity_id": entity_id},
                metadata=EventMetadata()
            ))

        return success

    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new relation."""
        await self._initialize()
        relation_id = await self._base.create_relation(relation_type, from_entity, to_entity, properties)

        # Cache the newly created relation if not in transaction
        if not self._transaction.is_in_transaction():
            relation = await self._base.get_relation(relation_id)
            if relation is not None:
                entry = CacheEntry(
                    value=relation,
                    relation_type=relation_type
                )
                await self._cache_manager.store_manager.get_relation_store().set(relation_id, entry)

        # Notify cache manager about the write
        await self._cache_manager.handle_event(EventContext(
            event=GraphEvent.RELATION_WRITE,
            data={"relation_id": relation_id},
            metadata=EventMetadata(relation_type=relation_type)
        ))

        return relation_id

    async def update_relation(self, relation_id: str, properties: Dict[str, Any]) -> bool:
        """Update an existing relation."""
        await self._initialize()
        success = await self._base.update_relation(relation_id, properties)

        if success:
            # Invalidate cache
            await self._cache_manager.store_manager.get_relation_store().delete(relation_id)

            # Notify cache manager about the write
            await self._cache_manager.handle_event(EventContext(
                event=GraphEvent.RELATION_WRITE,
                data={"relation_id": relation_id},
                metadata=EventMetadata()
            ))

        return success

    async def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation."""
        await self._initialize()
        success = await self._base.delete_relation(relation_id)

        if success:
            # Invalidate cache
            await self._cache_manager.store_manager.get_relation_store().delete(relation_id)

            # Notify cache manager about the delete
            await self._cache_manager.handle_event(EventContext(
                event=GraphEvent.RELATION_DELETE,
                data={"relation_id": relation_id},
                metadata=EventMetadata()
            ))

        return success

    async def begin_transaction(self) -> None:
        """Begin a new transaction."""
        await self._initialize()
        await self._transaction.begin_transaction()

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        await self._initialize()
        await self._transaction.commit_transaction()

    async def rollback_transaction(self) -> None:
        """Roll back the current transaction."""
        await self._initialize()
        await self._transaction.rollback_transaction()