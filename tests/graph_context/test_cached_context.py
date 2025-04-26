from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch, call

import pytest

from graph_context.caching.cached_context import CachedGraphContext, CacheTransactionManager
from graph_context.caching.cache_store import CacheEntry
from graph_context.exceptions import TransactionError, EntityNotFoundError, RelationNotFoundError
from graph_context.event_system import GraphEvent


class TestCacheTransactionManager:
    """Tests for the CacheTransactionManager."""

    @pytest.fixture
    def transaction_manager(self):
        """Create a CacheTransactionManager with mocked dependencies."""
        base_context = AsyncMock()
        cache_manager = MagicMock()
        cache_manager.store_manager = MagicMock()
        cache_manager.store_manager.clear_all = AsyncMock()
        cache_manager.handle_event = AsyncMock()

        manager = CacheTransactionManager(base_context, cache_manager)
        yield manager

    def test_initial_state(self, transaction_manager):
        """Test initial transaction state."""
        assert not transaction_manager.is_in_transaction()

    def test_check_transaction_no_transaction(self, transaction_manager):
        """Test check_transaction when no transaction is in progress."""
        # Should raise when transaction is required but not active
        with pytest.raises(TransactionError) as exc_info:
            transaction_manager.check_transaction(required=True)
        assert "Operation requires an active transaction" in str(exc_info.value)

        # Should not raise when transaction is not required and not active
        transaction_manager.check_transaction(required=False)  # Should not raise

    def test_check_transaction_with_transaction(self, transaction_manager):
        """Test check_transaction when a transaction is in progress."""
        # Set transaction state manually for testing
        transaction_manager._in_transaction = True

        # Should not raise when transaction is required and active
        transaction_manager.check_transaction(required=True)  # Should not raise

        # Should raise when transaction is not required but is active
        with pytest.raises(TransactionError) as exc_info:
            transaction_manager.check_transaction(required=False)
        assert "Operation cannot be performed in a transaction" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_begin_transaction(self, transaction_manager):
        """Test begin_transaction."""
        # Begin transaction
        await transaction_manager.begin_transaction()

        # Verify state changes
        assert transaction_manager.is_in_transaction()

        # Verify calls to dependencies
        transaction_manager._base_context.begin_transaction.assert_called_once()
        transaction_manager._cache_manager.store_manager.clear_all.assert_called_once()
        transaction_manager._cache_manager.handle_event.assert_called_once()

        # The event should be TRANSACTION_BEGIN
        event_context = transaction_manager._cache_manager.handle_event.call_args[0][0]
        assert event_context.event == GraphEvent.TRANSACTION_BEGIN

    @pytest.mark.asyncio
    async def test_begin_transaction_already_active(self, transaction_manager):
        """Test begin_transaction when a transaction is already active."""
        # Set transaction state manually for testing
        transaction_manager._in_transaction = True

        # Try to begin another transaction
        with pytest.raises(TransactionError) as exc_info:
            await transaction_manager.begin_transaction()
        assert "Transaction already in progress" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_commit_transaction(self, transaction_manager):
        """Test commit_transaction."""
        # Set transaction state manually for testing
        transaction_manager._in_transaction = True

        # Commit transaction
        await transaction_manager.commit_transaction()

        # Verify state changes
        assert not transaction_manager.is_in_transaction()

        # Verify calls to dependencies
        transaction_manager._base_context.commit_transaction.assert_called_once()
        transaction_manager._cache_manager.store_manager.clear_all.assert_called_once()
        transaction_manager._cache_manager.handle_event.assert_called_once()

        # The event should be TRANSACTION_COMMIT
        event_context = transaction_manager._cache_manager.handle_event.call_args[0][0]
        assert event_context.event == GraphEvent.TRANSACTION_COMMIT

    @pytest.mark.asyncio
    async def test_commit_transaction_no_active(self, transaction_manager):
        """Test commit_transaction when no transaction is active."""
        # Try to commit without active transaction
        with pytest.raises(TransactionError) as exc_info:
            await transaction_manager.commit_transaction()
        assert "No transaction in progress" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rollback_transaction(self, transaction_manager):
        """Test rollback_transaction."""
        # Set transaction state manually for testing
        transaction_manager._in_transaction = True

        # Rollback transaction
        await transaction_manager.rollback_transaction()

        # Verify state changes
        assert not transaction_manager.is_in_transaction()

        # Verify calls to dependencies
        transaction_manager._base_context.rollback_transaction.assert_called_once()
        transaction_manager._cache_manager.store_manager.clear_all.assert_called_once()
        transaction_manager._cache_manager.handle_event.assert_called_once()

        # The event should be TRANSACTION_ROLLBACK
        event_context = transaction_manager._cache_manager.handle_event.call_args[0][0]
        assert event_context.event == GraphEvent.TRANSACTION_ROLLBACK

    @pytest.mark.asyncio
    async def test_rollback_transaction_no_active(self, transaction_manager):
        """Test rollback_transaction when no transaction is active."""
        # Try to rollback without active transaction
        with pytest.raises(TransactionError) as exc_info:
            await transaction_manager.rollback_transaction()
        assert "No transaction in progress" in str(exc_info.value)


class TestCachedGraphContextTransactions:
    """Tests for transaction handling in CachedGraphContext."""

    @pytest.fixture
    def cached_context(self):
        """Create a CachedGraphContext with mocked dependencies."""
        base_context = AsyncMock()
        cache_manager = MagicMock()
        cache_manager.store_manager = MagicMock()
        cache_manager.store_manager.clear_all = AsyncMock()
        cache_manager.handle_event = AsyncMock()

        # Create context with mocked dependencies
        context = CachedGraphContext(base_context, cache_manager)

        # Mock transaction methods for testing
        context._transaction.begin_transaction = AsyncMock()
        context._transaction.commit_transaction = AsyncMock()
        context._transaction.rollback_transaction = AsyncMock()
        context._transaction.is_in_transaction = MagicMock()

        yield context

    @pytest.mark.asyncio
    async def test_begin_transaction(self, cached_context):
        """Test that begin_transaction delegates to transaction manager."""
        # Initialize context
        cached_context._initialized = True

        # Begin transaction
        await cached_context.begin_transaction()

        # Verify delegation to transaction manager
        cached_context._transaction.begin_transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_commit_transaction(self, cached_context):
        """Test that commit_transaction delegates to transaction manager."""
        # Initialize context
        cached_context._initialized = True

        # Commit transaction
        await cached_context.commit_transaction()

        # Verify delegation to transaction manager
        cached_context._transaction.commit_transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_transaction(self, cached_context):
        """Test that rollback_transaction delegates to transaction manager."""
        # Initialize context
        cached_context._initialized = True

        # Rollback transaction
        await cached_context.rollback_transaction()

        # Verify delegation to transaction manager
        cached_context._transaction.rollback_transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_active_transaction(self, cached_context):
        """Test cleanup behavior with active transaction."""
        # Set active transaction
        cached_context._transaction.is_in_transaction.return_value = True

        # Call cleanup
        await cached_context.cleanup()

        # Verify transaction rollback
        cached_context._transaction.rollback_transaction.assert_called_once()

        # Verify base cleanup
        cached_context._base.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_without_transaction(self, cached_context):
        """Test cleanup behavior without active transaction."""
        # Set no active transaction
        cached_context._transaction.is_in_transaction.return_value = False

        # Call cleanup
        await cached_context.cleanup()

        # Verify no transaction rollback
        cached_context._transaction.rollback_transaction.assert_not_called()

        # Verify base cleanup
        cached_context._base.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_operations_in_transaction(self, cached_context):
        """Test entity operations during transaction."""
        # Initialize context
        cached_context._initialized = True

        # Set active transaction state
        cached_context._transaction.is_in_transaction.return_value = True

        # Mock base context methods
        entity = {"id": "test1", "type": "test_type", "properties": {"name": "Test"}}
        cached_context._base.get_entity.return_value = entity

        # Get entity during transaction
        result = await cached_context.get_entity("test1")

        # Should return entity from base without caching
        assert result == entity
        cached_context._base.get_entity.assert_called_once_with("test1")

        # Should not attempt to check cache
        assert not hasattr(cached_context._cache_manager.store_manager.get_entity_store(), "get") or \
               not cached_context._cache_manager.store_manager.get_entity_store().get.called

    @pytest.mark.asyncio
    async def test_entity_creation_in_transaction(self, cached_context):
        """Test entity creation during transaction."""
        # Initialize context
        cached_context._initialized = True

        # Set active transaction state
        cached_context._transaction.is_in_transaction.return_value = True

        # Mock base context methods
        cached_context._base.create_entity.return_value = "test1"

        # Create entity during transaction
        entity_id = await cached_context.create_entity("test_type", {"name": "Test"})

        # Should delegate to base context
        assert entity_id == "test1"
        cached_context._base.create_entity.assert_called_once_with("test_type", {"name": "Test"})

        # Should not try to cache the new entity
        assert not hasattr(cached_context._cache_manager.store_manager.get_entity_store(), "set") or \
               not cached_context._cache_manager.store_manager.get_entity_store().set.called


class TestCachedGraphContextCaching:
    """Tests for caching behavior in CachedGraphContext."""

    @pytest.fixture
    def cached_context(self):
        """Create a CachedGraphContext with mocked dependencies for caching tests."""
        base_context = AsyncMock()
        cache_manager = MagicMock()

        # Set up cache stores
        entity_store = AsyncMock()
        relation_store = AsyncMock()
        query_store = AsyncMock()
        traversal_store = AsyncMock()

        cache_manager.store_manager = MagicMock()
        cache_manager.store_manager.get_entity_store.return_value = entity_store
        cache_manager.store_manager.get_relation_store.return_value = relation_store
        cache_manager.store_manager.get_query_store.return_value = query_store
        cache_manager.store_manager.get_traversal_store.return_value = traversal_store
        cache_manager.handle_event = AsyncMock()
        cache_manager._hash_query = MagicMock(return_value="test_hash")

        # Ensure clear_all is properly mocked
        cache_manager.store_manager.clear_all = AsyncMock()

        # Mock cache enable/disable methods
        cache_manager.enable = MagicMock()
        cache_manager.disable = MagicMock()

        context = CachedGraphContext(base_context, cache_manager)
        context._initialized = True  # Skip initialization step

        # Ensure transaction state is not active for these tests
        context._transaction.is_in_transaction = MagicMock(return_value=False)

        yield context

    @pytest.mark.asyncio
    async def test_enable_disable_caching(self, cached_context):
        """Test enabling and disabling caching functionality."""
        # Test enable caching
        cached_context.enable_caching()
        cached_context._cache_manager.enable.assert_called_once()

        # Test disable caching
        cached_context.disable_caching()
        cached_context._cache_manager.disable.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entity_cache_hit(self, cached_context):
        """Test getting an entity with a cache hit."""
        entity = {"id": "test1", "type": "test_type", "properties": {"name": "Test"}}

        # Set up cache hit
        cache_entry = MagicMock()
        cache_entry.value = entity
        cached_context._cache_manager.store_manager.get_entity_store().get.return_value = cache_entry

        # Get entity
        result = await cached_context.get_entity("test1")

        # Should return entity from cache
        assert result == entity
        cached_context._cache_manager.store_manager.get_entity_store().get.assert_called_once_with("test1")

        # Should not query base context
        cached_context._base.get_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_entity_cache_miss(self, cached_context):
        """Test getting an entity with a cache miss."""
        entity = {"id": "test1", "type": "test_type", "properties": {"name": "Test"}}

        # Set up cache miss
        cached_context._cache_manager.store_manager.get_entity_store().get.return_value = None
        cached_context._base.get_entity.return_value = entity

        # Get entity
        result = await cached_context.get_entity("test1")

        # Should return entity from base context
        assert result == entity
        cached_context._cache_manager.store_manager.get_entity_store().get.assert_called_once_with("test1")
        cached_context._base.get_entity.assert_called_once_with("test1")

        # Should cache the result
        cached_context._cache_manager.store_manager.get_entity_store().set.assert_called_once()
        # Verify entity_type is extracted from the entity
        assert cached_context._cache_manager.store_manager.get_entity_store().set.call_args[0][1].entity_type == "test_type"

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, cached_context):
        """Test getting a non-existent entity."""
        # Set up cache miss and base context miss
        cached_context._cache_manager.store_manager.get_entity_store().get.return_value = None
        cached_context._base.get_entity.return_value = None

        # Get non-existent entity
        with pytest.raises(EntityNotFoundError) as exc_info:
            await cached_context.get_entity("test1")

        assert "Entity test1 not found" in str(exc_info.value)

        # Should not cache anything
        cached_context._cache_manager.store_manager.get_entity_store().set.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_relation_cache_hit(self, cached_context):
        """Test getting a relation with a cache hit."""
        relation = {"id": "rel1", "type": "relates_to", "from": "entity1", "to": "entity2"}

        # Set up cache hit
        cache_entry = MagicMock()
        cache_entry.value = relation
        cached_context._cache_manager.store_manager.get_relation_store().get.return_value = cache_entry

        # Get relation
        result = await cached_context.get_relation("rel1")

        # Should return relation from cache
        assert result == relation
        cached_context._cache_manager.store_manager.get_relation_store().get.assert_called_once_with("rel1")

        # Should not query base context
        cached_context._base.get_relation.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_relation_cache_miss(self, cached_context):
        """Test getting a relation with a cache miss."""
        relation = {"id": "rel1", "type": "relates_to", "from": "entity1", "to": "entity2"}

        # Set up cache miss
        cached_context._cache_manager.store_manager.get_relation_store().get.return_value = None
        cached_context._base.get_relation.return_value = relation

        # Get relation
        result = await cached_context.get_relation("rel1")

        # Should return relation from base context
        assert result == relation
        cached_context._cache_manager.store_manager.get_relation_store().get.assert_called_once_with("rel1")
        cached_context._base.get_relation.assert_called_once_with("rel1")

        # Should cache the result
        cached_context._cache_manager.store_manager.get_relation_store().set.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_relation_not_found(self, cached_context):
        """Test getting a non-existent relation."""
        # Set up cache miss and base context miss
        cached_context._cache_manager.store_manager.get_relation_store().get.return_value = None
        cached_context._base.get_relation.return_value = None

        # Get non-existent relation
        with pytest.raises(RelationNotFoundError) as exc_info:
            await cached_context.get_relation("rel1")

        assert "Relation rel1 not found" in str(exc_info.value)

        # Should not cache anything
        cached_context._cache_manager.store_manager.get_relation_store().set.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_cache_hit(self, cached_context):
        """Test query with a cache hit."""
        query_spec = {"entity_type": "test_type", "filters": []}
        results = [{"id": "entity1"}, {"id": "entity2"}]

        # Set up cache hit
        cache_entry = MagicMock()
        cache_entry.value = results
        cached_context._cache_manager.store_manager.get_query_store().get.return_value = cache_entry

        # Execute query
        query_results = await cached_context.query(query_spec)

        # Should return results from cache
        assert query_results == results
        cached_context._cache_manager.store_manager.get_query_store().get.assert_called_once_with("test_hash")

        # Should not query base context
        cached_context._base.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_cache_miss(self, cached_context):
        """Test query with a cache miss."""
        query_spec = {"entity_type": "test_type", "filters": []}
        results = [{"id": "entity1"}, {"id": "entity2"}]

        # Set up cache miss
        cached_context._cache_manager.store_manager.get_query_store().get.return_value = None
        cached_context._base.query.return_value = results

        # Execute query
        query_results = await cached_context.query(query_spec)

        # Should return results from base context
        assert query_results == results
        cached_context._cache_manager.store_manager.get_query_store().get.assert_called_once_with("test_hash")
        cached_context._base.query.assert_called_once_with(query_spec)

        # Should cache the result
        cached_context._cache_manager.store_manager.get_query_store().set.assert_called_once()

        # Should notify cache manager about query execution
        cached_context._cache_manager.handle_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_traverse_cache_hit(self, cached_context):
        """Test traversal with a cache hit."""
        start_entity = "entity1"
        traversal_spec = {"relation_type": "relates_to", "direction": "outgoing"}
        results = [{"id": "entity2"}, {"id": "entity3"}]

        # Set up cache hit
        cache_entry = MagicMock()
        cache_entry.value = results
        cached_context._cache_manager.store_manager.get_traversal_store().get.return_value = cache_entry

        # Execute traversal
        traversal_results = await cached_context.traverse(start_entity, traversal_spec)

        # Should return results from cache
        assert traversal_results == results
        cached_context._cache_manager.store_manager.get_traversal_store().get.assert_called_once_with("test_hash")

        # Should not query base context
        cached_context._base.traverse.assert_not_called()

    @pytest.mark.asyncio
    async def test_traverse_cache_miss(self, cached_context):
        """Test traversal with a cache miss."""
        start_entity = "entity1"
        traversal_spec = {"relation_type": "relates_to", "direction": "outgoing"}
        results = [{"id": "entity2"}, {"id": "entity3"}]

        # Set up cache miss
        cached_context._cache_manager.store_manager.get_traversal_store().get.return_value = None
        cached_context._base.traverse.return_value = results

        # Execute traversal
        traversal_results = await cached_context.traverse(start_entity, traversal_spec)

        # Should return results from base context
        assert traversal_results == results
        cached_context._cache_manager.store_manager.get_traversal_store().get.assert_called_once_with("test_hash")
        cached_context._base.traverse.assert_called_once_with(start_entity, traversal_spec)

        # Should cache the result
        cached_context._cache_manager.store_manager.get_traversal_store().set.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_entity_cache_invalidation(self, cached_context):
        """Test update_entity with cache invalidation."""
        entity_id = "entity1"
        properties = {"name": "Updated Name"}

        # Set up successful update
        cached_context._base.update_entity.return_value = True

        # Update entity
        result = await cached_context.update_entity(entity_id, properties)

        # Should succeed
        assert result is True

        # Should delete from entity cache
        cached_context._cache_manager.store_manager.get_entity_store().delete.assert_called_once_with(entity_id)

        # Should notify cache manager about write event
        cached_context._cache_manager.handle_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_entity_cache_invalidation(self, cached_context):
        """Test delete_entity with cache invalidation."""
        entity_id = "entity1"

        # Set up successful delete
        cached_context._base.delete_entity.return_value = True

        # Delete entity
        result = await cached_context.delete_entity(entity_id)

        # Should succeed
        assert result is True

        # Should delete from entity cache
        cached_context._cache_manager.store_manager.get_entity_store().delete.assert_called_once_with(entity_id)

        # Should notify cache manager about delete event
        cached_context._cache_manager.handle_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_relation_cache_invalidation(self, cached_context):
        """Test update_relation with cache invalidation."""
        relation_id = "rel1"
        properties = {"weight": 5}

        # Set up successful update
        cached_context._base.update_relation.return_value = True

        # Update relation
        result = await cached_context.update_relation(relation_id, properties)

        # Should succeed
        assert result is True

        # Should delete from relation cache
        cached_context._cache_manager.store_manager.get_relation_store().delete.assert_called_once_with(relation_id)

        # Should notify cache manager about write event
        cached_context._cache_manager.handle_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_relation_cache_invalidation(self, cached_context):
        """Test delete_relation with cache invalidation."""
        relation_id = "rel1"

        # Set up successful delete
        cached_context._base.delete_relation.return_value = True

        # Delete relation
        result = await cached_context.delete_relation(relation_id)

        # Should succeed
        assert result is True

        # Should delete from relation cache
        cached_context._cache_manager.store_manager.get_relation_store().delete.assert_called_once_with(relation_id)

        # Should notify cache manager about delete event
        cached_context._cache_manager.handle_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_entity_caching(self, cached_context):
        """Test create_entity with caching of new entity."""
        entity_type = "test_type"
        properties = {"name": "New Entity"}
        entity_id = "new_entity_id"
        entity = {"id": entity_id, "type": entity_type, "properties": properties}

        # Set up create and get responses
        cached_context._base.create_entity.return_value = entity_id
        cached_context._base.get_entity.return_value = entity

        # Create entity
        result = await cached_context.create_entity(entity_type, properties)

        # Should return the entity id
        assert result == entity_id

        # Should cache the new entity
        cached_context._base.get_entity.assert_called_once_with(entity_id)
        cached_context._cache_manager.store_manager.get_entity_store().set.assert_called_once()

        # Should notify cache manager about write event
        cached_context._cache_manager.handle_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_relation_caching(self, cached_context):
        """Test create_relation with caching of new relation."""
        relation_type = "relates_to"
        from_entity = "entity1"
        to_entity = "entity2"
        properties = {"weight": 1}
        relation_id = "new_relation_id"
        relation = {"id": relation_id, "type": relation_type, "from": from_entity, "to": to_entity, "properties": properties}

        # Set up create and get responses
        cached_context._base.create_relation.return_value = relation_id
        cached_context._base.get_relation.return_value = relation

        # Create relation
        result = await cached_context.create_relation(relation_type, from_entity, to_entity, properties)

        # Should return the relation id
        assert result == relation_id

        # Should cache the new relation
        cached_context._base.get_relation.assert_called_once_with(relation_id)
        cached_context._cache_manager.store_manager.get_relation_store().set.assert_called_once()

        # Should notify cache manager about write event
        cached_context._cache_manager.handle_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_create_entities(self, cached_context):
        """Test bulk_create_entities with event notification."""
        entity_type = "test_type"
        entities = [
            {"name": "Entity 1"},
            {"name": "Entity 2"}
        ]
        entity_ids = ["entity1", "entity2"]

        # Set up bulk create response
        cached_context._base.bulk_create_entities.return_value = entity_ids

        # Bulk create entities
        result = await cached_context.bulk_create_entities(entity_type, entities)

        # Should return the entity ids
        assert result == entity_ids

        # Should notify cache manager about bulk write event
        cached_context._cache_manager.handle_event.assert_called_once()
        event_context = cached_context._cache_manager.handle_event.call_args[0][0]
        assert event_context.event == GraphEvent.ENTITY_BULK_WRITE
        assert event_context.data["entity_ids"] == entity_ids
        assert event_context.metadata.entity_type == entity_type

    @pytest.mark.asyncio
    async def test_bulk_update_entities(self, cached_context):
        """Test bulk_update_entities with event notification."""
        entity_type = "test_type"
        entities = [
            {"id": "entity1", "name": "Updated Entity 1"},
            {"id": "entity2", "name": "Updated Entity 2"}
        ]

        # Bulk update entities
        await cached_context.bulk_update_entities(entity_type, entities)

        # Should notify cache manager about bulk write event
        cached_context._cache_manager.handle_event.assert_called_once()
        event_context = cached_context._cache_manager.handle_event.call_args[0][0]
        assert event_context.event == GraphEvent.ENTITY_BULK_WRITE
        assert event_context.data["entity_ids"] == ["entity1", "entity2"]
        assert event_context.metadata.entity_type == entity_type

    @pytest.mark.asyncio
    async def test_bulk_delete_entities(self, cached_context):
        """Test bulk_delete_entities with event notification."""
        entity_type = "test_type"
        entity_ids = ["entity1", "entity2"]

        # Bulk delete entities
        await cached_context.bulk_delete_entities(entity_type, entity_ids)

        # Should notify cache manager about bulk delete event
        cached_context._cache_manager.handle_event.assert_called_once()
        event_context = cached_context._cache_manager.handle_event.call_args[0][0]
        assert event_context.event == GraphEvent.ENTITY_BULK_DELETE
        assert event_context.data["entity_ids"] == entity_ids
        assert event_context.metadata.entity_type == entity_type

    @pytest.mark.asyncio
    async def test_bulk_operations_for_relations(self, cached_context):
        """Test bulk operations for relations with event notifications."""
        relation_type = "relates_to"

        # Test bulk create relations
        relations = [
            {"from": "entity1", "to": "entity2"},
            {"from": "entity3", "to": "entity4"}
        ]
        relation_ids = ["rel1", "rel2"]
        cached_context._base.bulk_create_relations.return_value = relation_ids

        result = await cached_context.bulk_create_relations(relation_type, relations)
        assert result == relation_ids

        # Should notify about bulk relation write
        event_context = cached_context._cache_manager.handle_event.call_args[0][0]
        assert event_context.event == GraphEvent.RELATION_BULK_WRITE
        assert event_context.data["relation_ids"] == relation_ids

        # Reset mock for next test
        cached_context._cache_manager.handle_event.reset_mock()

        # Test bulk update relations
        update_relations = [
            {"id": "rel1", "weight": 5},
            {"id": "rel2", "weight": 10}
        ]

        await cached_context.bulk_update_relations(relation_type, update_relations)

        # Should notify about bulk relation update
        event_context = cached_context._cache_manager.handle_event.call_args[0][0]
        assert event_context.event == GraphEvent.RELATION_BULK_WRITE
        assert event_context.data["relation_ids"] == ["rel1", "rel2"]

        # Reset mock for next test
        cached_context._cache_manager.handle_event.reset_mock()

        # Test bulk delete relations
        delete_relation_ids = ["rel1", "rel2"]

        await cached_context.bulk_delete_relations(relation_type, delete_relation_ids)

        # Should notify about bulk relation delete
        event_context = cached_context._cache_manager.handle_event.call_args[0][0]
        assert event_context.event == GraphEvent.RELATION_BULK_DELETE
        assert event_context.data["relation_ids"] == delete_relation_ids

    @pytest.mark.asyncio
    async def test_initialize(self, cached_context):
        """Test the initialization of event subscriptions."""
        # Reset initialization flag to test initialization
        cached_context._initialized = False

        # Set up the base context with an _events attribute
        cached_context._base._events = MagicMock()
        cached_context._base._events.subscribe = AsyncMock()

        # Call a method that triggers initialization
        entity = {"id": "test1", "type": "test_type", "properties": {"name": "Test"}}
        cached_context._cache_manager.store_manager.get_entity_store().get.return_value = None
        cached_context._base.get_entity.return_value = entity

        await cached_context.get_entity("test1")

        # Verify that initialization happened
        assert cached_context._initialized

        # Verify subscriptions were set up for various events
        assert cached_context._base._events.subscribe.call_count >= 16  # At least one for each event type

        # Verify some key event subscriptions
        events_subscribed = [call[0][0] for call in cached_context._base._events.subscribe.call_args_list]
        assert GraphEvent.ENTITY_READ in events_subscribed
        assert GraphEvent.ENTITY_WRITE in events_subscribed
        assert GraphEvent.RELATION_READ in events_subscribed
        assert GraphEvent.QUERY_EXECUTED in events_subscribed
        assert GraphEvent.TRANSACTION_BEGIN in events_subscribed

    @pytest.mark.asyncio
    async def test_initialize_direct(self, cached_context):
        """Test the _initialize method directly."""
        # Reset initialization flag
        cached_context._initialized = False

        # Set up the base context with an _events attribute
        cached_context._base._events = MagicMock()
        cached_context._base._events.subscribe = AsyncMock()

        # Directly call the _initialize method
        await cached_context._initialize()

        # Verify that initialization happened
        assert cached_context._initialized

        # Verify all 17 event subscriptions were set up
        assert cached_context._base._events.subscribe.call_count == 17

        # Check that each specific event was subscribed to
        all_events = [
            GraphEvent.ENTITY_READ,
            GraphEvent.ENTITY_WRITE,
            GraphEvent.ENTITY_BULK_WRITE,
            GraphEvent.ENTITY_DELETE,
            GraphEvent.ENTITY_BULK_DELETE,
            GraphEvent.RELATION_READ,
            GraphEvent.RELATION_WRITE,
            GraphEvent.RELATION_BULK_WRITE,
            GraphEvent.RELATION_DELETE,
            GraphEvent.RELATION_BULK_DELETE,
            GraphEvent.QUERY_EXECUTED,
            GraphEvent.TRAVERSAL_EXECUTED,
            GraphEvent.SCHEMA_MODIFIED,
            GraphEvent.TYPE_MODIFIED,
            GraphEvent.TRANSACTION_BEGIN,
            GraphEvent.TRANSACTION_COMMIT,
            GraphEvent.TRANSACTION_ROLLBACK
        ]

        # Verify each event was subscribed
        events_subscribed = [call[0][0] for call in cached_context._base._events.subscribe.call_args_list]
        for event in all_events:
            assert event in events_subscribed, f"Event {event} was not subscribed"

        # Verify the event handler is the cache manager's handle_event
        for call_args in cached_context._base._events.subscribe.call_args_list:
            assert call_args[0][1] == cached_context._cache_manager.handle_event

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, cached_context):
        """Test _initialize when already initialized."""
        # Set initialization flag to True
        cached_context._initialized = True

        # Set up mock to detect if called
        cached_context._base._events = MagicMock()
        cached_context._base._events.subscribe = AsyncMock()

        # Call _initialize
        await cached_context._initialize()

        # No subscriptions should be attempted
        cached_context._base._events.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_no_events(self, cached_context):
        """Test _initialize when base context has no _events attribute."""
        # Reset initialization flag
        cached_context._initialized = False

        # Remove _events attribute from base context
        if hasattr(cached_context._base, '_events'):
            delattr(cached_context._base, '_events')

        # Call _initialize
        await cached_context._initialize()

        # Should still mark as initialized
        assert cached_context._initialized

        # No error should occur despite missing _events

    @pytest.mark.asyncio
    async def test_get_entity_not_found_in_transaction(self, cached_context):
        """Test getting a non-existent entity during a transaction."""
        # Set transaction state to active
        cached_context._transaction.is_in_transaction.return_value = True

        # Set up base context to return None
        cached_context._base.get_entity.return_value = None

        # Get non-existent entity in transaction
        with pytest.raises(EntityNotFoundError) as exc_info:
            await cached_context.get_entity("test1")

        assert "Entity test1 not found" in str(exc_info.value)

        # Should check base context directly
        cached_context._base.get_entity.assert_called_once_with("test1")

        # Should not check cache
        cached_context._cache_manager.store_manager.get_entity_store().get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_relation_not_found_in_transaction(self, cached_context):
        """Test getting a non-existent relation during a transaction."""
        # Set transaction state to active
        cached_context._transaction.is_in_transaction.return_value = True

        # Set up base context to return None
        cached_context._base.get_relation.return_value = None

        # Get non-existent relation in transaction
        with pytest.raises(RelationNotFoundError) as exc_info:
            await cached_context.get_relation("rel1")

        assert "Relation rel1 not found" in str(exc_info.value)

        # Should check base context directly
        cached_context._base.get_relation.assert_called_once_with("rel1")

        # Should not check cache
        cached_context._cache_manager.store_manager.get_relation_store().get.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_empty_results(self, cached_context):
        """Test query with empty results from base context."""
        query_spec = {"entity_type": "test_type", "filters": []}

        # Set up cache miss and empty result from base
        cached_context._cache_manager.store_manager.get_query_store().get.return_value = None
        cached_context._base.query.return_value = None

        # Execute query
        query_results = await cached_context.query(query_spec)

        # Should return empty list
        assert query_results == []
        cached_context._cache_manager.store_manager.get_query_store().get.assert_called_once_with("test_hash")
        cached_context._base.query.assert_called_once_with(query_spec)

    @pytest.mark.asyncio
    async def test_traverse_empty_results(self, cached_context):
        """Test traversal with empty results from base context."""
        start_entity = "entity1"
        traversal_spec = {"relation_type": "relates_to", "direction": "outgoing"}

        # Set up cache miss and empty result from base
        cached_context._cache_manager.store_manager.get_traversal_store().get.return_value = None
        cached_context._base.traverse.return_value = None

        # Execute traversal
        traversal_results = await cached_context.traverse(start_entity, traversal_spec)

        # Should return empty list
        assert traversal_results == []
        cached_context._cache_manager.store_manager.get_traversal_store().get.assert_called_once_with("test_hash")
        cached_context._base.traverse.assert_called_once_with(start_entity, traversal_spec)

    @pytest.mark.asyncio
    async def test_query_in_transaction(self, cached_context):
        """Test query during transaction."""
        # Set transaction state to active
        cached_context._transaction.is_in_transaction.return_value = True

        query_spec = {"entity_type": "test_type", "filters": []}
        results = [{"id": "entity1"}, {"id": "entity2"}]

        # Set up base context result
        cached_context._base.query.return_value = results

        # Execute query
        query_results = await cached_context.query(query_spec)

        # Should return results from base context
        assert query_results == results
        cached_context._base.query.assert_called_once_with(query_spec)

        # Should not check cache
        cached_context._cache_manager.store_manager.get_query_store().get.assert_not_called()

    @pytest.mark.asyncio
    async def test_traverse_in_transaction(self, cached_context):
        """Test traversal during transaction."""
        # Set transaction state to active
        cached_context._transaction.is_in_transaction.return_value = True

        start_entity = "entity1"
        traversal_spec = {"relation_type": "relates_to", "direction": "outgoing"}
        results = [{"id": "entity2"}, {"id": "entity3"}]

        # Set up base context result
        cached_context._base.traverse.return_value = results

        # Execute traversal
        traversal_results = await cached_context.traverse(start_entity, traversal_spec)

        # Should return results from base context
        assert traversal_results == results
        cached_context._base.traverse.assert_called_once_with(start_entity, traversal_spec)

        # Should not check cache
        cached_context._cache_manager.store_manager.get_traversal_store().get.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_entity_failure(self, cached_context):
        """Test update_entity when base update fails."""
        entity_id = "entity1"
        properties = {"name": "Updated Name"}

        # Set up failed update
        cached_context._base.update_entity.return_value = False

        # Update entity
        result = await cached_context.update_entity(entity_id, properties)

        # Should fail
        assert result is False

        # Should not delete from entity cache or notify listeners
        cached_context._cache_manager.store_manager.get_entity_store().delete.assert_not_called()
        cached_context._cache_manager.handle_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_entity_failure(self, cached_context):
        """Test delete_entity when base delete fails."""
        entity_id = "entity1"

        # Set up failed delete
        cached_context._base.delete_entity.return_value = False

        # Delete entity
        result = await cached_context.delete_entity(entity_id)

        # Should fail
        assert result is False

        # Should not delete from entity cache or notify listeners
        cached_context._cache_manager.store_manager.get_entity_store().delete.assert_not_called()
        cached_context._cache_manager.handle_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_relation_failure(self, cached_context):
        """Test update_relation when base update fails."""
        relation_id = "rel1"
        properties = {"weight": 5}

        # Set up failed update
        cached_context._base.update_relation.return_value = False

        # Update relation
        result = await cached_context.update_relation(relation_id, properties)

        # Should fail
        assert result is False

        # Should not delete from relation cache or notify listeners
        cached_context._cache_manager.store_manager.get_relation_store().delete.assert_not_called()
        cached_context._cache_manager.handle_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_relation_failure(self, cached_context):
        """Test delete_relation when base delete fails."""
        relation_id = "rel1"

        # Set up failed delete
        cached_context._base.delete_relation.return_value = False

        # Delete relation
        result = await cached_context.delete_relation(relation_id)

        # Should fail
        assert result is False

        # Should not delete from relation cache or notify listeners
        cached_context._cache_manager.store_manager.get_relation_store().delete.assert_not_called()
        cached_context._cache_manager.handle_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_entity_caching_lines_400_408(self, cached_context):
        """
        Test specifically covering lines 400-408 in cached_context.py where entity caching occurs.
        Tests both transaction and non-transaction paths.
        """
        entity_id = "test_entity_id"
        entity_type = "test_type"
        properties = {"name": "Test Entity"}
        entity = {
            "id": entity_id,
            "type": entity_type,
            "properties": properties
        }

        # Mock the minimum required methods
        cached_context._base.create_entity.return_value = entity_id
        cached_context._base.get_entity.return_value = entity

        # Test 1: When in transaction - should not cache
        cached_context._transaction.is_in_transaction.return_value = True

        result = await cached_context.create_entity(entity_type, properties)
        assert result == entity_id
        cached_context._base.create_entity.assert_called_once_with(entity_type, properties)
        cached_context._base.get_entity.assert_not_called()  # Should not try to get entity
        cached_context._cache_manager.store_manager.get_entity_store().set.assert_not_called()  # Should not cache

        # Reset mocks for next test
        cached_context._base.create_entity.reset_mock()
        cached_context._base.get_entity.reset_mock()
        cached_context._cache_manager.store_manager.get_entity_store().set.reset_mock()

        # Test 2: When not in transaction - should cache
        cached_context._transaction.is_in_transaction.return_value = False

        result = await cached_context.create_entity(entity_type, properties)
        assert result == entity_id

        # Verify exact sequence of operations
        cached_context._base.create_entity.assert_called_once_with(entity_type, properties)
        cached_context._base.get_entity.assert_called_once_with(entity_id)  # Line 401

        # Verify the cache entry creation and storage (lines 404-408)
        cached_context._cache_manager.store_manager.get_entity_store().set.assert_called_once()
        call_args = cached_context._cache_manager.store_manager.get_entity_store().set.call_args[0]
        assert call_args[0] == entity_id  # First arg should be entity_id
        cache_entry = call_args[1]  # Second arg should be CacheEntry
        assert cache_entry.value == entity
        assert cache_entry.entity_type == entity_type

        # Test 3: When not in transaction but get_entity returns None
        cached_context._base.get_entity.return_value = None
        cached_context._base.create_entity.reset_mock()
        cached_context._base.get_entity.reset_mock()
        cached_context._cache_manager.store_manager.get_entity_store().set.reset_mock()

        result = await cached_context.create_entity(entity_type, properties)
        assert result == entity_id
        cached_context._base.create_entity.assert_called_once_with(entity_type, properties)
        cached_context._base.get_entity.assert_called_once_with(entity_id)
        cached_context._cache_manager.store_manager.get_entity_store().set.assert_not_called()  # Should not cache None

    @pytest.mark.asyncio
    async def test_create_relation_caching_lines_474_482(self, cached_context):
        """
        Test specifically covering lines 474-482 in cached_context.py where relation caching occurs.
        Tests both transaction and non-transaction paths.
        """
        relation_type = "test_relation"
        from_entity = "entity1"
        to_entity = "entity2"
        properties = {"weight": 5}
        relation_id = "test_relation_id"
        relation = {
            "id": relation_id,
            "type": relation_type,
            "from": from_entity,
            "to": to_entity,
            "properties": properties
        }

        # Mock the minimum required methods
        cached_context._base.create_relation.return_value = relation_id
        cached_context._base.get_relation.return_value = relation

        # Test 1: When in transaction - should not cache
        cached_context._transaction.is_in_transaction.return_value = True

        result = await cached_context.create_relation(relation_type, from_entity, to_entity, properties)
        assert result == relation_id
        cached_context._base.create_relation.assert_called_once_with(relation_type, from_entity, to_entity, properties)
        cached_context._base.get_relation.assert_not_called()  # Should not try to get relation
        cached_context._cache_manager.store_manager.get_relation_store().set.assert_not_called()  # Should not cache

        # Reset mocks for next test
        cached_context._base.create_relation.reset_mock()
        cached_context._base.get_relation.reset_mock()
        cached_context._cache_manager.store_manager.get_relation_store().set.reset_mock()

        # Test 2: When not in transaction - should cache
        cached_context._transaction.is_in_transaction.return_value = False

        result = await cached_context.create_relation(relation_type, from_entity, to_entity, properties)
        assert result == relation_id

        # Verify exact sequence of operations
        cached_context._base.create_relation.assert_called_once_with(relation_type, from_entity, to_entity, properties)
        cached_context._base.get_relation.assert_called_once_with(relation_id)  # Line 475

        # Verify the cache entry creation and storage (lines 478-482)
        cached_context._cache_manager.store_manager.get_relation_store().set.assert_called_once()
        call_args = cached_context._cache_manager.store_manager.get_relation_store().set.call_args[0]
        assert call_args[0] == relation_id  # First arg should be relation_id
        cache_entry = call_args[1]  # Second arg should be CacheEntry
        assert cache_entry.value == relation
        assert cache_entry.relation_type == relation_type

        # Test 3: When not in transaction but get_relation returns None
        cached_context._base.get_relation.return_value = None
        cached_context._base.create_relation.reset_mock()
        cached_context._base.get_relation.reset_mock()
        cached_context._cache_manager.store_manager.get_relation_store().set.reset_mock()

        result = await cached_context.create_relation(relation_type, from_entity, to_entity, properties)
        assert result == relation_id
        cached_context._base.create_relation.assert_called_once_with(relation_type, from_entity, to_entity, properties)
        cached_context._base.get_relation.assert_called_once_with(relation_id)
        cached_context._cache_manager.store_manager.get_relation_store().set.assert_not_called()  # Should not cache None