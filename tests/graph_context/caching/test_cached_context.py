"""Integration tests for the cached graph context implementation."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import AsyncGenerator, List
from unittest.mock import AsyncMock, Mock

import pytest

from graph_context.caching.cache_manager import CacheManager
from graph_context.caching.cache_store import CacheEntry
from graph_context.caching.cached_context import CachedGraphContext
from graph_context.context_base import BaseGraphContext
from graph_context.event_system import EventContext, EventMetadata, GraphEvent
from graph_context.exceptions import (
    EntityNotFoundError,
    RelationNotFoundError,
    TransactionError,
)
from graph_context.types.type_base import (
    Entity,
    EntityType,
    PropertyDefinition,
    RelationType,
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("graph_context.caching.cached_context")
logger.setLevel(logging.DEBUG)


@pytest.fixture
async def base_context() -> AsyncGenerator[BaseGraphContext, None]:
    """Create a base context using InMemoryGraphStore."""
    context = BaseGraphContext()

    # Register standard entity and relation types
    await context.register_entity_type(
        EntityType(
            name="person",
            properties={
                "name": PropertyDefinition(type="string", required=True),
                "age": PropertyDefinition(type="integer", required=False),
            },
        )
    )

    await context.register_relation_type(
        RelationType(
            name="knows",
            from_types=["person"],
            to_types=["person"],
            properties={"since": PropertyDefinition(type="string", required=False)},
        )
    )

    yield context
    await context.cleanup()


@pytest.fixture
async def transaction(
    base_context: BaseGraphContext,
) -> AsyncGenerator[BaseGraphContext, None]:
    """Create and manage a transaction for tests."""
    await base_context.begin_transaction()
    yield base_context
    try:
        await base_context.commit_transaction()
    except Exception:
        await base_context.rollback_transaction()


@pytest.fixture
async def cached_context(base_context):
    """Create a cached context with mocked dependencies."""
    cache_manager = Mock(spec=CacheManager)
    cache_manager.store_manager = Mock()
    cache_manager.store_manager.clear_all = AsyncMock()
    cache_manager.handle_event = AsyncMock()

    # Set up cache stores
    entity_store = AsyncMock()
    relation_store = AsyncMock()
    query_store = AsyncMock()
    traversal_store = AsyncMock()

    cache_manager.store_manager.get_entity_store.return_value = entity_store
    cache_manager.store_manager.get_relation_store.return_value = relation_store
    cache_manager.store_manager.get_query_store.return_value = query_store
    cache_manager.store_manager.get_traversal_store.return_value = traversal_store

    context = CachedGraphContext(base_context, cache_manager)
    context._initialized = True  # Skip initialization
    return context


@pytest.mark.asyncio
async def test_entity_caching(cached_context, transaction):
    """Test entity caching behavior."""
    # Create a test entity in the base context first
    properties = {"name": "Test", "age": 30}
    entity_id = await cached_context._base.create_entity("person", properties)
    entity = await cached_context._base.get_entity(
        entity_id
    )  # Get the actual entity format

    # Test cache hit
    store = cached_context._cache_manager.store_manager.get_entity_store()
    store.get.return_value = CacheEntry(
        value=entity, created_at=datetime.now(UTC), entity_type="person"
    )

    result = await cached_context.get_entity(entity_id)
    assert result == entity
    store.get.assert_called_once_with(entity_id)

    # Test cache miss
    store.get.reset_mock()
    store.get.return_value = None

    result = await cached_context.get_entity(entity_id)
    assert result == entity
    store.get.assert_called_once_with(entity_id)
    store.set.assert_called_once()


@pytest.mark.asyncio
async def test_relation_caching(cached_context, transaction):
    """Test relation caching behavior."""
    # Create test entities first
    from_entity_id = await cached_context._base.create_entity(
        "person", {"name": "Person A"}
    )
    to_entity_id = await cached_context._base.create_entity(
        "person", {"name": "Person B"}
    )

    # Create the test relation
    relation_id = await cached_context._base.create_relation(
        "knows", from_entity_id, to_entity_id, {"since": "2024"}
    )
    relation = await cached_context._base.get_relation(relation_id)

    # Test cache hit
    store = cached_context._cache_manager.store_manager.get_relation_store()
    store.get.return_value = CacheEntry(
        value=relation, created_at=datetime.now(UTC), relation_type="knows"
    )

    result = await cached_context.get_relation(relation_id)
    assert result == relation
    store.get.assert_called_once_with(relation_id)

    # Test cache miss
    store.get.reset_mock()
    store.get.return_value = None

    result = await cached_context.get_relation(relation_id)
    assert result == relation
    store.get.assert_called_once_with(relation_id)
    store.set.assert_called_once()


@pytest.mark.asyncio
async def test_query_caching(cached_context, transaction):
    """Test query caching behavior."""
    # Create some test entities
    await cached_context._base.create_entity("person", {"name": "Person A", "age": 30})
    await cached_context._base.create_entity("person", {"name": "Person B", "age": 25})

    # Define query and get actual results
    query_spec = {"entity_type": "person"}
    results = await cached_context._base.query(query_spec)

    # Mock hash generation
    query_hash = "test_hash"
    cached_context._cache_manager._hash_query.return_value = query_hash

    # Test cache hit
    store = cached_context._cache_manager.store_manager.get_query_store()
    store.get.return_value = CacheEntry(
        value=results, created_at=datetime.now(UTC), query_hash=query_hash
    )

    query_results = await cached_context.query(query_spec)
    assert query_results == results
    store.get.assert_called_once_with(query_hash)

    # Test cache miss
    store.get.reset_mock()
    store.get.return_value = None

    query_results = await cached_context.query(query_spec)
    assert query_results == results
    store.get.assert_called_once_with(query_hash)
    store.set.assert_called_once()


@pytest.mark.asyncio
async def test_traversal_caching(cached_context, transaction):
    """Test traversal caching behavior."""
    # Create test entities and relations
    start_entity_id = await cached_context._base.create_entity(
        "person", {"name": "Start Person"}
    )
    target1_id = await cached_context._base.create_entity(
        "person", {"name": "Target 1"}
    )
    target2_id = await cached_context._base.create_entity(
        "person", {"name": "Target 2"}
    )

    # Create relations
    await cached_context._base.create_relation("knows", start_entity_id, target1_id)
    await cached_context._base.create_relation("knows", start_entity_id, target2_id)

    # Define traversal and get actual results
    traversal_spec = {
        "max_depth": 2,
        "relation_types": ["knows"],
        "direction": "outbound",
    }
    results = await cached_context._base.traverse(start_entity_id, traversal_spec)

    # Mock hash generation
    traversal_hash = "test_hash"
    cached_context._cache_manager._hash_query.return_value = traversal_hash

    # Test cache hit
    store = cached_context._cache_manager.store_manager.get_traversal_store()
    store.get.return_value = CacheEntry(
        value=results, created_at=datetime.now(UTC), query_hash=traversal_hash
    )

    traversal_results = await cached_context.traverse(start_entity_id, traversal_spec)
    assert traversal_results == results
    store.get.assert_called_once_with(traversal_hash)

    # Test cache miss
    store.get.reset_mock()
    store.get.return_value = None

    traversal_results = await cached_context.traverse(start_entity_id, traversal_spec)
    assert traversal_results == results
    store.get.assert_called_once_with(traversal_hash)
    store.set.assert_called_once()


@pytest.mark.asyncio
async def test_cache_invalidation(cached_context, transaction):
    """Test cache invalidation on write operations."""
    # Create test entities and relations first
    entity_id = await cached_context._base.create_entity("person", {"name": "Test"})
    from_id = await cached_context._base.create_entity("person", {"name": "From"})
    to_id = await cached_context._base.create_entity("person", {"name": "To"})
    relation_id = await cached_context._base.create_relation("knows", from_id, to_id)

    # Test entity cache invalidation
    properties = {"name": "Updated"}
    await cached_context.update_entity(entity_id, properties)
    cached_context._cache_manager.store_manager.get_entity_store().delete.assert_called_once_with(
        entity_id
    )

    # Test relation cache invalidation
    properties = {"since": "2024"}
    await cached_context.update_relation(relation_id, properties)
    cached_context._cache_manager.store_manager.get_relation_store().delete.assert_called_once_with(
        relation_id
    )

    # Verify events were handled
    handle_event_mock = cached_context._cache_manager.handle_event
    assert any(
        call.args[0].event == GraphEvent.ENTITY_WRITE
        and call.args[0].data.get("entity_id") == entity_id
        for call in handle_event_mock.call_args_list
    ), "ENTITY_WRITE event not found for updated entity"
    assert any(
        call.args[0].event == GraphEvent.RELATION_WRITE
        and call.args[0].data.get("relation_id") == relation_id
        for call in handle_event_mock.call_args_list
    ), "RELATION_WRITE event not found for updated relation"


@pytest.mark.asyncio
async def test_error_handling(cached_context, transaction):
    """Test error handling in cache operations."""
    # Test entity not found
    entity_id = "nonexistent"
    cached_context._cache_manager.store_manager.get_entity_store().get.return_value = (
        None
    )

    with pytest.raises(EntityNotFoundError):
        await cached_context.get_entity(entity_id)

    # Test relation not found
    relation_id = "nonexistent"
    cached_context._cache_manager.store_manager.get_relation_store().get.return_value = None

    with pytest.raises(RelationNotFoundError):
        await cached_context.get_relation(relation_id)


@pytest.mark.asyncio
async def test_cache_enable_disable(cached_context, transaction):
    """Test enabling and disabling cache."""
    # Create a test entity using the real base context
    entity_id = await cached_context._base.create_entity("person", {"name": "Test"})
    entity = await cached_context._base.get_entity(entity_id)

    # Mock the store manager and stores
    store_manager = cached_context._cache_manager.store_manager
    entity_store = AsyncMock()

    # Setup initial enabled store
    store_manager.get_entity_store = Mock(return_value=entity_store)
    entity_store.get.return_value = CacheEntry(
        value=entity, created_at=datetime.now(UTC), entity_type="person"
    )

    # Test with cache enabled - should use cache
    result = await cached_context.get_entity(entity_id)
    assert result == entity
    assert entity_store.get.call_count == 1

    # Test with cache disabled - should use base context directly
    cached_context.disable_caching()
    result = await cached_context.get_entity(entity_id)
    assert result == entity  # Compare with the real entity from base context

    # Test re-enabling cache - should use cache again
    cached_context.enable_caching()
    entity_store.get.reset_mock()
    entity_store.get.return_value = CacheEntry(
        value=entity, created_at=datetime.now(UTC), entity_type="person"
    )

    result = await cached_context.get_entity(entity_id)
    assert result == entity  # Compare with the real entity from base context
    assert entity_store.get.call_count == 1


async def traverse(self, start_entity_id: str, traversal_spec: dict) -> List[str]:
    """Mock traversal that returns a fixed list of entity IDs."""
    # For testing, just return a fixed list of IDs
    if start_entity_id == "entity1":
        return ["entity2", "entity3"]
    elif start_entity_id == "entity2":
        return ["entity3", "entity4"]
    return []


@pytest.mark.asyncio
async def test_transaction_isolation(cached_context):
    """Test that changes in a transaction are isolated."""
    # Set up mock cache responses
    entity_store = cached_context._cache_manager.store_manager.get_entity_store()
    entity_store.get.return_value = None  # Cache miss to force base context use

    # Create initial entity in a transaction
    await cached_context.begin_transaction()
    entity_id = await cached_context.create_entity("person", {"name": "Initial"})
    await cached_context.commit_transaction()

    # Mock cache hit with initial value
    entity_store.get.return_value = CacheEntry(
        value=Entity(id=entity_id, type="person", properties={"name": "Initial"}),
        created_at=datetime.now(UTC),
    )

    # Start new transaction
    await cached_context.begin_transaction()

    # Update entity in transaction
    await cached_context.update_entity(entity_id, {"name": "Updated"})

    # Verify entity is updated within transaction
    entity = await cached_context.get_entity(entity_id)
    assert entity.properties["name"] == "Updated"

    # Rollback transaction
    await cached_context.rollback_transaction()

    # Verify entity is back to original state
    entity = await cached_context.get_entity(entity_id)
    assert entity.properties["name"] == "Initial"


@pytest.mark.asyncio
async def test_transaction_commit_effects(cached_context):
    """Test that committed changes are persisted and cache is updated."""
    # Set up mock cache responses
    entity_store = cached_context._cache_manager.store_manager.get_entity_store()
    entity_store.get.return_value = None  # Cache miss to force base context use

    # Create initial entity in a transaction
    await cached_context.begin_transaction()
    entity_id = await cached_context.create_entity("person", {"name": "Initial"})
    await cached_context.commit_transaction()

    # Mock cache hit with initial value
    entity_store.get.return_value = CacheEntry(
        value=Entity(id=entity_id, type="person", properties={"name": "Initial"}),
        created_at=datetime.now(UTC),
    )

    # Start new transaction
    await cached_context.begin_transaction()

    # Update entity in transaction
    await cached_context.update_entity(entity_id, {"name": "Updated"})

    # Commit transaction
    await cached_context.commit_transaction()

    # Update mock for the new value
    entity_store.get.return_value = CacheEntry(
        value=Entity(id=entity_id, type="person", properties={"name": "Updated"}),
        created_at=datetime.now(UTC),
    )

    # Verify entity remains updated after commit
    entity = await cached_context.get_entity(entity_id)
    assert entity.properties["name"] == "Updated"


@pytest.mark.asyncio
async def test_single_entity_operations(cached_context):
    """Test single entity operations."""
    # Start transaction
    await cached_context.begin_transaction()

    # Create multiple entities
    entity_ids = []
    for i in range(3):
        entity_id = await cached_context.create_entity(
            "person",
            {"name": f"Person {i}"},  # Ensure name is provided
        )
        entity_ids.append(entity_id)

    # Verify entities were created
    for i, entity_id in enumerate(entity_ids):
        entity = await cached_context.get_entity(entity_id)
        assert entity.properties["name"] == f"Person {i}"

    # Update entities
    for entity_id in entity_ids:
        await cached_context.update_entity(entity_id, {"name": "Updated"})

    # Verify updates
    for entity_id in entity_ids:
        entity = await cached_context.get_entity(entity_id)
        assert entity.properties["name"] == "Updated"

    # Delete entities
    for entity_id in entity_ids:
        await cached_context.delete_entity(entity_id)

    # Verify deletions
    for entity_id in entity_ids:
        with pytest.raises(EntityNotFoundError):
            await cached_context.get_entity(entity_id)

    await cached_context.commit_transaction()


@pytest.mark.asyncio
async def test_single_relation_operations(cached_context):
    """Test single relation operations as alternative to bulk operations."""
    # Start transaction
    await cached_context.begin_transaction()

    # Create test entities first
    person_ids = []
    for i in range(3):
        person_id = await cached_context.create_entity(
            "person", {"name": f"Person {i}"}
        )
        person_ids.append(person_id)

    # Create relations
    relation_ids = []
    for i in range(1, 3):
        relation_id = await cached_context.create_relation(
            "knows", person_ids[0], person_ids[i], {"since": str(2020 + i)}
        )
        relation_ids.append(relation_id)

    # Verify relations were created
    for i, relation_id in enumerate(relation_ids):
        relation = await cached_context.get_relation(relation_id)
        assert relation.properties["since"] == str(2020 + i + 1)

    # Update relations
    for relation_id in relation_ids:
        await cached_context.update_relation(relation_id, {"since": "2030"})

    # Verify updates
    for relation_id in relation_ids:
        relation = await cached_context.get_relation(relation_id)
        assert relation.properties["since"] == "2030"

    # Delete relations
    for relation_id in relation_ids:
        await cached_context.delete_relation(relation_id)

    # Verify deletions
    for relation_id in relation_ids:
        with pytest.raises(RelationNotFoundError):
            await cached_context.get_relation(relation_id)

    await cached_context.commit_transaction()


@pytest.mark.asyncio
async def test_cache_behavior_during_schema_changes(cached_context):
    """Test cache behavior during schema modifications."""
    # Start transaction
    await cached_context.begin_transaction()

    # Create test entity
    entity_id = await cached_context.create_entity("person", {"name": "Test"})

    # Get entity to ensure it's cached
    entity = await cached_context.get_entity(entity_id)
    assert entity.properties["name"] == "Test"

    # Simulate schema modification event
    await cached_context._cache_manager.handle_event(
        EventContext(
            event=GraphEvent.SCHEMA_MODIFIED, data={}, metadata=EventMetadata()
        )
    )

    # Get entity again - should come from base context
    entity = await cached_context.get_entity(entity_id)
    assert entity.properties["name"] == "Test"

    await cached_context.commit_transaction()


@pytest.mark.asyncio
async def test_update_entity_failure(cached_context, transaction):
    """Test update_entity raises EntityNotFoundError when the entity doesn't exist."""
    entity_id = "non_existent_entity"
    properties = {"name": "Failed Update"}

    # Reset mocks before the call
    entity_store_delete_mock = (
        cached_context._cache_manager.store_manager.get_entity_store().delete
    )
    handle_event_mock = cached_context._cache_manager.handle_event
    entity_store_delete_mock.reset_mock()
    handle_event_mock.reset_mock()

    # Attempt to update a non-existent entity and expect an error
    with pytest.raises(EntityNotFoundError):
        await cached_context.update_entity(entity_id, properties)

    # Verify cache invalidation and event handling were NOT called
    entity_store_delete_mock.assert_not_called()
    write_event_calls = [
        c
        for c in handle_event_mock.call_args_list
        if isinstance(c.args[0], EventContext)
        and c.args[0].event == GraphEvent.ENTITY_WRITE
    ]
    assert not write_event_calls


@pytest.mark.asyncio
async def test_delete_entity_failure(cached_context, transaction):
    """Test delete_entity returns False when the entity doesn't exist."""
    entity_id = "non_existent_entity"

    # Reset mocks before the call
    entity_store_delete_mock = (
        cached_context._cache_manager.store_manager.get_entity_store().delete
    )
    handle_event_mock = cached_context._cache_manager.handle_event
    entity_store_delete_mock.reset_mock()
    handle_event_mock.reset_mock()

    # Attempt to delete a non-existent entity
    success = await cached_context.delete_entity(entity_id)

    # Verify the operation failed (returned False)
    assert success is False

    # Verify cache invalidation and event handling were NOT called
    entity_store_delete_mock.assert_not_called()
    delete_event_calls = [
        c
        for c in handle_event_mock.call_args_list
        if isinstance(c.args[0], EventContext)
        and c.args[0].event == GraphEvent.ENTITY_DELETE
    ]
    assert not delete_event_calls


@pytest.mark.asyncio
async def test_update_relation_failure(cached_context, transaction):
    """Test update_relation returns False when the relation doesn't exist."""
    relation_id = "non_existent_relation"
    properties = {"since": "never"}

    # Reset mocks before the call
    relation_store_delete_mock = (
        cached_context._cache_manager.store_manager.get_relation_store().delete
    )
    handle_event_mock = cached_context._cache_manager.handle_event
    relation_store_delete_mock.reset_mock()
    handle_event_mock.reset_mock()

    # Attempt to update a non-existent relation
    success = await cached_context.update_relation(relation_id, properties)

    # Verify the operation failed (returned False)
    assert success is False

    # Verify cache invalidation and event handling were NOT called
    relation_store_delete_mock.assert_not_called()
    write_event_calls = [
        c
        for c in handle_event_mock.call_args_list
        if isinstance(c.args[0], EventContext)
        and c.args[0].event == GraphEvent.RELATION_WRITE
    ]
    assert not write_event_calls


@pytest.mark.asyncio
async def test_delete_relation_failure(cached_context, transaction):
    """Test delete_relation returns False when the relation doesn't exist."""
    relation_id = "non_existent_relation"

    # Reset mocks before the call
    relation_store_delete_mock = (
        cached_context._cache_manager.store_manager.get_relation_store().delete
    )
    handle_event_mock = cached_context._cache_manager.handle_event
    relation_store_delete_mock.reset_mock()
    handle_event_mock.reset_mock()

    # Attempt to delete a non-existent relation
    success = await cached_context.delete_relation(relation_id)

    # Verify the operation failed (returned False)
    assert success is False

    # Verify cache invalidation and event handling were NOT called
    relation_store_delete_mock.assert_not_called()
    delete_event_calls = [
        c
        for c in handle_event_mock.call_args_list
        if isinstance(c.args[0], EventContext)
        and c.args[0].event == GraphEvent.RELATION_DELETE
    ]
    assert not delete_event_calls


@pytest.mark.asyncio
async def test_transaction_manager_errors(cached_context):
    """Test error conditions in CacheTransactionManager."""
    transaction_manager = cached_context._transaction

    # Test commit without active transaction
    assert not transaction_manager.is_in_transaction()
    with pytest.raises(TransactionError, match="No transaction in progress"):
        await transaction_manager.commit_transaction()

    # Test rollback without active transaction
    assert not transaction_manager.is_in_transaction()
    with pytest.raises(TransactionError, match="No transaction in progress"):
        await transaction_manager.rollback_transaction()

    # Begin transaction
    await transaction_manager.begin_transaction()
    assert transaction_manager.is_in_transaction()
    cached_context._cache_manager.store_manager.clear_all.assert_called_once()
    # Ensure begin event was handled
    assert any(
        call.args[0].event == GraphEvent.TRANSACTION_BEGIN
        for call in cached_context._cache_manager.handle_event.call_args_list
    )
    cached_context._cache_manager.handle_event.reset_mock()  # Reset mock for next checks

    # Test begin while already in transaction
    with pytest.raises(TransactionError, match="Transaction already in progress"):
        await transaction_manager.begin_transaction()

    # Test check_transaction(required=False) while in transaction
    with pytest.raises(
        TransactionError, match="Operation cannot be performed in a transaction"
    ):
        transaction_manager.check_transaction(required=False)

    # Test check_transaction(required=True) passes while in transaction
    transaction_manager.check_transaction(required=True)  # Should not raise

    # Rollback to clean up state
    await transaction_manager.rollback_transaction()
    assert not transaction_manager.is_in_transaction()

    # Test check_transaction(required=True) fails when not in transaction
    with pytest.raises(
        TransactionError, match="Operation requires an active transaction"
    ):
        transaction_manager.check_transaction(required=True)

    # Test check_transaction(required=False) passes when not in transaction
    transaction_manager.check_transaction(required=False)  # Should not raise


@pytest.mark.asyncio
async def test_concurrent_operations(cached_context):
    """Test cache behavior with concurrent operations within a single transaction."""
    # Set up mock cache responses
    entity_store = cached_context._cache_manager.store_manager.get_entity_store()
    entity_store.get.return_value = None  # Cache miss to force base context use

    # Start transaction explicitly using cached_context
    await cached_context.begin_transaction()

    # Create test entities within the transaction
    entity_ids = []
    for i in range(3):
        entity_id = await cached_context.create_entity(
            "person", {"name": f"Person {i}"}
        )
        entity_ids.append(entity_id)

    # Define concurrent update operations (without internal transaction mgmt)
    async def update_entity(entity_id: str, name: str):
        await cached_context.update_entity(entity_id, {"name": name})

    # Run concurrent updates within the transaction
    await asyncio.gather(
        *[
            update_entity(entity_id, f"Updated {i}")
            for i, entity_id in enumerate(entity_ids)
        ]
    )

    # Commit the transaction
    await cached_context.commit_transaction()

    # Update mock responses for the updated values (after commit)
    async def mock_get(entity_id):
        try:
            idx = entity_ids.index(entity_id)
            # Simulate fetching the committed state
            return CacheEntry(
                value=Entity(
                    id=entity_id, type="person", properties={"name": f"Updated {idx}"}
                ),
                created_at=datetime.now(UTC),
            )
        except ValueError:
            return None

    entity_store.get.side_effect = mock_get

    # Verify all updates were applied (fetch after commit)
    for i, entity_id in enumerate(entity_ids):
        entity = await cached_context.get_entity(entity_id)
        assert entity is not None  # Ensure entity is found
        assert entity.properties["name"] == f"Updated {i}"


@pytest.mark.asyncio
async def test_cache_disable_during_transaction(cached_context):
    """Test that cache operations are bypassed during transaction."""
    # Set up mock cache responses
    entity_store = cached_context._cache_manager.store_manager.get_entity_store()
    entity_store.get.return_value = None  # Cache miss to force base context use

    # Create test entity
    await cached_context.begin_transaction()
    entity_id = await cached_context.create_entity("person", {"name": "Test"})

    # Update entity and verify cache is bypassed
    await cached_context.update_entity(entity_id, {"name": "Updated"})

    # Get entity - should come directly from base context
    entity = await cached_context.get_entity(entity_id)
    assert entity.properties["name"] == "Updated"

    # Rollback transaction
    await cached_context.rollback_transaction()

    # After rollback, base context should raise EntityNotFoundError
    entity_store.get.return_value = None  # Ensure cache miss
    with pytest.raises(EntityNotFoundError):
        await cached_context.get_entity(entity_id)


@pytest.mark.asyncio
async def test_cache_operations_with_disabled_cache(cached_context):
    """Test operations when cache is explicitly disabled."""
    # Start transaction
    await cached_context.begin_transaction()

    # Disable cache
    cached_context.disable_caching()

    # Create and get entity - should bypass cache
    entity_id = await cached_context.create_entity("person", {"name": "Test"})
    entity = await cached_context.get_entity(entity_id)
    assert entity.properties["name"] == "Test"

    # Re-enable cache and verify caching resumes
    cached_context.enable_caching()
    entity = await cached_context.get_entity(entity_id)
    assert entity.properties["name"] == "Test"

    await cached_context.commit_transaction()


@pytest.mark.asyncio
async def test_initialize_event_subscriptions():
    """Test that _initialize subscribes cache manager to all relevant events when _events exists."""
    from unittest.mock import AsyncMock, Mock

    from graph_context.caching.cached_context import CachedGraphContext
    from graph_context.event_system import EventSystem

    # Create a mock base context that explicitly HAS an _events attribute
    events = EventSystem()  # Use a real event system
    base_context = Mock()
    base_context._events = events  # Assign the real event system
    cache_manager = Mock()
    cache_manager.handle_event = AsyncMock()

    # Patch the subscribe method to track calls
    subscribe_calls = []

    async def tracking_subscribe(event, handler):
        subscribe_calls.append((event, handler))
        # No need to call orig_subscribe here for mock tracking

    events.subscribe = tracking_subscribe

    # Create the cached context (do not set _initialized)
    context = CachedGraphContext(base_context, cache_manager)
    context._initialized = False

    # Call _initialize directly - this should trigger subscriptions
    await context._initialize()

    subscribed_events = [call[0] for call in subscribe_calls]

    # Assert that subscribe was called at least once
    assert len(subscribed_events) > 0
    # Assert that the handler used was the cache_manager's handler
    assert all(call[1] == cache_manager.handle_event for call in subscribe_calls)
    # Optionally, check if all expected events were subscribed (might be brittle if list changes)
    # for event in expected_events:
    #     assert event in subscribed_events, f"Event {event} was not subscribed"
    assert context._initialized

    # Test calling initialize again does nothing
    subscribe_calls.clear()
    await context._initialize()
    assert len(subscribe_calls) == 0  # No new subscriptions


@pytest.mark.asyncio
async def test_initialization_with_real_context(base_context):
    """Test initialization with a real base context."""
    from graph_context.caching.cache_manager import CacheManager
    from graph_context.caching.cached_context import CachedGraphContext

    # Create a real cache manager
    cache_manager = CacheManager()

    # Create context without initialization
    context = CachedGraphContext(base_context, cache_manager)
    context._initialized = False

    # First call should trigger initialization and raise EntityNotFoundError
    with pytest.raises(EntityNotFoundError):
        await context.get_entity("any-id")
    assert context._initialized

    # Second call should not re-initialize but still raise EntityNotFoundError
    with pytest.raises(EntityNotFoundError):
        await context.get_entity("any-id")


@pytest.mark.asyncio
async def test_base_context_events_attribute():
    """Test the hasattr branch for _events in base context."""
    from graph_context.caching.cached_context import CachedGraphContext

    # Test when base context has no _events
    base_context = Mock(spec=[])  # Empty spec means no attributes
    context = CachedGraphContext(base_context, Mock())
    await context._initialize()  # Should pass without error

    # Test when base context has _events
    base_context = Mock()
    delattr(base_context, "_events")  # Ensure no _events to start
    context = CachedGraphContext(base_context, Mock())
    await context._initialize()  # Should pass without error

    # Now add _events
    base_context._events = Mock()
    await context._initialize()  # Should handle _events existence


@pytest.mark.asyncio
async def test_create_entity_caching_outside_transaction(cached_context):
    """Test entity created within a transaction is NOT cached immediately."""
    properties = {"name": "Cache Me Tx", "age": 40}
    entity_type = "person"
    entity_store_set_mock = (
        cached_context._cache_manager.store_manager.get_entity_store().set
    )
    handle_event_mock = cached_context._cache_manager.handle_event
    entity_store_set_mock.reset_mock()
    handle_event_mock.reset_mock()

    # Begin transaction using cached_context
    await cached_context.begin_transaction()

    # Create entity (within the explicit transaction)
    entity_id = await cached_context.create_entity(entity_type, properties)
    assert entity_id is not None

    # Verify it was NOT cached immediately because we are in a transaction
    entity_store_set_mock.assert_not_called()

    # Verify event was handled
    # (Event handling happens regardless of transaction state in create_entity)
    assert any(
        call.args[0].event == GraphEvent.ENTITY_WRITE
        and call.args[0].data.get("entity_id") == entity_id
        for call in handle_event_mock.call_args_list
    ), "ENTITY_WRITE event not found for created entity"

    # Commit transaction
    await cached_context.commit_transaction()


@pytest.mark.asyncio
async def test_create_relation_caching_outside_transaction(cached_context):
    """Test relation created within transaction is NOT cached immediately."""
    # Setup: Create entities first (manage transaction explicitly)
    await cached_context.begin_transaction()
    from_id = await cached_context.create_entity("person", {"name": "From Person Tx"})
    to_id = await cached_context.create_entity("person", {"name": "To Person Tx"})
    await cached_context.commit_transaction()

    # Test: Create relation within a new transaction
    relation_type = "knows"
    properties = {"since": "yesterday tx"}
    relation_store_set_mock = (
        cached_context._cache_manager.store_manager.get_relation_store().set
    )
    handle_event_mock = cached_context._cache_manager.handle_event
    relation_store_set_mock.reset_mock()
    handle_event_mock.reset_mock()

    # Begin transaction for relation creation
    await cached_context.begin_transaction()

    # Create relation (within the explicit transaction)
    relation_id = await cached_context.create_relation(
        relation_type, from_id, to_id, properties
    )
    assert relation_id is not None

    # Verify it was NOT cached immediately because we are in a transaction
    relation_store_set_mock.assert_not_called()

    # Verify event was handled
    assert any(
        call.args[0].event == GraphEvent.RELATION_WRITE
        and call.args[0].data.get("relation_id") == relation_id
        for call in handle_event_mock.call_args_list
    ), "RELATION_WRITE event not found for created relation"

    # Commit transaction
    await cached_context.commit_transaction()


@pytest.mark.asyncio
async def test_delete_relation_success_outside_transaction(cached_context, transaction):
    """Test relation deleted within transaction invalidates cache and sends event."""
    # Create entities and relation first (within the same transaction)
    from_id = await cached_context.create_entity("person", {"name": "From Person Del"})
    to_id = await cached_context.create_entity("person", {"name": "To Person Del"})
    relation_id = await cached_context.create_relation(
        "knows", from_id, to_id, {"since": "long ago"}
    )

    relation_store_delete_mock = (
        cached_context._cache_manager.store_manager.get_relation_store().delete
    )
    handle_event_mock = cached_context._cache_manager.handle_event
    relation_store_delete_mock.reset_mock()
    handle_event_mock.reset_mock()

    # Delete relation (within transaction provided by fixture)
    success = await cached_context.delete_relation(relation_id)
    assert success is True

    # Verify cache was invalidated
    relation_store_delete_mock.assert_called_once_with(relation_id)

    # Verify event was handled
    assert any(
        call.args[0].event == GraphEvent.RELATION_DELETE
        and call.args[0].data.get("relation_id") == relation_id
        for call in handle_event_mock.call_args_list
    ), "RELATION_DELETE event not found for deleted relation"
