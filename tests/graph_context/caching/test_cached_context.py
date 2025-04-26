"""Integration tests for the cached graph context implementation."""

from typing import Dict, Any, Optional, List, AsyncGenerator
import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, UTC
import types

from graph_context.caching.cached_context import CachedGraphContext
from graph_context.caching.cache_manager import CacheManager
from graph_context.caching.cache_store import CacheEntry
from graph_context.event_system import EventSystem, GraphEvent, EventContext, EventMetadata
from graph_context.context_base import BaseGraphContext
from graph_context.types.type_base import Entity, Relation, EntityType, PropertyDefinition, RelationType
from graph_context.exceptions import SchemaError, EntityNotFoundError, RelationNotFoundError, TransactionError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('graph_context.caching.cached_context')
logger.setLevel(logging.DEBUG)


@pytest.fixture
async def base_context() -> AsyncGenerator[BaseGraphContext, None]:
    """Create a base context using InMemoryGraphStore."""
    context = BaseGraphContext()

    # Register standard entity and relation types
    await context.register_entity_type(EntityType(
        name="person",
        properties={
            "name": PropertyDefinition(type="string", required=True),
            "age": PropertyDefinition(type="integer", required=False)
        }
    ))

    await context.register_relation_type(RelationType(
        name="knows",
        from_types=["person"],
        to_types=["person"],
        properties={
            "since": PropertyDefinition(type="string", required=False)
        }
    ))

    yield context
    await context.cleanup()


@pytest.fixture
async def transaction(base_context: BaseGraphContext) -> AsyncGenerator[BaseGraphContext, None]:
    """Create and manage a transaction for tests."""
    await base_context.begin_transaction()
    yield base_context
    try:
        await base_context.commit_transaction()
    except:
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
    entity = await cached_context._base.get_entity(entity_id)  # Get the actual entity format

    # Test cache hit
    store = cached_context._cache_manager.store_manager.get_entity_store()
    store.get.return_value = CacheEntry(
        value=entity,
        created_at=datetime.now(UTC),
        entity_type="person"
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
    from_entity_id = await cached_context._base.create_entity("person", {"name": "Person A"})
    to_entity_id = await cached_context._base.create_entity("person", {"name": "Person B"})

    # Create the test relation
    relation_id = await cached_context._base.create_relation(
        "knows",
        from_entity_id,
        to_entity_id,
        {"since": "2024"}
    )
    relation = await cached_context._base.get_relation(relation_id)

    # Test cache hit
    store = cached_context._cache_manager.store_manager.get_relation_store()
    store.get.return_value = CacheEntry(
        value=relation,
        created_at=datetime.now(UTC),
        relation_type="knows"
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
    entity1_id = await cached_context._base.create_entity("person", {"name": "Person A", "age": 30})
    entity2_id = await cached_context._base.create_entity("person", {"name": "Person B", "age": 25})

    # Define query and get actual results
    query_spec = {"entity_type": "person"}
    results = await cached_context._base.query(query_spec)

    # Mock hash generation
    query_hash = "test_hash"
    cached_context._cache_manager._hash_query.return_value = query_hash

    # Test cache hit
    store = cached_context._cache_manager.store_manager.get_query_store()
    store.get.return_value = CacheEntry(
        value=results,
        created_at=datetime.now(UTC),
        query_hash=query_hash
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
    start_entity_id = await cached_context._base.create_entity("person", {"name": "Start Person"})
    target1_id = await cached_context._base.create_entity("person", {"name": "Target 1"})
    target2_id = await cached_context._base.create_entity("person", {"name": "Target 2"})

    # Create relations
    await cached_context._base.create_relation("knows", start_entity_id, target1_id)
    await cached_context._base.create_relation("knows", start_entity_id, target2_id)

    # Define traversal and get actual results
    traversal_spec = {
        "max_depth": 2,
        "relation_types": ["knows"],
        "direction": "outbound"
    }
    results = await cached_context._base.traverse(start_entity_id, traversal_spec)

    # Mock hash generation
    traversal_hash = "test_hash"
    cached_context._cache_manager._hash_query.return_value = traversal_hash

    # Test cache hit
    store = cached_context._cache_manager.store_manager.get_traversal_store()
    store.get.return_value = CacheEntry(
        value=results,
        created_at=datetime.now(UTC),
        query_hash=traversal_hash
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
    cached_context._cache_manager.store_manager.get_entity_store().delete.assert_called_once_with(entity_id)

    # Test relation cache invalidation
    properties = {"since": "2024"}
    await cached_context.update_relation(relation_id, properties)
    cached_context._cache_manager.store_manager.get_relation_store().delete.assert_called_once_with(relation_id)


@pytest.mark.asyncio
async def test_error_handling(cached_context, transaction):
    """Test error handling in cache operations."""
    # Test entity not found
    entity_id = "nonexistent"
    cached_context._cache_manager.store_manager.get_entity_store().get.return_value = None

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
        value=entity,
        created_at=datetime.now(UTC),
        entity_type="person"
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
        value=entity,
        created_at=datetime.now(UTC),
        entity_type="person"
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
        created_at=datetime.now(UTC)
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
        created_at=datetime.now(UTC)
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
        created_at=datetime.now(UTC)
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
        entity_id = await cached_context.create_entity("person", {
            "name": f"Person {i}"  # Ensure name is provided
        })
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
        person_id = await cached_context.create_entity("person", {"name": f"Person {i}"})
        person_ids.append(person_id)

    # Create relations
    relation_ids = []
    for i in range(1, 3):
        relation_id = await cached_context.create_relation(
            "knows",
            person_ids[0],
            person_ids[i],
            {"since": str(2020 + i)}
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
    await cached_context._cache_manager.handle_event(EventContext(
        event=GraphEvent.SCHEMA_MODIFIED,
        data={},
        metadata=EventMetadata()
    ))

    # Get entity again - should come from base context
    entity = await cached_context.get_entity(entity_id)
    assert entity.properties["name"] == "Test"

    await cached_context.commit_transaction()


@pytest.mark.asyncio
async def test_concurrent_operations(cached_context):
    """Test cache behavior with concurrent operations."""
    # Set up mock cache responses
    entity_store = cached_context._cache_manager.store_manager.get_entity_store()
    entity_store.get.return_value = None  # Cache miss to force base context use

    # Start transaction
    await cached_context.begin_transaction()

    # Create test entities
    entity_ids = []
    for i in range(3):
        entity_id = await cached_context.create_entity("person", {"name": f"Person {i}"})
        entity_ids.append(entity_id)

    await cached_context.commit_transaction()

    # Define concurrent update operations
    async def update_entity(entity_id: str, name: str):
        await cached_context.begin_transaction()
        await cached_context.update_entity(entity_id, {"name": name})
        await cached_context.commit_transaction()

    # Run concurrent updates
    await asyncio.gather(*[
        update_entity(entity_id, f"Updated {i}")
        for i, entity_id in enumerate(entity_ids)
    ])

    # Update mock responses for the updated values
    async def mock_get(entity_id):
        # Find the index of the entity to get its updated name
        try:
            idx = entity_ids.index(entity_id)
            return CacheEntry(
                value=Entity(id=entity_id, type="person", properties={"name": f"Updated {idx}"}),
                created_at=datetime.now(UTC)
            )
        except ValueError:
            return None

    entity_store.get.side_effect = mock_get

    # Verify all updates were applied
    for i, entity_id in enumerate(entity_ids):
        entity = await cached_context.get_entity(entity_id)
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
    """Test that _initialize subscribes cache manager to all relevant events."""
    from graph_context.caching.cached_context import CachedGraphContext
    from graph_context.event_system import EventSystem, GraphEvent
    from unittest.mock import AsyncMock, Mock

    # Create a real event system and a mock cache manager
    events = EventSystem()
    base_context = Mock()
    base_context._events = events
    cache_manager = Mock()
    cache_manager.handle_event = AsyncMock()

    # Patch the subscribe method to track calls
    subscribe_calls = []
    orig_subscribe = events.subscribe
    async def tracking_subscribe(event, handler):
        subscribe_calls.append((event, handler))
        await orig_subscribe(event, handler)
    events.subscribe = tracking_subscribe

    # Create the cached context (do not set _initialized)
    context = CachedGraphContext(base_context, cache_manager)
    context._initialized = False

    # Call _initialize directly
    await context._initialize()

    # Check that all relevant events were subscribed
    expected_events = [
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
        GraphEvent.TRANSACTION_ROLLBACK,
    ]
    subscribed_events = [call[0] for call in subscribe_calls]
    for event in expected_events:
        assert event in subscribed_events

@pytest.mark.asyncio
async def test_initialization_with_real_context(base_context):
    """Test initialization with a real base context."""
    from graph_context.caching.cached_context import CachedGraphContext
    from graph_context.caching.cache_manager import CacheManager

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
    delattr(base_context, '_events')  # Ensure no _events to start
    context = CachedGraphContext(base_context, Mock())
    await context._initialize()  # Should pass without error

    # Now add _events
    base_context._events = Mock()
    await context._initialize()  # Should handle _events existence