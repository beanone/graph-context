"""Integration tests for the cached graph context implementation."""

from typing import Dict, Any, Optional, List, AsyncGenerator
import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, UTC

from graph_context.caching.cached_context import CachedGraphContext
from graph_context.caching.cache_manager import CacheManager
from graph_context.caching.cache_store import CacheEntry
from graph_context.event_system import EventSystem, GraphEvent, EventContext, EventMetadata
from graph_context.context_base import BaseGraphContext
from graph_context.types.type_base import Entity, Relation, EntityType, PropertyDefinition, RelationType
from graph_context.exceptions import SchemaError, EntityNotFoundError, RelationNotFoundError

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