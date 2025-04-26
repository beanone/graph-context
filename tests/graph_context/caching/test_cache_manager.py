"""Tests for the CacheManager class."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from graph_context.caching.cache_manager import CacheManager
from graph_context.caching.cache_store import CacheEntry
from graph_context.caching.config import CacheConfig
from graph_context.event_system import (
    EventContext,
    EventMetadata,
    EventSystem,
    GraphEvent,
)


@pytest.fixture
def event_system():
    """Create a mock event system."""
    system = Mock(spec=EventSystem)
    system.subscribe = Mock()  # Not async since it's called in __init__
    return system


@pytest.fixture
def cache_config():
    """Create a cache configuration."""
    return CacheConfig(
        enable_metrics=True,
        entity_cache_size=100,
        entity_cache_ttl=3600,
        relation_cache_size=100,
        relation_cache_ttl=3600,
        query_cache_size=100,
        query_cache_ttl=3600,
        traversal_cache_size=100,
        traversal_cache_ttl=3600,
    )


@pytest.fixture
def mock_store():
    """Create a mock cache store."""
    store = AsyncMock()
    store.get = AsyncMock()
    store.set = AsyncMock()
    store.delete = AsyncMock()
    store.clear = AsyncMock()
    return store


@pytest.fixture
def mock_store_manager():
    """Create a mock store manager with all stores."""
    manager = Mock()

    # Create individual store mocks
    entity_store = AsyncMock()
    entity_store.get = AsyncMock()
    entity_store.set = AsyncMock()
    entity_store.delete = AsyncMock()
    entity_store.clear = AsyncMock()

    relation_store = AsyncMock()
    relation_store.get = AsyncMock()
    relation_store.set = AsyncMock()
    relation_store.delete = AsyncMock()
    relation_store.clear = AsyncMock()

    query_store = AsyncMock()
    query_store.get = AsyncMock()
    query_store.set = AsyncMock()
    query_store.delete = AsyncMock()
    query_store.clear = AsyncMock()

    traversal_store = AsyncMock()
    traversal_store.get = AsyncMock()
    traversal_store.set = AsyncMock()
    traversal_store.delete = AsyncMock()
    traversal_store.clear = AsyncMock()

    # Set up store getters
    manager.get_entity_store = Mock(return_value=entity_store)
    manager.get_relation_store = Mock(return_value=relation_store)
    manager.get_query_store = Mock(return_value=query_store)
    manager.get_traversal_store = Mock(return_value=traversal_store)

    # Set up clear_all as async
    manager.clear_all = AsyncMock()

    return manager


@pytest.fixture
def cache_manager(event_system, cache_config, mock_store_manager):
    """Create a cache manager instance."""
    with patch("graph_context.caching.cache_manager.CacheStoreManager", return_value=mock_store_manager):
        manager = CacheManager(config=cache_config, event_system=event_system)
        return manager


@pytest.mark.asyncio
async def test_entity_caching(cache_manager, mock_store_manager):
    """Test entity caching behavior."""
    # Setup
    entity_id = "test_entity"
    entity_data = {"id": entity_id, "type": "person", "properties": {"name": "Test"}}
    store = mock_store_manager.get_entity_store()
    store.get.return_value = None  # First call returns cache miss

    # Test cache miss
    context = EventContext(
        event=GraphEvent.ENTITY_READ,
        data={"entity_id": entity_id, "result": entity_data},
        metadata=EventMetadata(entity_type="person"),
    )
    await cache_manager.handle_event(context)

    # Verify cache set
    store.set.assert_awaited_once()
    call_args = store.set.await_args[0]
    assert call_args[0] == entity_id
    assert isinstance(call_args[1], CacheEntry)
    assert call_args[1].value == entity_data

    # Setup cache hit
    store.get.return_value = CacheEntry(value=entity_data, created_at=datetime.now(UTC), entity_type="person")

    # Test cache hit
    await cache_manager.handle_event(context)
    assert store.get.await_count == 2  # Called twice now

    # Test cache invalidation
    context = EventContext(
        event=GraphEvent.ENTITY_WRITE, data={"entity_id": entity_id}, metadata=EventMetadata(entity_type="person")
    )
    await cache_manager.handle_event(context)
    store.delete.assert_awaited_once_with(entity_id)


@pytest.mark.asyncio
async def test_relation_caching(cache_manager, mock_store_manager):
    """Test relation caching behavior."""
    # Setup
    relation_id = "test_relation"
    relation_data = {"id": relation_id, "type": "knows", "start_id": "entity1", "end_id": "entity2"}
    store = mock_store_manager.get_relation_store()
    store.get.return_value = None  # First call returns cache miss

    # Test cache miss
    context = EventContext(
        event=GraphEvent.RELATION_READ,
        data={"relation_id": relation_id, "result": relation_data},
        metadata=EventMetadata(relation_type="knows"),
    )
    await cache_manager.handle_event(context)

    # Verify cache set
    store.set.assert_awaited_once()
    call_args = store.set.await_args[0]
    assert call_args[0] == relation_id
    assert isinstance(call_args[1], CacheEntry)
    assert call_args[1].value == relation_data

    # Setup cache hit
    store.get.return_value = CacheEntry(value=relation_data, created_at=datetime.now(UTC), relation_type="knows")

    # Test cache hit
    await cache_manager.handle_event(context)
    assert store.get.await_count == 2  # Called twice now

    # Test cache invalidation
    context = EventContext(
        event=GraphEvent.RELATION_WRITE,
        data={"relation_id": relation_id},
        metadata=EventMetadata(relation_type="knows"),
    )
    await cache_manager.handle_event(context)
    store.delete.assert_awaited_once_with(relation_id)


@pytest.mark.asyncio
async def test_query_caching(cache_manager, mock_store_manager):
    """Test query caching behavior."""
    # Setup
    query_hash = "test_hash"
    query_results = [{"id": "entity1"}, {"id": "entity2"}]
    store = mock_store_manager.get_query_store()
    store.get.return_value = None  # First call returns cache miss

    # Test cache miss
    context = EventContext(event=GraphEvent.QUERY_EXECUTED, data={"query_hash": query_hash, "result": query_results})
    await cache_manager.handle_event(context)

    # Verify cache set
    store.set.assert_awaited_once()
    call_args = store.set.await_args[0]
    assert call_args[0] == query_hash
    assert isinstance(call_args[1], CacheEntry)
    assert call_args[1].value == query_results

    # Setup cache hit
    store.get.return_value = CacheEntry(value=query_results, created_at=datetime.now(UTC), query_hash=query_hash)

    # Test cache hit
    await cache_manager.handle_event(context)
    assert store.get.await_count == 2  # Called twice now


@pytest.mark.asyncio
async def test_traversal_caching(cache_manager, mock_store_manager):
    """Test traversal caching behavior."""
    # Setup
    traversal_hash = "test_hash"
    traversal_results = [{"id": "entity2"}, {"id": "entity3"}]
    store = mock_store_manager.get_traversal_store()
    store.get.return_value = None  # First call returns cache miss

    # Test cache miss
    context = EventContext(
        event=GraphEvent.TRAVERSAL_EXECUTED, data={"traversal_hash": traversal_hash, "result": traversal_results}
    )
    await cache_manager.handle_event(context)

    # Verify cache set
    store.set.assert_awaited_once()
    call_args = store.set.await_args[0]
    assert call_args[0] == traversal_hash
    assert isinstance(call_args[1], CacheEntry)
    assert call_args[1].value == traversal_results

    # Setup cache hit
    store.get.return_value = CacheEntry(
        value=traversal_results, created_at=datetime.now(UTC), query_hash=traversal_hash
    )

    # Test cache hit
    await cache_manager.handle_event(context)
    assert store.get.await_count == 2  # Called twice now


@pytest.mark.asyncio
async def test_schema_modification(cache_manager, mock_store_manager):
    """Test schema modification handling."""
    # Test schema modification
    context = EventContext(event=GraphEvent.SCHEMA_MODIFIED)
    await cache_manager.handle_event(context)

    # Should clear all caches
    mock_store_manager.clear_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_metrics(cache_manager, mock_store_manager):
    """Test cache metrics tracking."""
    # Setup
    entity_id = "test_entity"
    entity_data = {"id": entity_id, "type": "person", "properties": {"name": "Test"}}
    store = mock_store_manager.get_entity_store()

    # Test cache miss
    store.get.return_value = None
    context = EventContext(
        event=GraphEvent.ENTITY_READ,
        data={"entity_id": entity_id, "result": entity_data},
        metadata=EventMetadata(entity_type="person"),
    )
    await cache_manager.handle_event(context)

    # Test cache hit
    store.get.return_value = CacheEntry(value=entity_data, created_at=datetime.now(UTC), entity_type="person")
    await cache_manager.handle_event(context)

    # Verify metrics
    metrics = cache_manager.get_metrics()
    assert metrics is not None
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1
    assert metrics["total_time"] > 0


@pytest.mark.asyncio
async def test_cache_enable_disable(cache_manager):
    """Test enabling and disabling cache."""
    # Test enable (should already be enabled by default)
    assert cache_manager.is_enabled()
    cache_manager.enable()
    assert cache_manager.is_enabled()

    # Test disable
    cache_manager.disable()
    assert not cache_manager.is_enabled()

    # Test re-enable
    cache_manager.enable()
    assert cache_manager.is_enabled()
