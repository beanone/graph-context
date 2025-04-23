"""Unit tests for the cache manager implementation."""

import pytest
from typing import Dict, Any
from datetime import datetime, UTC

from graph_context.caching.cache_manager import CacheManager
from graph_context.caching.config import CacheConfig
from graph_context.event_system import (
    GraphEvent,
    EventContext,
    EventMetadata
)


@pytest.fixture
def cache_config():
    """Create a cache configuration for testing."""
    return CacheConfig(
        max_size=100,
        default_ttl=300,
        enable_metrics=True,
        type_ttls={
            "test_type": 600,
            "query": 300,
            "traversal": 300,
        }
    )


@pytest.fixture
def cache_manager(cache_config):
    """Create a cache manager instance for testing."""
    return CacheManager(config=cache_config)


@pytest.fixture
def sample_entity() -> Dict[str, Any]:
    """Create a sample entity for testing."""
    return {
        "id": "test123",
        "type": "test_type",
        "properties": {"name": "Test Entity"}
    }


@pytest.fixture
def sample_relation() -> Dict[str, Any]:
    """Create a sample relation for testing."""
    return {
        "id": "rel123",
        "type": "test_relation",
        "from_entity": "entity1",
        "to_entity": "entity2",
        "properties": {"weight": 1.0}
    }


@pytest.mark.asyncio
async def test_entity_read_event(cache_manager, sample_entity):
    """Test handling of entity read events."""
    # Handle read event without cache hit
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.ENTITY_READ,
        metadata=EventMetadata(entity_type=sample_entity["type"]),
        data={
            "entity_id": sample_entity["id"],
            "result": sample_entity
        }
    ))

    # Verify metrics
    metrics = cache_manager.get_metrics()
    assert metrics is not None
    assert metrics["hits"] == 0
    assert metrics["misses"] == 1

    # Handle read event with cache hit
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.ENTITY_READ,
        metadata=EventMetadata(entity_type=sample_entity["type"]),
        data={"entity_id": sample_entity["id"]}
    ))

    # Verify metrics updated
    metrics = cache_manager.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1


@pytest.mark.asyncio
async def test_entity_write_event(cache_manager, sample_entity):
    """Test handling of entity write events."""
    # First cache an entity
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.ENTITY_READ,
        metadata=EventMetadata(entity_type=sample_entity["type"]),
        data={
            "entity_id": sample_entity["id"],
            "result": sample_entity
        }
    ))

    # Now write the same entity
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.ENTITY_WRITE,
        metadata=EventMetadata(entity_type=sample_entity["type"]),
        data={"entity_id": sample_entity["id"]}
    ))

    # Verify cache was invalidated
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.ENTITY_READ,
        metadata=EventMetadata(entity_type=sample_entity["type"]),
        data={"entity_id": sample_entity["id"]}
    ))

    # Should be a cache miss
    metrics = cache_manager.get_metrics()
    assert metrics["misses"] == 2
    assert metrics["hits"] == 0


@pytest.mark.asyncio
async def test_relation_operations(cache_manager, sample_relation):
    """Test handling of relation operations."""
    # Cache a relation
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.RELATION_READ,
        metadata=EventMetadata(relation_type=sample_relation["type"]),
        data={
            "relation_id": sample_relation["id"],
            "result": sample_relation
        }
    ))

    # Verify metrics
    metrics = cache_manager.get_metrics()
    assert metrics["misses"] == 1
    assert metrics["hits"] == 0

    # Read cached relation
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.RELATION_READ,
        metadata=EventMetadata(relation_type=sample_relation["type"]),
        data={"relation_id": sample_relation["id"]}
    ))

    # Verify cache hit
    metrics = cache_manager.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1

    # Delete relation
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.RELATION_DELETE,
        metadata=EventMetadata(relation_type=sample_relation["type"]),
        data={"relation_id": sample_relation["id"]}
    ))

    # Verify cache miss after deletion
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.RELATION_READ,
        metadata=EventMetadata(relation_type=sample_relation["type"]),
        data={"relation_id": sample_relation["id"]}
    ))

    metrics = cache_manager.get_metrics()
    assert metrics["misses"] == 2
    assert metrics["hits"] == 1


@pytest.mark.asyncio
async def test_query_caching(cache_manager):
    """Test handling of query execution events."""
    query_hash = "test_query_hash"
    query_result = [{"id": "1", "type": "test_type"}]
    dependencies = {"test_type"}

    # Cache query results
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.QUERY_EXECUTED,
        metadata=EventMetadata(affected_types=dependencies),
        data={
            "query_hash": query_hash,
            "result": query_result
        }
    ))

    # Read cached query results
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.QUERY_EXECUTED,
        metadata=EventMetadata(affected_types=dependencies),
        data={"query_hash": query_hash}
    ))

    # Verify metrics
    metrics = cache_manager.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1

    # Invalidate by modifying a dependent type
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.SCHEMA_MODIFIED,
        metadata=EventMetadata(affected_types={"test_type"}),
        data={}
    ))

    # Verify query cache was invalidated
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.QUERY_EXECUTED,
        metadata=EventMetadata(affected_types=dependencies),
        data={"query_hash": query_hash}
    ))

    metrics = cache_manager.get_metrics()
    assert metrics["misses"] == 3  # Initial write + schema modification + read after invalidation
    assert metrics["hits"] == 1    # First successful read


@pytest.mark.asyncio
async def test_traversal_caching(cache_manager):
    """Test handling of traversal execution events."""
    traversal_hash = "test_traversal_hash"
    traversal_result = [{"path": [{"id": "1", "type": "test_type"}]}]
    dependencies = {"test_type"}

    # Cache traversal results
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.TRAVERSAL_EXECUTED,
        metadata=EventMetadata(affected_types=dependencies),
        data={
            "traversal_hash": traversal_hash,
            "result": traversal_result
        }
    ))

    # Read cached traversal results
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.TRAVERSAL_EXECUTED,
        metadata=EventMetadata(affected_types=dependencies),
        data={"traversal_hash": traversal_hash}
    ))

    # Verify metrics
    metrics = cache_manager.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 1

    # Invalidate by modifying a dependent type
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.SCHEMA_MODIFIED,
        metadata=EventMetadata(affected_types={"test_type"}),
        data={}
    ))

    # Verify traversal cache was invalidated
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.TRAVERSAL_EXECUTED,
        metadata=EventMetadata(affected_types=dependencies),
        data={"traversal_hash": traversal_hash}
    ))

    metrics = cache_manager.get_metrics()
    assert metrics["misses"] == 3  # Initial write + schema modification + read after invalidation
    assert metrics["hits"] == 1    # First successful read


@pytest.mark.asyncio
async def test_cache_enable_disable(cache_manager, sample_entity):
    """Test enabling and disabling the cache."""
    # Cache should be enabled by default
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.ENTITY_READ,
        metadata=EventMetadata(entity_type=sample_entity["type"]),
        data={
            "entity_id": sample_entity["id"],
            "result": sample_entity
        }
    ))

    metrics = cache_manager.get_metrics()
    assert metrics["misses"] == 1

    # Disable cache
    cache_manager.disable()

    # Cache should not be used when disabled
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.ENTITY_READ,
        metadata=EventMetadata(entity_type=sample_entity["type"]),
        data={"entity_id": sample_entity["id"]}
    ))

    metrics = cache_manager.get_metrics()
    assert metrics["misses"] == 2
    assert metrics["hits"] == 0

    # Re-enable cache
    cache_manager.enable()

    # Cache should be used again
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.ENTITY_READ,
        metadata=EventMetadata(entity_type=sample_entity["type"]),
        data={"entity_id": sample_entity["id"]}
    ))

    metrics = cache_manager.get_metrics()
    assert metrics["hits"] == 1
    assert metrics["misses"] == 2


@pytest.mark.asyncio
async def test_cache_clear(cache_manager, sample_entity):
    """Test clearing the cache."""
    # Cache an entity
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.ENTITY_READ,
        metadata=EventMetadata(entity_type=sample_entity["type"]),
        data={
            "entity_id": sample_entity["id"],
            "result": sample_entity
        }
    ))

    # Clear the cache
    await cache_manager.clear()

    # Verify cache was cleared
    await cache_manager.handle_event(EventContext(
        event=GraphEvent.ENTITY_READ,
        metadata=EventMetadata(entity_type=sample_entity["type"]),
        data={"entity_id": sample_entity["id"]}
    ))

    metrics = cache_manager.get_metrics()
    assert metrics["misses"] == 2
    assert metrics["hits"] == 0