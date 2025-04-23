"""Unit tests for the cache store implementation."""

import pytest
import asyncio
from uuid import uuid4

from graph_context.caching.cache_store import CacheStore, CacheEntry
from graph_context.types.type_base import Entity, Relation

@pytest.fixture
def cache_store():
    """Create a cache store instance for testing."""
    return CacheStore(maxsize=100, ttl=1)  # 1 second TTL for testing

@pytest.fixture
def sample_entity():
    """Create a sample entity for testing."""
    return Entity(id="test123", type="person", properties={"name": "Test Person"})

@pytest.fixture
def sample_relation():
    """Create a sample relation for testing."""
    return Relation(
        id="rel123",
        type="knows",
        source_id="person1",
        target_id="person2",
        properties={}
    )

@pytest.mark.asyncio
async def test_cache_set_get(cache_store, sample_entity):
    """Test basic cache set and get operations."""
    key = "test:key"
    entry = CacheEntry(
        value=sample_entity,
        entity_type="person",
        operation_id=str(uuid4())
    )

    # Set the entry
    await cache_store.set(key, entry)

    # Get the entry
    result = await cache_store.get(key)
    assert result is not None
    assert result.value == sample_entity
    assert result.entity_type == "person"

@pytest.mark.asyncio
async def test_cache_ttl(cache_store, sample_entity):
    """Test TTL expiration."""
    key = "test:ttl"
    entry = CacheEntry(
        value=sample_entity,
        entity_type="person",
        operation_id=str(uuid4())
    )

    # Set the entry
    await cache_store.set(key, entry)

    # Wait for TTL to expire (1 second)
    await asyncio.sleep(1.1)

    # Try to get expired entry
    result = await cache_store.get(key)
    assert result is None

@pytest.mark.asyncio
async def test_cache_delete(cache_store, sample_entity):
    """Test cache entry deletion."""
    key = "test:delete"
    entry = CacheEntry(
        value=sample_entity,
        entity_type="person",
        operation_id=str(uuid4())
    )

    # Set and verify
    await cache_store.set(key, entry)
    assert await cache_store.get(key) is not None

    # Delete and verify
    await cache_store.delete(key)
    assert await cache_store.get(key) is None

@pytest.mark.asyncio
async def test_cache_clear(cache_store, sample_entity):
    """Test clearing all cache entries."""
    # Add multiple entries
    entries = {
        "key1": CacheEntry(value=sample_entity, entity_type="person", operation_id=str(uuid4())),
        "key2": CacheEntry(value=sample_entity, entity_type="person", operation_id=str(uuid4())),
        "key3": CacheEntry(value=sample_entity, entity_type="person", operation_id=str(uuid4()))
    }

    for key, entry in entries.items():
        await cache_store.set(key, entry)

    # Clear cache
    await cache_store.clear()

    # Verify all entries are gone
    for key in entries:
        assert await cache_store.get(key) is None

@pytest.mark.asyncio
async def test_type_dependencies(cache_store, sample_entity):
    """Test type-based dependency tracking and invalidation."""
    # Add entries of different types
    person_entries = {
        "person:1": CacheEntry(value=sample_entity, entity_type="person", operation_id=str(uuid4())),
        "person:2": CacheEntry(value=sample_entity, entity_type="person", operation_id=str(uuid4()))
    }
    org_entries = {
        "org:1": CacheEntry(value=sample_entity, entity_type="organization", operation_id=str(uuid4()))
    }

    # Set all entries
    for key, entry in {**person_entries, **org_entries}.items():
        await cache_store.set(key, entry)

    # Verify initial state
    assert len(cache_store._type_dependencies["person"]) == 2
    assert len(cache_store._type_dependencies["organization"]) == 1

    # Invalidate person type
    await cache_store.invalidate_type("person")

    # Verify person entries are gone but org entries remain
    for key in person_entries:
        assert await cache_store.get(key) is None
    for key in org_entries:
        assert await cache_store.get(key) is not None

    # Verify dependencies are cleaned up
    assert len(cache_store._type_dependencies["person"]) == 0
    assert len(cache_store._type_dependencies["organization"]) == 1

@pytest.mark.asyncio
async def test_query_dependencies(cache_store, sample_entity):
    """Test query dependency tracking and invalidation."""
    query_hash = "test_query_hash"

    # Create entries with query dependency
    entry = CacheEntry(
        value=sample_entity,
        entity_type="person",
        operation_id=str(uuid4()),
        query_hash=query_hash
    )

    await cache_store.set("query:result", entry)

    # Verify initial state
    assert len(cache_store._query_dependencies[query_hash]) == 1

    # Invalidate query
    await cache_store.invalidate_query(query_hash)

    # Verify query result is invalidated
    assert await cache_store.get("query:result") is None

    # Verify dependencies are cleaned up
    assert len(cache_store._query_dependencies[query_hash]) == 0

@pytest.mark.asyncio
async def test_reverse_dependencies(cache_store, sample_entity):
    """Test reverse dependency tracking and invalidation."""
    # Create entries with dependencies
    main_key = "main:entry"
    dependent_keys = {"dep:1", "dep:2", "dep:3"}

    main_entry = CacheEntry(
        value=sample_entity,
        entity_type="person",
        operation_id=str(uuid4())
    )

    # Set main entry and dependents
    await cache_store.set(main_key, main_entry)
    for key in dependent_keys:
        entry = CacheEntry(value=sample_entity, entity_type="person", operation_id=str(uuid4()))
        await cache_store.set(key, entry, dependencies={main_key})

    # Verify initial state
    assert len(cache_store._reverse_dependencies[main_key]) == len(dependent_keys)

    # Invalidate dependencies
    await cache_store.invalidate_dependencies(main_key)

    # Verify dependents are invalidated
    for key in dependent_keys:
        assert await cache_store.get(key) is None

    # Verify dependencies are cleaned up
    assert len(cache_store._reverse_dependencies[main_key]) == 0

@pytest.mark.asyncio
async def test_bulk_operations(cache_store, sample_entity):
    """Test bulk delete operations."""
    # Create multiple entries
    keys = {f"bulk:test:{i}" for i in range(10)}
    for key in keys:
        await cache_store.set(
            key,
            CacheEntry(value=sample_entity, entity_type="person", operation_id=str(uuid4()))
        )

    # Delete in bulk
    await cache_store.delete_many(keys)

    # Verify all are deleted
    for key in keys:
        assert await cache_store.get(key) is None

@pytest.mark.asyncio
async def test_scan_operation(cache_store, sample_entity):
    """Test scanning cache entries."""
    # Add multiple entries
    entries = {
        "scan:1": CacheEntry(value=sample_entity, entity_type="person", operation_id=str(uuid4())),
        "scan:2": CacheEntry(value=sample_entity, entity_type="person", operation_id=str(uuid4())),
        "scan:3": CacheEntry(value=sample_entity, entity_type="person", operation_id=str(uuid4()))
    }

    for key, entry in entries.items():
        await cache_store.set(key, entry)

    # Scan and collect results
    scanned = {}
    async for key, entry in cache_store.scan():
        scanned[key] = entry

    # Verify all entries are found
    assert len(scanned) == len(entries)
    for key, entry in entries.items():
        assert key in scanned
        assert scanned[key].value == entry.value