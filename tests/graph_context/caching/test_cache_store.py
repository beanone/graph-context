"""Unit tests for the cache store implementation."""

import asyncio
from uuid import uuid4

import pytest

from graph_context.caching.cache_store import CacheEntry, CacheStore


@pytest.fixture
def cache_store():
    """Create a cache store instance for testing."""
    return CacheStore(maxsize=100, ttl=1)  # 1 second TTL for testing


@pytest.fixture
def non_ttl_cache_store():
    """Create a cache store without TTL for testing."""
    return CacheStore(maxsize=100, ttl=None)


@pytest.fixture
def sample_entity():
    """Create a sample entity for testing."""
    return {"id": "test123", "type": "person", "properties": {"name": "Test Person"}}


@pytest.mark.asyncio
async def test_cache_set_get(cache_store, sample_entity):
    """Test basic cache set and get operations."""
    key = "test:key"
    entry = CacheEntry(
        value=sample_entity, entity_type="person", operation_id=str(uuid4())
    )

    # Set the entry
    await cache_store.set(key, entry)

    # Get the entry
    result = await cache_store.get(key)
    assert result is not None
    assert result.value == sample_entity
    assert result.entity_type == "person"


@pytest.mark.asyncio
async def test_non_existent_key(cache_store):
    """Test getting a non-existent key."""
    result = await cache_store.get("non_existent_key")
    assert result is None


@pytest.mark.asyncio
async def test_cache_non_ttl(non_ttl_cache_store, sample_entity):
    """Test cache without TTL."""
    key = "test:no_ttl"
    entry = CacheEntry(
        value=sample_entity, entity_type="person", operation_id=str(uuid4())
    )

    # Set the entry
    await non_ttl_cache_store.set(key, entry)

    # Wait for what would normally be TTL
    await asyncio.sleep(1.1)

    # Entry should still be there
    result = await non_ttl_cache_store.get(key)
    assert result is not None
    assert result.value == sample_entity


@pytest.mark.asyncio
async def test_cache_ttl(cache_store, sample_entity):
    """Test TTL expiration."""
    key = "test:ttl"
    entry = CacheEntry(
        value=sample_entity, entity_type="person", operation_id=str(uuid4())
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
        value=sample_entity, entity_type="person", operation_id=str(uuid4())
    )

    # Set and verify
    await cache_store.set(key, entry)
    assert await cache_store.get(key) is not None

    # Delete and verify
    await cache_store.delete(key)
    assert await cache_store.get(key) is None


@pytest.mark.asyncio
async def test_delete_missing_key(cache_store):
    """Test deleting a non-existent key doesn't raise an error."""
    # Should not raise an exception
    await cache_store.delete("non_existent_key")


@pytest.mark.asyncio
async def test_cache_clear(cache_store, sample_entity):
    """Test clearing all cache entries."""
    # Add multiple entries
    entries = {
        "key1": CacheEntry(
            value=sample_entity, entity_type="person", operation_id=str(uuid4())
        ),
        "key2": CacheEntry(
            value=sample_entity, entity_type="person", operation_id=str(uuid4())
        ),
        "key3": CacheEntry(
            value=sample_entity, entity_type="person", operation_id=str(uuid4())
        ),
    }

    for key, entry in entries.items():
        await cache_store.set(key, entry)

    # Clear cache
    await cache_store.clear()

    # Verify all entries are gone
    for key in entries:
        assert await cache_store.get(key) is None


@pytest.mark.asyncio
async def test_scan_operation(cache_store, sample_entity):
    """Test scanning cache entries."""
    # Add multiple entries
    entries = {
        "scan:1": CacheEntry(
            value=sample_entity, entity_type="person", operation_id=str(uuid4())
        ),
        "scan:2": CacheEntry(
            value=sample_entity, entity_type="person", operation_id=str(uuid4())
        ),
        "scan:3": CacheEntry(
            value=sample_entity, entity_type="person", operation_id=str(uuid4())
        ),
    }

    for key, entry in entries.items():
        await cache_store.set(key, entry)

    # Scan and collect results
    scanned = {}
    async for key, entry in cache_store.scan():
        scanned[key] = entry

    # Verify all entries are found
    assert len(scanned) >= len(entries)
    for key, entry in entries.items():
        assert key in scanned
        assert scanned[key].value == entry.value


@pytest.mark.asyncio
async def test_scan_empty_cache(non_ttl_cache_store):
    """Test scanning an empty cache."""
    # Should not raise an exception and yield nothing
    count = 0
    async for _, _ in non_ttl_cache_store.scan():
        count += 1
    assert count == 0
