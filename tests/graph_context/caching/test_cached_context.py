"""Integration tests for the cached graph context implementation."""

from typing import Dict, Any, Optional, List
import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, Mock, patch

from graph_context.caching.cached_context import CachedGraphContext
from graph_context.caching.cache_manager import CacheManager
from graph_context.event_system import EventSystem, GraphEvent, EventContext, EventMetadata
from graph_context.interface import GraphContext
from graph_context.types.type_base import Entity, Relation, EntityType, PropertyDefinition, RelationType
from graph_context.exceptions import SchemaError, EntityNotFoundError, RelationNotFoundError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('graph_context.caching.cached_context')
logger.setLevel(logging.DEBUG)

class MockGraphContext(GraphContext):
    """Mock implementation of GraphContext for testing."""

    def __init__(self):
        self.entities = {}
        self.relations = {}
        self.next_id = 1
        self.in_transaction = False
        self._events = EventSystem()

        # Register standard entity and relation types
        self.entity_types = {
            "person": EntityType(
                name="person",
                properties={
                    "name": PropertyDefinition(type="string", required=True),
                    "age": PropertyDefinition(type="integer", required=False)
                }
            )
        }

        self.relation_types = {
            "knows": RelationType(
                name="knows",
                from_types=["person"],
                to_types=["person"],
                properties={
                    "since": PropertyDefinition(type="string", required=False)
                }
            )
        }

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        id_str = str(self.next_id)
        self.next_id += 1
        return id_str

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.entities = {}
        self.relations = {}
        self.in_transaction = False

    async def register_entity_type(self, entity_type: EntityType) -> None:
        """Register an entity type."""
        self.entity_types[entity_type.name] = entity_type
        await self._events.emit(GraphEvent.SCHEMA_MODIFIED)

    async def register_relation_type(self, relation_type: RelationType) -> None:
        """Register a relation type."""
        self.relation_types[relation_type.name] = relation_type
        await self._events.emit(GraphEvent.SCHEMA_MODIFIED)

    async def begin_transaction(self) -> None:
        """Begin a transaction."""
        self.in_transaction = True
        await self._events.emit(GraphEvent.TRANSACTION_BEGIN)

    async def commit_transaction(self) -> None:
        """Commit a transaction."""
        self.in_transaction = False
        await self._events.emit(GraphEvent.TRANSACTION_COMMIT)

    async def rollback_transaction(self) -> None:
        """Rollback a transaction."""
        self.in_transaction = False
        await self._events.emit(GraphEvent.TRANSACTION_ROLLBACK)

    async def get_entity(self, entity_id: str) -> Entity:
        """Get an entity by ID."""
        if entity_id not in self.entities:
            raise EntityNotFoundError(f"Entity {entity_id} not found")

        entity = self.entities[entity_id]
        await self._events.emit(
            GraphEvent.ENTITY_READ,
            entity_id=entity_id,
            entity_type=entity["type"]
        )
        return {"id": entity_id, **entity}

    async def create_entity(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """Create a new entity."""
        if not self.in_transaction:
            raise Exception("Operation requires transaction")

        entity_id = self._generate_id()
        entity = {"type": entity_type, "properties": properties}
        self.entities[entity_id] = entity

        await self._events.emit(
            GraphEvent.ENTITY_WRITE,
            entity_id=entity_id,
            entity_type=entity_type
        )
        return entity_id

    async def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Update an entity."""
        if not self.in_transaction:
            raise Exception("Operation requires transaction")

        if entity_id not in self.entities:
            return False

        self.entities[entity_id]["properties"] = properties

        await self._events.emit(
            GraphEvent.ENTITY_WRITE,
            entity_id=entity_id,
            entity_type=self.entities[entity_id]["type"]
        )
        return True

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        if not self.in_transaction:
            raise Exception("Operation requires transaction")

        if entity_id not in self.entities:
            return False

        entity_type = self.entities[entity_id]["type"]
        del self.entities[entity_id]

        await self._events.emit(
            GraphEvent.ENTITY_DELETE,
            entity_id=entity_id,
            entity_type=entity_type
        )
        return True

    async def get_relation(self, relation_id: str) -> Relation:
        """Get a relation by ID."""
        if relation_id not in self.relations:
            raise RelationNotFoundError(f"Relation {relation_id} not found")

        relation = self.relations[relation_id]
        await self._events.emit(
            GraphEvent.RELATION_READ,
            relation_id=relation_id,
            relation_type=relation["type"]
        )
        return {"id": relation_id, **relation}

    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new relation."""
        if not self.in_transaction:
            raise Exception("Operation requires transaction")

        # Verify entities exist
        if from_entity not in self.entities:
            raise EntityNotFoundError(f"From entity {from_entity} not found")

        if to_entity not in self.entities:
            raise EntityNotFoundError(f"To entity {to_entity} not found")

        relation_id = self._generate_id()
        relation = {
            "type": relation_type,
            "from_entity": from_entity,
            "to_entity": to_entity,
            "properties": properties or {}
        }
        self.relations[relation_id] = relation

        await self._events.emit(
            GraphEvent.RELATION_WRITE,
            relation_id=relation_id,
            relation_type=relation_type
        )
        return relation_id

    async def update_relation(self, relation_id: str, properties: Dict[str, Any]) -> bool:
        """Update a relation."""
        if not self.in_transaction:
            raise Exception("Operation requires transaction")

        if relation_id not in self.relations:
            return False

        self.relations[relation_id]["properties"] = properties

        await self._events.emit(
            GraphEvent.RELATION_WRITE,
            relation_id=relation_id,
            relation_type=self.relations[relation_id]["type"]
        )
        return True

    async def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation."""
        if not self.in_transaction:
            raise Exception("Operation requires transaction")

        if relation_id not in self.relations:
            return False

        relation_type = self.relations[relation_id]["type"]
        del self.relations[relation_id]

        await self._events.emit(
            GraphEvent.RELATION_DELETE,
            relation_id=relation_id,
            relation_type=relation_type
        )
        return True

    async def query(self, query_spec: Dict[str, Any]) -> List[Entity]:
        """Execute a query."""
        results = []

        # Simple implementation for testing
        if "type" in query_spec:
            for entity_id, entity in self.entities.items():
                if entity["type"] == query_spec["type"]:
                    results.append({"id": entity_id, **entity})

        await self._events.emit(
            GraphEvent.QUERY_EXECUTED,
            query_spec=query_spec
        )
        return results

    async def traverse(self, start_entity: str, traversal_spec: Dict[str, Any]) -> List[Entity]:
        """Execute a traversal."""
        results = []

        # Verify start entity exists
        if start_entity not in self.entities:
            raise EntityNotFoundError(f"Start entity {start_entity} not found")

        # Simple implementation for testing
        if "relation_type" in traversal_spec:
            target_ids = set()
            for relation_id, relation in self.relations.items():
                if (relation["type"] == traversal_spec["relation_type"] and
                    relation["from_entity"] == start_entity):
                    target_ids.add(relation["to_entity"])

            for entity_id in target_ids:
                entity = self.entities.get(entity_id)
                if entity:
                    results.append({"id": entity_id, **entity})

        await self._events.emit(
            GraphEvent.TRAVERSAL_EXECUTED,
            start_entity=start_entity,
            traversal_spec=traversal_spec
        )
        return results


@pytest.fixture
async def base_context():
    """Create a base context for testing."""
    return MockGraphContext()


@pytest.fixture
async def cached_context(base_context):
    """Create a cached context for testing."""
    cache_manager = CacheManager()
    context = CachedGraphContext(base_context, cache_manager)
    yield context
    await context.cleanup()


@pytest.mark.asyncio
async def test_entity_caching(cached_context, base_context):
    """Test that entities are properly cached and retrieved."""
    # Start transaction for writing
    await cached_context.begin_transaction()

    # Create an entity
    entity_id = await cached_context.create_entity("person", {"name": "Alice", "age": 30})

    # Commit to make it persistent
    await cached_context.commit_transaction()

    # Get the entity - first time should retrieve from base context
    entity1 = await cached_context.get_entity(entity_id)

    # Access the cache directly to verify it's there
    cache_store = cached_context._cache_manager.store_manager.get_entity_store()
    cached_entry = await cache_store.get(entity_id)
    assert cached_entry is not None
    assert cached_entry.value["properties"]["name"] == "Alice"

    # Patch base context's get_entity to fail if called
    original_get_entity = base_context.get_entity
    base_context.get_entity = AsyncMock(side_effect=Exception("Should not be called"))

    try:
        # Get the entity again - should use cache and not call base context
        entity2 = await cached_context.get_entity(entity_id)
        assert entity1 == entity2
        assert entity1["properties"]["name"] == "Alice"
        assert entity1["properties"]["age"] == 30
    finally:
        # Restore original method
        base_context.get_entity = original_get_entity


@pytest.mark.asyncio
async def test_relation_caching(cached_context):
    """Test that relations are properly cached and retrieved."""
    # Start transaction for writing
    await cached_context.begin_transaction()

    # Create two entities
    entity1_id = await cached_context.create_entity("person", {"name": "Alice"})
    entity2_id = await cached_context.create_entity("person", {"name": "Bob"})

    # Create a relation between them
    relation_id = await cached_context.create_relation(
        "knows",
        entity1_id,
        entity2_id,
        {"since": "2020"}
    )

    # Commit to make it persistent
    await cached_context.commit_transaction()

    # Get the relation - first time should retrieve from base context
    relation1 = await cached_context.get_relation(relation_id)

    # Access the cache directly to verify it's there
    cache_store = cached_context._cache_manager.store_manager.get_relation_store()
    cached_entry = await cache_store.get(relation_id)
    assert cached_entry is not None
    assert cached_entry.value["properties"]["since"] == "2020"

    # Get the relation again - should use cache
    relation2 = await cached_context.get_relation(relation_id)
    assert relation1 == relation2
    assert relation1["from_entity"] == entity1_id
    assert relation1["to_entity"] == entity2_id
    assert relation1["properties"]["since"] == "2020"


@pytest.mark.asyncio
async def test_cache_invalidation(cached_context):
    """Test that cache is properly invalidated on updates and deletes."""
    # Start transaction for writing
    await cached_context.begin_transaction()

    # Create an entity
    entity_id = await cached_context.create_entity("person", {"name": "Alice"})

    # Commit to make it persistent
    await cached_context.commit_transaction()

    # Get it once to ensure it's cached
    entity1 = await cached_context.get_entity(entity_id)

    # Verify it's in the cache
    cache_store = cached_context._cache_manager.store_manager.get_entity_store()
    assert await cache_store.get(entity_id) is not None

    # Start transaction for updating
    await cached_context.begin_transaction()

    # Update it
    await cached_context.update_entity(entity_id, {"name": "Alice Smith"})

    # Commit the update
    await cached_context.commit_transaction()

    # Verify it's invalidated in the cache
    assert await cache_store.get(entity_id) is None

    # Get it again - should fetch fresh data
    entity2 = await cached_context.get_entity(entity_id)
    assert entity2["properties"]["name"] == "Alice Smith"

    # Verify it's back in the cache
    assert await cache_store.get(entity_id) is not None

    # Start transaction for deletion
    await cached_context.begin_transaction()

    # Delete it
    await cached_context.delete_entity(entity_id)

    # Commit the deletion
    await cached_context.commit_transaction()

    # Verify it's invalidated in the cache
    assert await cache_store.get(entity_id) is None

    # Try to get it - should raise EntityNotFoundError
    with pytest.raises(EntityNotFoundError):
        await cached_context.get_entity(entity_id)


@pytest.mark.asyncio
async def test_transaction_behavior(cached_context):
    """Test that caching works correctly with transactions."""
    # Start transaction for writing
    await cached_context.begin_transaction()

    # Create an entity
    entity_id = await cached_context.create_entity("person", {"name": "Alice"})

    # Commit to make it persistent
    await cached_context.commit_transaction()

    # Get it to ensure it's cached
    entity = await cached_context.get_entity(entity_id)
    assert entity["properties"]["name"] == "Alice"

    # Start a new transaction
    await cached_context.begin_transaction()

    # Verify all caches were cleared
    entity_cache = cached_context._cache_manager.store_manager.get_entity_store()
    assert await entity_cache.get(entity_id) is None

    # Update the entity in transaction
    await cached_context.update_entity(entity_id, {"name": "Alice Smith"})

    # Get the entity - should see transaction changes
    updated_entity = await cached_context.get_entity(entity_id)

    # The mock will return the updated entity directly from memory
    assert updated_entity["properties"]["name"] == "Alice Smith"

    # Rollback the transaction
    await cached_context.rollback_transaction()

    # Refresh our mock context's state to match the rollback behavior
    # (In a real context this would happen automatically)
    base_context = cached_context._base
    base_context.entities[entity_id]["properties"]["name"] = "Alice"

    # Get the entity - should see original state
    entity = await cached_context.get_entity(entity_id)
    assert entity["properties"]["name"] == "Alice"


@pytest.mark.asyncio
async def test_query_caching(cached_context):
    """Test that query results are properly cached."""
    # Start transaction for writing
    await cached_context.begin_transaction()

    # Create some test entities
    await cached_context.create_entity("person", {"name": "Alice"})
    await cached_context.create_entity("person", {"name": "Bob"})

    # Commit to make them persistent
    await cached_context.commit_transaction()

    # Run the query once
    query_spec = {"type": "person"}
    results1 = await cached_context.query(query_spec)
    assert len(results1) == 2

    # Verify it's in the cache
    query_hash = cached_context._cache_manager._hash_query(query_spec)
    query_cache = cached_context._cache_manager.store_manager.get_query_store()
    cached_entry = await query_cache.get(query_hash)
    assert cached_entry is not None

    # Run the same query again - should use cache
    results2 = await cached_context.query(query_spec)
    assert results1 == results2


@pytest.mark.asyncio
async def test_traversal_caching(cached_context):
    """Test that traversal results are properly cached."""
    # Start transaction for writing
    await cached_context.begin_transaction()

    # Create some test entities and relations
    entity1_id = await cached_context.create_entity("person", {"name": "Alice"})
    entity2_id = await cached_context.create_entity("person", {"name": "Bob"})
    await cached_context.create_relation("knows", entity1_id, entity2_id)

    # Commit to make them persistent
    await cached_context.commit_transaction()

    # Run the traversal once
    traversal_spec = {"relation_type": "knows"}
    results1 = await cached_context.traverse(entity1_id, traversal_spec)
    assert len(results1) == 1

    # Verify it's in the cache
    traversal_hash = cached_context._cache_manager._hash_query(traversal_spec)
    traversal_cache = cached_context._cache_manager.store_manager.get_traversal_store()
    cached_entry = await traversal_cache.get(traversal_hash)
    assert cached_entry is not None

    # Run the same traversal again - should use cache
    results2 = await cached_context.traverse(entity1_id, traversal_spec)
    assert results1 == results2


@pytest.mark.asyncio
async def test_cache_enable_disable(cached_context):
    """Test that cache can be enabled and disabled."""
    # Create a new context with a mock cache manager to track calls
    base_context = MockGraphContext()
    mock_cache_manager = CacheManager()

    # Replace the store_manager.get_entity_store().get/set methods with mocks
    entity_store = mock_cache_manager.store_manager.get_entity_store()
    original_get = entity_store.get
    original_set = entity_store.set

    entity_store.get = AsyncMock(return_value=None)
    entity_store.set = AsyncMock()

    test_context = CachedGraphContext(base_context, mock_cache_manager)

    try:
        # Start transaction for writing
        await test_context.begin_transaction()

        # Create an entity to work with
        entity_id = await test_context.create_entity("person", {"name": "Alice"})

        # Commit transaction
        await test_context.commit_transaction()

        # Reset the mocks to track new calls
        entity_store.get.reset_mock()
        entity_store.set.reset_mock()

        # CASE 1: With caching enabled (default)
        # Get entity should try to read from cache and then store result
        await test_context.get_entity(entity_id)

        # Verify cache was checked and then updated
        assert entity_store.get.called, "Cache get not called with caching enabled"
        assert entity_store.set.called, "Cache set not called with caching enabled"

        # Reset mocks again
        entity_store.get.reset_mock()
        entity_store.set.reset_mock()

        # CASE 2: With caching disabled
        test_context.disable_caching()

        # Get entity should bypass cache
        await test_context.get_entity(entity_id)

        # Verify cache was NOT checked or updated
        assert not entity_store.get.called, "Cache get called with caching disabled"
        assert not entity_store.set.called, "Cache set called with caching disabled"

        # Reset mocks again
        entity_store.get.reset_mock()
        entity_store.set.reset_mock()

        # CASE 3: Re-enable caching
        test_context.enable_caching()

        # Get the new fresh store and mock its methods
        entity_store = test_context._cache_manager.store_manager.get_entity_store()
        # Save original methods for cleanup
        new_original_get = entity_store.get
        new_original_set = entity_store.set
        # Replace with mocks
        entity_store.get = AsyncMock(return_value=None)
        entity_store.set = AsyncMock()

        # Get entity should use cache again
        await test_context.get_entity(entity_id)

        # Verify cache was checked and updated again
        assert entity_store.get.called, "Cache get not called after re-enabling caching"
        assert entity_store.set.called, "Cache set not called after re-enabling caching"

        # Restore the newest original methods
        entity_store.get = new_original_get
        entity_store.set = new_original_set

    finally:
        # Restore original methods
        if 'original_get' in locals() and entity_store:
            entity_store.get = original_get
        if 'original_set' in locals() and entity_store:
            entity_store.set = original_set
        if 'new_original_get' in locals() and entity_store:
            entity_store.get = new_original_get
        if 'new_original_set' in locals() and entity_store:
            entity_store.set = new_original_set
        await test_context.cleanup()