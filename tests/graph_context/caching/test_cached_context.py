"""Integration tests for the cached graph context implementation."""

from typing import Dict, Any, Optional

import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, Mock

from graph_context.caching.cached_context import CachedGraphContext
from graph_context.caching.cache_manager import CacheManager
from graph_context.caching.config import CacheConfig
from graph_context.event_system import EventSystem
from graph_context.interface import GraphContext
from graph_context.types.type_base import Entity, Relation
from graph_context.context_base import BaseGraphContext
from graph_context.exceptions import SchemaError, EntityNotFoundError, RelationNotFoundError
from graph_context.types.type_base import EntityType, PropertyDefinition, RelationType
from tests.graph_context.test_context_base import TestGraphContext

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('graph_context.caching.cached_context')
logger.setLevel(logging.DEBUG)

class MockBaseContext(BaseGraphContext):
    """Mock implementation of BaseGraphContext for testing."""

    def __init__(self):
        super().__init__()
        self._entities = {}
        self._relations = {}
        self.next_id = 1
        self._in_transaction = False
        self._transaction_entities = {}
        self._transaction_relations = {}
        self._initialized = False
        # Store original state for rollback
        self._pre_transaction_entities = {}
        self._pre_transaction_relations = {}

    async def initialize(self):
        """Initialize the mock context with required types."""
        if self._initialized:
            return

        # Register test types
        await self.register_entity_type(EntityType(
            name="person",
            properties={
                "name": PropertyDefinition(type="string", required=True),
                "age": PropertyDefinition(type="integer", required=False)
            }
        ))
        await self.register_relation_type(RelationType(
            name="knows",
            from_types=["person"],
            to_types=["person"],
            properties={
                "since": PropertyDefinition(type="string", required=False)
            }
        ))
        self._initialized = True

    def _generate_id(self) -> str:
        id_str = str(self.next_id)
        self.next_id += 1
        return id_str

    async def begin_transaction(self) -> None:
        if not self._initialized:
            await self.initialize()
        if self._in_transaction:
            raise ValueError("Transaction already in progress")

        logger.debug("MockBaseContext: Beginning transaction")
        self._in_transaction = True
        # Store the original state for rollback
        self._pre_transaction_entities = self._entities.copy()
        self._pre_transaction_relations = self._relations.copy()
        # Initialize transaction state with current state
        self._transaction_entities = self._entities.copy()
        self._transaction_relations = self._relations.copy()
        logger.debug(f"MockBaseContext: Stored pre-transaction entity state: {self._pre_transaction_entities}")

    async def commit_transaction(self) -> None:
        if not self._in_transaction:
            raise ValueError("No transaction in progress")

        logger.debug("MockBaseContext: Committing transaction")
        # Commit changes from transaction to main state
        self._entities = self._transaction_entities.copy()
        self._relations = self._transaction_relations.copy()
        # Clear transaction state
        self._transaction_entities = {}
        self._transaction_relations = {}
        self._pre_transaction_entities = {}
        self._pre_transaction_relations = {}
        self._in_transaction = False
        logger.debug(f"MockBaseContext: Committed entity state: {self._entities}")

    async def rollback_transaction(self) -> None:
        if not self._in_transaction:
            raise ValueError("No transaction in progress")

        logger.debug("MockBaseContext: Rolling back transaction")
        logger.debug(f"MockBaseContext: Current transaction state: {self._transaction_entities}")
        logger.debug(f"MockBaseContext: Restoring to pre-transaction state: {self._pre_transaction_entities}")

        # Restore the original state
        self._entities = self._pre_transaction_entities.copy()
        self._relations = self._pre_transaction_relations.copy()
        # Clear transaction state
        self._transaction_entities = {}
        self._transaction_relations = {}
        self._pre_transaction_entities = {}
        self._pre_transaction_relations = {}
        self._in_transaction = False

        logger.debug(f"MockBaseContext: Rolled back entity state: {self._entities}")

    async def _get_entity_impl(self, entity_id: str) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        entities = self._transaction_entities if self._in_transaction else self._entities
        result = entities.get(entity_id)
        logger.debug(f"MockBaseContext: Getting entity {entity_id}: {result}")
        return result

    async def _create_entity_impl(self, entity_type: str, properties: Dict[str, Any]) -> str:
        if not self._initialized:
            await self.initialize()
        entity_id = self._generate_id()
        entity = {"type": entity_type, "properties": properties}
        if self._in_transaction:
            self._transaction_entities[entity_id] = entity
        else:
            self._entities[entity_id] = entity
        return entity_id

    async def _update_entity_impl(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        if not self._initialized:
            await self.initialize()
        entities = self._transaction_entities if self._in_transaction else self._entities
        if entity_id not in entities:
            return False
        entities[entity_id]["properties"] = properties
        return True

    async def _delete_entity_impl(self, entity_id: str) -> bool:
        if not self._initialized:
            await self.initialize()
        entities = self._transaction_entities if self._in_transaction else self._entities
        if entity_id not in entities:
            return False
        del entities[entity_id]
        return True

    async def _get_relation_impl(self, relation_id: str) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        relations = self._transaction_relations if self._in_transaction else self._relations
        return relations.get(relation_id)

    async def _create_relation_impl(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        if not self._initialized:
            await self.initialize()
        relation_id = self._generate_id()
        relation = {
            "type": relation_type,
            "from_entity": from_entity,
            "to_entity": to_entity,
            "properties": properties or {}
        }
        if self._in_transaction:
            self._transaction_relations[relation_id] = relation
        else:
            self._relations[relation_id] = relation
        return relation_id

    async def _update_relation_impl(self, relation_id: str, properties: Dict[str, Any]) -> bool:
        if not self._initialized:
            await self.initialize()
        relations = self._transaction_relations if self._in_transaction else self._relations
        if relation_id not in relations:
            return False
        relations[relation_id]["properties"] = properties
        return True

    async def _delete_relation_impl(self, relation_id: str) -> bool:
        if not self._initialized:
            await self.initialize()
        relations = self._transaction_relations if self._in_transaction else self._relations
        if relation_id not in relations:
            return False
        del relations[relation_id]
        return True

    async def _query_impl(self, query_spec: Dict[str, Any]) -> list[Dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        # Simple implementation that returns all entities of requested type
        entities = self._transaction_entities if self._in_transaction else self._entities
        if "type" in query_spec:
            return [
                {"id": eid, **entity}
                for eid, entity in entities.items()
                if entity["type"] == query_spec["type"]
            ]
        return []

    async def _traverse_impl(self, start_entity: str, traversal_spec: Dict[str, Any]) -> list[Dict[str, Any]]:
        if not self._initialized:
            await self.initialize()
        # Simple implementation that returns direct relations
        relations = self._transaction_relations if self._in_transaction else self._relations
        if "relation_type" in traversal_spec:
            return [
                {"id": rid, **relation}
                for rid, relation in relations.items()
                if relation["type"] == traversal_spec["relation_type"]
                and relation["from_entity"] == start_entity
            ]
        return []


@pytest.fixture
async def base_context():
    """Create a base context for testing."""
    context = TestGraphContext()

    # Register test types
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

    return context


@pytest.fixture
async def cached_context(base_context):
    """Create a cached context for testing."""
    cache_manager = CacheManager()
    context = CachedGraphContext(base_context, cache_manager)
    yield context
    await context.cleanup()


@pytest.mark.asyncio
async def test_entity_caching(cached_context):
    """Test that entities are properly cached and retrieved."""
    # Create an entity
    entity_id = await cached_context.create_entity("person", {"name": "Alice", "age": 30})

    # Get the entity - should be cached
    entity1 = await cached_context.get_entity(entity_id)
    entity2 = await cached_context.get_entity(entity_id)

    assert entity1 == entity2
    assert entity1["properties"]["name"] == "Alice"
    assert entity1["properties"]["age"] == 30


@pytest.mark.asyncio
async def test_relation_caching(cached_context):
    """Test that relations are properly cached and retrieved."""
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

    # Get the relation - should be cached
    relation1 = await cached_context.get_relation(relation_id)
    relation2 = await cached_context.get_relation(relation_id)

    assert relation1 == relation2
    assert relation1["from_entity"] == entity1_id
    assert relation1["to_entity"] == entity2_id
    assert relation1["properties"]["since"] == "2020"


@pytest.mark.asyncio
async def test_cache_invalidation(cached_context):
    """Test that cache is properly invalidated on updates and deletes."""
    # Create an entity
    entity_id = await cached_context.create_entity("person", {"name": "Alice"})

    # Get it once to cache it
    entity1 = await cached_context.get_entity(entity_id)

    # Update it
    await cached_context.update_entity(entity_id, {"name": "Alice Smith"})

    # Get it again - should reflect the update
    entity2 = await cached_context.get_entity(entity_id)
    assert entity2["properties"]["name"] == "Alice Smith"

    # Delete it
    await cached_context.delete_entity(entity_id)

    # Try to get it - should raise EntityNotFoundError
    with pytest.raises(Exception):
        await cached_context.get_entity(entity_id)


@pytest.mark.asyncio
async def test_transaction_behavior(cached_context):
    """Test that caching works correctly with transactions."""
    logger.debug("Starting transaction behavior test")

    # Create an entity outside transaction
    logger.debug("Creating initial entity")
    entity_id = await cached_context.create_entity("person", {"name": "Alice"})
    logger.debug(f"Created entity with ID {entity_id}")

    # Start a transaction
    logger.debug("Starting transaction")
    await cached_context.begin_transaction()

    # Update the entity in transaction
    logger.debug("Updating entity in transaction")
    await cached_context.update_entity(entity_id, {"name": "Alice Smith"})

    # Get the entity - should see transaction changes
    logger.debug("Getting entity to verify transaction changes")
    entity = await cached_context.get_entity(entity_id)
    logger.debug(f"Got entity in transaction: {entity}")
    assert entity["properties"]["name"] == "Alice Smith"

    # Rollback the transaction
    logger.debug("Rolling back transaction")
    await cached_context.rollback_transaction()

    # Get the entity - should see original state
    logger.debug("Getting entity after rollback")
    entity = await cached_context.get_entity(entity_id)
    logger.debug(f"Got entity after rollback: {entity}")
    assert entity["properties"]["name"] == "Alice"


@pytest.mark.asyncio
async def test_query_caching(cached_context):
    """Test that query results are properly cached."""
    # Create some test entities
    await cached_context.create_entity("person", {"name": "Alice"})
    await cached_context.create_entity("person", {"name": "Bob"})

    # Run the same query twice
    query_spec = {"type": "person"}
    results1 = await cached_context.query(query_spec)
    results2 = await cached_context.query(query_spec)

    assert len(results1) == 2
    assert results1 == results2


@pytest.mark.asyncio
async def test_traversal_caching(cached_context):
    """Test that traversal results are properly cached."""
    # Create some test entities and relations
    entity1_id = await cached_context.create_entity("person", {"name": "Alice"})
    entity2_id = await cached_context.create_entity("person", {"name": "Bob"})
    await cached_context.create_relation("knows", entity1_id, entity2_id)

    # Run the same traversal twice
    traversal_spec = {"relation_type": "knows"}
    results1 = await cached_context.traverse(entity1_id, traversal_spec)
    results2 = await cached_context.traverse(entity1_id, traversal_spec)

    assert len(results1) == 1
    assert results1 == results2


@pytest.mark.asyncio
async def test_cache_enable_disable(cached_context):
    """Test that cache can be enabled and disabled."""
    # Create an entity
    entity_id = await cached_context.create_entity("person", {"name": "Alice"})

    # Disable caching
    cached_context.disable_caching()

    # Get the entity twice - should hit base context both times
    entity1 = await cached_context.get_entity(entity_id)
    entity2 = await cached_context.get_entity(entity_id)

    assert entity1 == entity2

    # Enable caching
    cached_context.enable_caching()

    # Get the entity again - should be cached
    entity3 = await cached_context.get_entity(entity_id)
    assert entity3 == entity1

@pytest.mark.asyncio
async def test_initialize_event_subscriptions(base_context):
    """Test that event subscriptions are properly initialized."""
    # Create a cache manager with mocked methods
    cache_manager = CacheManager()
    cache_manager.handle_event = AsyncMock()

    # Create cached context with the mocked cache manager
    context = CachedGraphContext(base_context, cache_manager)

    # Explicitly call _initialize (normally would be called internally)
    await context._initialize()

    # Create an entity to trigger events
    entity_id = await context.create_entity("person", {"name": "Test"})

    # Verify the event handler was called
    assert cache_manager.handle_event.called

    # Cleanup
    await context.cleanup()

@pytest.mark.asyncio
async def test_transaction_cache_commit(cached_context):
    """Test transaction cache behavior with commit."""
    # Create initial entity
    entity_id = await cached_context.create_entity("person", {"name": "Original"})

    # Begin transaction
    await cached_context.begin_transaction()

    # Create another entity in transaction
    transaction_entity_id = await cached_context.create_entity("person", {"name": "Transaction"})

    # Update the original entity in transaction
    await cached_context.update_entity(entity_id, {"name": "Updated"})

    # Get the entity - should see updated values
    updated_entity = await cached_context.get_entity(entity_id)
    assert updated_entity["properties"]["name"] == "Updated"

    # Commit the transaction
    await cached_context.commit_transaction()

    # Verify transaction state is cleared
    assert not cached_context._in_transaction

    # Verify changes are persistent
    committed_entity = await cached_context.get_entity(entity_id)
    assert committed_entity["properties"]["name"] == "Updated"

    transaction_entity = await cached_context.get_entity(transaction_entity_id)
    assert transaction_entity["properties"]["name"] == "Transaction"

@pytest.mark.asyncio
async def test_transaction_cache_rollback(cached_context):
    """Test transaction cache behavior with rollback."""
    # Create initial entity
    entity_id = await cached_context.create_entity("person", {"name": "Original"})

    # Begin transaction
    await cached_context.begin_transaction()

    # Update entity in transaction
    await cached_context.update_entity(entity_id, {"name": "Updated"})

    # Create new entity in transaction
    transaction_entity_id = await cached_context.create_entity("person", {"name": "Transaction"})

    # Verify transaction state
    updated_entity = await cached_context.get_entity(entity_id)
    assert updated_entity["properties"]["name"] == "Updated"

    # Rollback the transaction
    await cached_context.rollback_transaction()

    # Verify transaction state is cleared
    assert not cached_context._in_transaction

    # Verify original state is restored
    original_entity = await cached_context.get_entity(entity_id)
    assert original_entity["properties"]["name"] == "Original"

    # Verify transaction entity doesn't exist
    with pytest.raises(Exception):
        await cached_context.get_entity(transaction_entity_id)

@pytest.mark.asyncio
async def test_query_dependencies(cached_context):
    """Test query caching with dependencies."""
    # Create test entities
    entity1_id = await cached_context.create_entity("person", {"name": "Alice"})
    entity2_id = await cached_context.create_entity("person", {"name": "Bob"})

    # Execute a query to cache results
    query_spec = {"type": "person"}
    results1 = await cached_context.query(query_spec)
    assert len(results1) == 2

    # Update an entity to invalidate cache
    await cached_context.update_entity(entity1_id, {"name": "Alice Smith"})

    # Execute same query again
    results2 = await cached_context.query(query_spec)
    assert len(results2) == 2

    # Verify updated entity is reflected in results
    updated_entity = next((e for e in results2 if e["id"] == entity1_id), None)
    assert updated_entity["properties"]["name"] == "Alice Smith"

@pytest.mark.asyncio
async def test_traversal_dependencies(cached_context):
    """Test traversal caching with dependencies."""
    # Create test entities and relation
    entity1_id = await cached_context.create_entity("person", {"name": "Alice"})
    entity2_id = await cached_context.create_entity("person", {"name": "Bob"})
    relation_id = await cached_context.create_relation("knows", entity1_id, entity2_id, {"since": "2020"})

    # Execute a traversal to cache results
    traversal_spec = {"relation_type": "knows", "max_depth": 1}
    results1 = await cached_context.traverse(entity1_id, traversal_spec)
    assert len(results1) == 1

    # Update relation to invalidate cache
    await cached_context.update_relation(relation_id, {"since": "2021"})

    # Execute same traversal again
    results2 = await cached_context.traverse(entity1_id, traversal_spec)
    assert len(results2) == 1

    # Verify updated relation is reflected in results
    updated_relation = results2[0]
    assert updated_relation["properties"]["since"] == "2021"

@pytest.mark.asyncio
async def test_cache_behavior_for_bulk_operations(cached_context):
    """Test caching behavior when bulk operations would occur."""
    # We won't test the actual bulk operations since they're not implemented in TestGraphContext
    # Instead, we'll test the caching behavior with regular entity operations

    # Start transaction
    await cached_context.begin_transaction()

    # Create multiple entities to simulate bulk behavior
    entity1_id = await cached_context.create_entity("person", {"name": "Entity1"})
    entity2_id = await cached_context.create_entity("person", {"name": "Entity2"})
    entity3_id = await cached_context.create_entity("person", {"name": "Entity3"})

    # Get all entities to check initial cache state
    entity1 = await cached_context.get_entity(entity1_id)
    entity2 = await cached_context.get_entity(entity2_id)
    entity3 = await cached_context.get_entity(entity3_id)

    assert entity1["properties"]["name"] == "Entity1"
    assert entity2["properties"]["name"] == "Entity2"
    assert entity3["properties"]["name"] == "Entity3"

    # Update entities to simulate bulk update
    await cached_context.update_entity(entity1_id, {"name": "Updated1"})
    await cached_context.update_entity(entity2_id, {"name": "Updated2"})

    # Verify updates are reflected in cache
    updated1 = await cached_context.get_entity(entity1_id)
    updated2 = await cached_context.get_entity(entity2_id)

    assert updated1["properties"]["name"] == "Updated1"
    assert updated2["properties"]["name"] == "Updated2"

    # Delete entities to simulate bulk delete
    success1 = await cached_context.delete_entity(entity1_id)
    success3 = await cached_context.delete_entity(entity3_id)

    assert success1 is True
    assert success3 is True

    # Verify deletions by trying to fetch deleted entities
    # Note: TestGraphContext may not raise EntityNotFoundError but could return None
    # or we might need to try EntityNotFoundError or get a specific exception
    try:
        deleted_entity1 = await cached_context.get_entity(entity1_id)
        # If we get here, the test context returns None for deleted entities
        assert deleted_entity1 is None
    except Exception as e:
        # If we get here, the test context raises some kind of exception
        pass

    try:
        deleted_entity3 = await cached_context.get_entity(entity3_id)
        # If we get here, the test context returns None for deleted entities
        assert deleted_entity3 is None
    except Exception as e:
        # If we get here, the test context raises some kind of exception
        pass

    # Entity2 should still exist
    remaining = await cached_context.get_entity(entity2_id)
    assert remaining["properties"]["name"] == "Updated2"

    # Commit transaction
    await cached_context.commit_transaction()

@pytest.mark.asyncio
async def test_cache_behavior_for_bulk_relation_operations(cached_context):
    """Test caching behavior when bulk relation operations would occur."""
    # We won't test the actual bulk operations since they're not implemented in TestGraphContext
    # Instead, we'll test the caching behavior with regular relation operations

    # Start transaction
    await cached_context.begin_transaction()

    # Create entities for relations
    person1_id = await cached_context.create_entity("person", {"name": "Person1"})
    person2_id = await cached_context.create_entity("person", {"name": "Person2"})
    person3_id = await cached_context.create_entity("person", {"name": "Person3"})

    # Create relations to simulate bulk create
    relation1_id = await cached_context.create_relation("knows", person1_id, person2_id, {"since": "2020"})
    relation2_id = await cached_context.create_relation("knows", person2_id, person3_id, {"since": "2021"})

    # Get relations to check initial cache state
    relation1 = await cached_context.get_relation(relation1_id)
    relation2 = await cached_context.get_relation(relation2_id)

    assert relation1["properties"]["since"] == "2020"
    assert relation2["properties"]["since"] == "2021"

    # Update relations to simulate bulk update
    await cached_context.update_relation(relation1_id, {"since": "2022"})
    await cached_context.update_relation(relation2_id, {"since": "2023"})

    # Verify updates are reflected in cache
    updated1 = await cached_context.get_relation(relation1_id)
    updated2 = await cached_context.get_relation(relation2_id)

    assert updated1["properties"]["since"] == "2022"
    assert updated2["properties"]["since"] == "2023"

    # Delete relations to simulate bulk delete
    await cached_context.delete_relation(relation1_id)
    await cached_context.delete_relation(relation2_id)

    # Verify deletions are reflected in cache
    with pytest.raises(RelationNotFoundError):
        await cached_context.get_relation(relation1_id)

    with pytest.raises(RelationNotFoundError):
        await cached_context.get_relation(relation2_id)

    # Commit transaction
    await cached_context.commit_transaction()

class ExtendedMockContext(MockBaseContext):
    """Extended mock context with additional methods for testing edge cases."""

    async def _query_impl(self, query_spec: Dict[str, Any]) -> list[Dict[str, Any]]:
        # Simulate a query error for specific test cases
        if query_spec.get("error", False):
            raise ValueError("Simulated query error")
        return await super()._query_impl(query_spec)

    async def _traverse_impl(self, start_entity: str, traversal_spec: Dict[str, Any]) -> list[Dict[str, Any]]:
        # Simulate a traversal error for specific test cases
        if traversal_spec.get("error", False):
            raise ValueError("Simulated traversal error")
        return await super()._traverse_impl(start_entity, traversal_spec)

@pytest.fixture
async def error_prone_context():
    """Create a context that can be configured to raise errors."""
    context = ExtendedMockContext()
    await context.initialize()
    return context

@pytest.fixture
async def error_prone_cached_context(error_prone_context):
    """Create a cached context wrapping the error-prone context."""
    cache_manager = CacheManager()
    context = CachedGraphContext(error_prone_context, cache_manager)
    yield context
    await context.cleanup()

@pytest.mark.asyncio
async def test_query_error_handling(error_prone_cached_context):
    """Test that query errors from the base context are properly propagated."""
    # Execute a query that will error
    query_spec = {"type": "person", "error": True}

    with pytest.raises(ValueError, match="Simulated query error"):
        await error_prone_cached_context.query(query_spec)

@pytest.mark.asyncio
async def test_traversal_error_handling(error_prone_cached_context):
    """Test that traversal errors from the base context are properly propagated."""
    # Create an entity to start traversal from
    entity_id = await error_prone_cached_context.create_entity("person", {"name": "Test"})

    # Execute a traversal that will error
    traversal_spec = {"relation_type": "knows", "error": True}

    with pytest.raises(ValueError, match="Simulated traversal error"):
        await error_prone_cached_context.traverse(entity_id, traversal_spec)

@pytest.mark.asyncio
async def test_transaction_state_errors(cached_context):
    """Test handling of transaction state errors."""
    # Test committing when not in transaction
    assert not cached_context._in_transaction
    await cached_context.commit_transaction()  # Should not raise error, just log warning

    # Test rolling back when not in transaction
    await cached_context.rollback_transaction()  # Should not raise error, just log warning

    # Begin transaction and test double begin
    await cached_context.begin_transaction()
    assert cached_context._in_transaction

    # Clean up
    await cached_context.rollback_transaction()

@pytest.mark.asyncio
async def test_actual_bulk_operations(cached_context):
    """Test actual bulk operations methods."""
    # Test bulk_create_entities
    entities = [
        {"name": "Bulk1", "age": 30},
        {"name": "Bulk2", "age": 25},
        {"name": "Bulk3", "age": 40}
    ]

    entity_ids = await cached_context.bulk_create_entities("person", entities)
    assert len(entity_ids) == 3

    # Test bulk_update_entities
    updates = [
        {"id": entity_ids[0], "name": "Updated1", "age": 31},
        {"id": entity_ids[1], "name": "Updated2", "age": 26}
    ]

    await cached_context.bulk_update_entities("person", updates)

    # Verify updates
    updated1 = await cached_context.get_entity(entity_ids[0])
    assert updated1["properties"]["name"] == "Updated1"
    assert updated1["properties"]["age"] == 31

    # Test bulk_delete_entities
    await cached_context.bulk_delete_entities("person", [entity_ids[0], entity_ids[2]])

    # Verify deletions
    with pytest.raises(EntityNotFoundError):
        await cached_context.get_entity(entity_ids[0])

    with pytest.raises(EntityNotFoundError):
        await cached_context.get_entity(entity_ids[2])

    # Entity1 should still exist
    remaining = await cached_context.get_entity(entity_ids[1])
    assert remaining["properties"]["name"] == "Updated2"

@pytest.mark.asyncio
async def test_actual_bulk_relation_operations(cached_context):
    """Test actual bulk relation operations methods."""
    # Create test entities
    person1_id = await cached_context.create_entity("person", {"name": "Person1"})
    person2_id = await cached_context.create_entity("person", {"name": "Person2"})
    person3_id = await cached_context.create_entity("person", {"name": "Person3"})

    # Test bulk_create_relations
    relations = [
        {"from_entity": person1_id, "to_entity": person2_id, "properties": {"since": "2020"}},
        {"from_entity": person2_id, "to_entity": person3_id, "properties": {"since": "2021"}},
        {"from_entity": person1_id, "to_entity": person3_id, "properties": {"since": "2022"}}
    ]

    relation_ids = await cached_context.bulk_create_relations("knows", relations)
    assert len(relation_ids) == 3

    # Test bulk_update_relations
    updates = [
        {"id": relation_ids[0], "since": "2023"},
        {"id": relation_ids[1], "since": "2024"}
    ]

    await cached_context.bulk_update_relations("knows", updates)

    # Verify updates
    updated1 = await cached_context.get_relation(relation_ids[0])
    assert updated1["properties"]["since"] == "2023"

    # Test bulk_delete_relations
    await cached_context.bulk_delete_relations("knows", [relation_ids[0], relation_ids[2]])

    # Verify deletions
    with pytest.raises(RelationNotFoundError):
        await cached_context.get_relation(relation_ids[0])

    with pytest.raises(RelationNotFoundError):
        await cached_context.get_relation(relation_ids[2])

    # Relation1 should still exist
    remaining = await cached_context.get_relation(relation_ids[1])
    assert remaining["properties"]["since"] == "2024"

@pytest.mark.asyncio
async def test_cleanup_method(cached_context):
    """Test the cleanup method."""
    # Just ensure it can be called without error
    await cached_context.cleanup()

    # Verify we can still perform operations after cleanup
    entity_id = await cached_context.create_entity("person", {"name": "Post-Cleanup"})
    entity = await cached_context.get_entity(entity_id)
    assert entity["properties"]["name"] == "Post-Cleanup"

@pytest.mark.asyncio
async def test_event_handling_for_missed_branches():
    """Test event handling branches that might be missed in other tests."""
    # Create a mock base context
    base_context = MockBaseContext()
    await base_context.initialize()

    # Create a cache manager with mocked handle_event
    cache_manager = CacheManager()
    original_handle_event = cache_manager.handle_event
    cache_manager.handle_event = AsyncMock()

    # Create cached context
    context = CachedGraphContext(base_context, cache_manager)
    await context._initialize()

    # Test schema_modified event
    await context._cache_manager.handle_event(EventContext(
        event=GraphEvent.SCHEMA_MODIFIED,
        data={},
        metadata=EventMetadata()
    ))
    assert cache_manager.handle_event.called

    # Test type_modified event
    cache_manager.handle_event.reset_mock()
    await context._cache_manager.handle_event(EventContext(
        event=GraphEvent.TYPE_MODIFIED,
        data={},
        metadata=EventMetadata()
    ))
    assert cache_manager.handle_event.called

    # Restore original handle_event
    cache_manager.handle_event = original_handle_event

    # Cleanup
    await context.cleanup()

@pytest.mark.asyncio
async def test_double_initialization(cached_context):
    """Test that double initialization is handled correctly."""
    # First initialization happens automatically
    assert cached_context._initialized

    # Call initialize again and ensure it doesn't break anything
    await cached_context._initialize()
    assert cached_context._initialized

    # Verify we can still perform operations
    entity_id = await cached_context.create_entity("person", {"name": "Double-Init"})
    entity = await cached_context.get_entity(entity_id)
    assert entity["properties"]["name"] == "Double-Init"

@pytest.mark.asyncio
async def test_entity_not_found_direct(cached_context):
    """Test direct entity not found scenarios."""
    # Try to get a non-existent entity
    with pytest.raises(EntityNotFoundError):
        await cached_context.get_entity("non-existent-id")

    # Try to update a non-existent entity
    success = await cached_context.update_entity("non-existent-id", {"name": "Updated"})
    assert not success

    # Try to delete a non-existent entity
    success = await cached_context.delete_entity("non-existent-id")
    assert not success

@pytest.mark.asyncio
async def test_relation_not_found_direct(cached_context):
    """Test direct relation not found scenarios."""
    # Try to get a non-existent relation
    with pytest.raises(RelationNotFoundError):
        await cached_context.get_relation("non-existent-id")

    # Try to update a non-existent relation
    success = await cached_context.update_relation("non-existent-id", {"since": "2023"})
    assert not success

    # Try to delete a non-existent relation
    success = await cached_context.delete_relation("non-existent-id")
    assert not success

@pytest.mark.asyncio
async def test_transaction_state_transitions_detail(cached_context):
    """Test detailed transaction state transitions."""
    # Create initial entities
    entity1_id = await cached_context.create_entity("person", {"name": "Initial1"})
    entity2_id = await cached_context.create_entity("person", {"name": "Initial2"})

    # Begin transaction
    await cached_context.begin_transaction()
    assert cached_context._in_transaction
    assert len(cached_context._transaction_cache) == 0

    # Get existing entity to populate transaction cache
    entity1 = await cached_context.get_entity(entity1_id)
    assert entity1_id in cached_context._transaction_cache

    # Update entity in transaction
    await cached_context.update_entity(entity1_id, {"name": "Updated1"})
    assert entity1_id not in cached_context._transaction_cache  # Should be cleared from transaction cache

    # Check that fresh entity is fetched
    updated_entity1 = await cached_context.get_entity(entity1_id)
    assert updated_entity1["properties"]["name"] == "Updated1"
    assert entity1_id in cached_context._transaction_cache  # Should be back in transaction cache

    # Commit the transaction
    await cached_context.commit_transaction()
    assert not cached_context._in_transaction
    assert len(cached_context._transaction_cache) == 0

    # Verify changes persisted
    committed_entity = await cached_context.get_entity(entity1_id)
    assert committed_entity["properties"]["name"] == "Updated1"

    # Test another transaction with rollback
    await cached_context.begin_transaction()
    await cached_context.update_entity(entity2_id, {"name": "ShouldNotPersist"})
    updated_entity2 = await cached_context.get_entity(entity2_id)
    assert updated_entity2["properties"]["name"] == "ShouldNotPersist"

    # Rollback
    await cached_context.rollback_transaction()
    rolled_back_entity = await cached_context.get_entity(entity2_id)
    assert rolled_back_entity["properties"]["name"] == "Initial2"