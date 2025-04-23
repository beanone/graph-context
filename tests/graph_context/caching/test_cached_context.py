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
from graph_context.exceptions import SchemaError
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