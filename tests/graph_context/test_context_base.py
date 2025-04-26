from unittest.mock import AsyncMock, MagicMock

import pytest

from graph_context.context_base import BaseGraphContext
from graph_context.exceptions import SchemaError
from graph_context.types.type_base import (
    EntityType,
    PropertyDefinition,
    RelationType,
)


class TestBaseGraphContext(BaseGraphContext):
    """Test implementation of BaseGraphContext that uses mocks for all managers."""

    def __init__(self):
        super().__init__()

        # Replace all managers with mocks
        self._store = AsyncMock()
        self._validator = MagicMock()
        self._transaction = MagicMock()
        self._events = AsyncMock()
        self._entity_manager = AsyncMock()
        self._relation_manager = AsyncMock()
        self._query_manager = AsyncMock()

        # Set up common is_in_transaction behavior
        self._transaction.is_in_transaction = MagicMock(return_value=False)


@pytest.fixture
async def base_context():
    """Create a BaseGraphContext with mocked components for testing."""
    context = TestBaseGraphContext()
    yield context


# Test that BaseGraphContext delegates to TransactionManager
@pytest.mark.asyncio
async def test_begin_transaction(base_context):
    """Test that begin_transaction delegates to the transaction manager."""
    # Set up the mock
    base_context._transaction.begin_transaction = AsyncMock()

    # Call the method
    await base_context.begin_transaction()

    # Verify delegation
    base_context._transaction.begin_transaction.assert_called_once()


@pytest.mark.asyncio
async def test_commit_transaction(base_context):
    """Test that commit_transaction delegates to the transaction manager."""
    # Set up the mock
    base_context._transaction.commit_transaction = AsyncMock()

    # Call the method
    await base_context.commit_transaction()

    # Verify delegation
    base_context._transaction.commit_transaction.assert_called_once()


@pytest.mark.asyncio
async def test_rollback_transaction(base_context):
    """Test that rollback_transaction delegates to the transaction manager."""
    # Set up the mock
    base_context._transaction.rollback_transaction = AsyncMock()

    # Call the method
    await base_context.rollback_transaction()

    # Verify delegation
    base_context._transaction.rollback_transaction.assert_called_once()


# Test that BaseGraphContext delegates to EntityManager
@pytest.mark.asyncio
async def test_get_entity(base_context):
    """Test that get_entity delegates to the entity manager."""
    # Set up the mock
    base_context._entity_manager.get = AsyncMock()
    entity_id = "test_id"

    # Call the method
    await base_context.get_entity(entity_id)

    # Verify delegation
    base_context._entity_manager.get.assert_called_once_with(entity_id)


@pytest.mark.asyncio
async def test_create_entity(base_context):
    """Test that create_entity delegates to the entity manager."""
    # Set up the mock
    base_context._entity_manager.create = AsyncMock()
    entity_type = "test_type"
    properties = {"name": "test"}

    # Call the method
    await base_context.create_entity(entity_type, properties)

    # Verify delegation
    base_context._entity_manager.create.assert_called_once_with(entity_type, properties)


@pytest.mark.asyncio
async def test_update_entity(base_context):
    """Test that update_entity delegates to the entity manager."""
    # Set up the mock
    base_context._entity_manager.update = AsyncMock()
    entity_id = "test_id"
    properties = {"name": "updated"}

    # Call the method
    await base_context.update_entity(entity_id, properties)

    # Verify delegation
    base_context._entity_manager.update.assert_called_once_with(entity_id, properties)


@pytest.mark.asyncio
async def test_delete_entity(base_context):
    """Test that delete_entity delegates to the entity manager."""
    # Set up the mock
    base_context._entity_manager.delete = AsyncMock()
    entity_id = "test_id"

    # Call the method
    await base_context.delete_entity(entity_id)

    # Verify delegation
    base_context._entity_manager.delete.assert_called_once_with(entity_id)


# Test that BaseGraphContext delegates to RelationManager
@pytest.mark.asyncio
async def test_get_relation(base_context):
    """Test that get_relation delegates to the relation manager."""
    # Set up the mock
    base_context._relation_manager.get = AsyncMock()
    relation_id = "test_id"

    # Call the method
    await base_context.get_relation(relation_id)

    # Verify delegation
    base_context._relation_manager.get.assert_called_once_with(relation_id)


@pytest.mark.asyncio
async def test_create_relation(base_context):
    """Test that create_relation delegates to the relation manager."""
    # Set up the mock
    base_context._relation_manager.create = AsyncMock()
    relation_type = "test_type"
    from_entity = "entity1"
    to_entity = "entity2"
    properties = {"label": "test"}

    # Call the method
    await base_context.create_relation(relation_type, from_entity, to_entity, properties)

    # Verify delegation
    base_context._relation_manager.create.assert_called_once_with(relation_type, from_entity, to_entity, properties)


@pytest.mark.asyncio
async def test_update_relation(base_context):
    """Test that update_relation delegates to the relation manager."""
    # Set up the mock
    base_context._relation_manager.update = AsyncMock()
    relation_id = "test_id"
    properties = {"label": "updated"}

    # Call the method
    await base_context.update_relation(relation_id, properties)

    # Verify delegation
    base_context._relation_manager.update.assert_called_once_with(relation_id, properties)


@pytest.mark.asyncio
async def test_delete_relation(base_context):
    """Test that delete_relation delegates to the relation manager."""
    # Set up the mock
    base_context._relation_manager.delete = AsyncMock()
    relation_id = "test_id"

    # Call the method
    await base_context.delete_relation(relation_id)

    # Verify delegation
    base_context._relation_manager.delete.assert_called_once_with(relation_id)


# Test that BaseGraphContext delegates to QueryManager
@pytest.mark.asyncio
async def test_query(base_context):
    """Test that query delegates to the query manager."""
    # Set up the mock
    base_context._query_manager.query = AsyncMock()
    query_spec = {"type": "Person"}

    # Call the method
    await base_context.query(query_spec)

    # Verify delegation
    base_context._query_manager.query.assert_called_once_with(query_spec)


@pytest.mark.asyncio
async def test_traverse(base_context):
    """Test that traverse delegates to the query manager."""
    # Set up the mock
    base_context._query_manager.traverse = AsyncMock()
    start_entity = "entity1"
    traversal_spec = {"max_depth": 2, "relation_types": ["knows"]}

    # Call the method
    await base_context.traverse(start_entity, traversal_spec)

    # Verify delegation
    base_context._query_manager.traverse.assert_called_once_with(start_entity, traversal_spec)


# Test validation methods
def test_validate_entity(base_context):
    """Test that validate_entity delegates to the validator."""
    # Set up the mock
    base_context._validator.validate_entity = MagicMock()
    entity_type = "Person"
    properties = {"name": "Test"}

    # Call the method
    base_context.validate_entity(entity_type, properties)

    # Verify delegation
    base_context._validator.validate_entity.assert_called_once_with(entity_type, properties)


def test_validate_relation(base_context):
    """Test that validate_relation delegates to the validator."""
    # Set up the mock
    base_context._validator.validate_relation = MagicMock()
    relation_type = "knows"
    from_entity_type = "Person"
    to_entity_type = "Person"
    properties = {"since": 2020}

    # Call the method
    base_context.validate_relation(relation_type, from_entity_type, to_entity_type, properties)

    # Verify delegation
    base_context._validator.validate_relation.assert_called_once_with(
        relation_type, from_entity_type, to_entity_type, properties
    )


# Test cleanup behavior
@pytest.mark.asyncio
async def test_cleanup(base_context):
    """Test cleanup method clears registries and handles transactions."""
    # Setup initial state
    base_context._transaction.is_in_transaction = MagicMock(return_value=True)
    base_context._transaction.rollback_transaction = AsyncMock()

    # Add some types to clear
    base_context._entity_types = {"Person": MagicMock()}
    base_context._relation_types = {"knows": MagicMock()}

    # Call cleanup
    await base_context.cleanup()

    # Verify transaction was rolled back
    base_context._transaction.rollback_transaction.assert_called_once()

    # Verify registries were cleared
    assert len(base_context._entity_types) == 0
    assert len(base_context._relation_types) == 0


# Test schema registration
@pytest.mark.asyncio
async def test_register_entity_type(base_context):
    """Test register_entity_type with a valid entity type."""
    # Setup
    base_context._events.emit = AsyncMock()
    entity_type = EntityType(
        name="TestType",
        properties={"name": PropertyDefinition(type="string", required=True)}
    )

    # Call the method
    await base_context.register_entity_type(entity_type)

    # Verify entity type was registered
    assert "TestType" in base_context._entity_types
    assert base_context._entity_types["TestType"] == entity_type

    # Verify events were emitted
    assert base_context._events.emit.call_count == 2


@pytest.mark.asyncio
async def test_register_entity_type_duplicate(base_context):
    """Test register_entity_type with a duplicate entity type."""
    # Setup
    entity_type = EntityType(
        name="TestType",
        properties={"name": PropertyDefinition(type="string", required=True)}
    )

    # Register once
    await base_context.register_entity_type(entity_type)

    # Try to register again
    with pytest.raises(SchemaError) as exc_info:
        await base_context.register_entity_type(entity_type)

    assert "Entity type already exists" in str(exc_info.value)


@pytest.mark.asyncio
async def test_register_relation_type(base_context):
    """Test register_relation_type with a valid relation type."""
    # Setup
    base_context._events.emit = AsyncMock()

    # Register required entity types
    person_type = EntityType(
        name="Person",
        properties={"name": PropertyDefinition(type="string", required=True)}
    )
    base_context._entity_types["Person"] = person_type

    # Create relation type
    relation_type = RelationType(
        name="knows",
        from_types=["Person"],
        to_types=["Person"]
    )

    # Call the method
    await base_context.register_relation_type(relation_type)

    # Verify relation type was registered
    assert "knows" in base_context._relation_types
    assert base_context._relation_types["knows"] == relation_type

    # Verify events were emitted
    assert base_context._events.emit.call_count == 2


@pytest.mark.asyncio
async def test_register_relation_type_unknown_entity(base_context):
    """Test register_relation_type with unknown entity types."""
    # Create relation type with unknown entity
    relation_type = RelationType(
        name="knows",
        from_types=["UnknownType"],
        to_types=["Person"]
    )

    # Call the method
    with pytest.raises(SchemaError) as exc_info:
        await base_context.register_relation_type(relation_type)

    assert "Unknown entity type in from_types" in str(exc_info.value)