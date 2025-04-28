"""Tests for the BaseGraphContext class."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from graph_context.context_base import BaseGraphContext
from graph_context.exceptions import SchemaError, TransactionError
from graph_context.types.type_base import EntityType, PropertyDefinition, RelationType


class MockBaseGraphContext(BaseGraphContext):
    """Test implementation of BaseGraphContext that uses mocks for all managers."""

    def __init__(self):
        """Initialize the mock base graph context."""
        super().__init__()

        # Replace managers with mocks, except transaction manager which we want to test
        self._store = AsyncMock()
        self._validator = MagicMock()
        self._events = AsyncMock()
        self._entity_manager = AsyncMock()
        self._relation_manager = AsyncMock()
        self._query_manager = AsyncMock()

        # Set up common is_in_transaction behavior
        self._transaction.is_in_transaction = MagicMock(return_value=False)


@pytest.fixture
async def base_context():
    """Create a BaseGraphContext for testing."""
    context = BaseGraphContext()
    yield context
    await context.cleanup()


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
    """Test that create_entity properly handles transaction state."""
    entity_type = "test_type"
    properties = {"name": "test"}

    # Register the entity type first
    await base_context.register_entity_type(
        EntityType(
            name=entity_type,
            properties={
                "name": PropertyDefinition(type="string", required=True),
            },
        )
    )

    # Test that operation fails when not in transaction
    with pytest.raises(TransactionError) as exc_info:
        await base_context.create_entity(entity_type, properties)
    assert "Operation requires an active transaction" in str(exc_info.value)

    # Start a transaction
    await base_context.begin_transaction()

    # Test that operation succeeds when in transaction
    entity_id = await base_context.create_entity(entity_type, properties)
    assert entity_id is not None

    # Verify entity was created
    entity = await base_context.get_entity(entity_id)
    assert entity is not None
    assert entity.type == entity_type
    assert entity.properties == properties

    # Commit the transaction
    await base_context.commit_transaction()

    # Test that operation fails again after transaction is committed
    with pytest.raises(TransactionError) as exc_info:
        await base_context.create_entity(entity_type, properties)
    assert "Operation requires an active transaction" in str(exc_info.value)


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
    await base_context.create_relation(
        relation_type, from_entity, to_entity, properties
    )

    # Verify delegation
    base_context._relation_manager.create.assert_called_once_with(
        relation_type, from_entity, to_entity, properties
    )


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
    base_context._relation_manager.update.assert_called_once_with(
        relation_id, properties
    )


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
    base_context._query_manager.traverse.assert_called_once_with(
        start_entity, traversal_spec
    )


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
    base_context._validator.validate_entity.assert_called_once_with(
        entity_type, properties
    )


def test_validate_relation(base_context):
    """Test that validate_relation delegates to the validator."""
    # Set up the mock
    base_context._validator.validate_relation = MagicMock()
    relation_type = "knows"
    from_entity_type = "Person"
    to_entity_type = "Person"
    properties = {"since": 2020}

    # Call the method
    base_context.validate_relation(
        relation_type, from_entity_type, to_entity_type, properties
    )

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
        properties={"name": PropertyDefinition(type="string", required=True)},
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
        properties={"name": PropertyDefinition(type="string", required=True)},
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
        properties={"name": PropertyDefinition(type="string", required=True)},
    )
    base_context._entity_types["Person"] = person_type

    # Create relation type
    relation_type = RelationType(
        name="knows", from_types=["Person"], to_types=["Person"]
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
        name="knows", from_types=["UnknownType"], to_types=["Person"]
    )

    # Call the method
    with pytest.raises(SchemaError) as exc_info:
        await base_context.register_relation_type(relation_type)

    assert "Unknown entity type in from_types" in str(exc_info.value)


@pytest.mark.asyncio
async def test_transaction_wrapping_for_entity_creation(base_context):
    """Test proper transaction wrapping for entity creation and related operations."""
    # Define test data
    person_type = "Person"
    company_type = "Company"
    person_props = {"name": "John Doe", "age": 30}
    company_props = {"name": "Acme Corp", "industry": "Technology"}
    relation_type = "WORKS_AT"
    relation_props = {"start_date": "2023-01-01", "role": "Engineer"}

    # Register entity types
    person_entity_type = EntityType(
        name=person_type,
        properties={
            "name": PropertyDefinition(type="string", required=True),
            "age": PropertyDefinition(type="integer", required=True),
        },
    )
    company_entity_type = EntityType(
        name=company_type,
        properties={
            "name": PropertyDefinition(type="string", required=True),
            "industry": PropertyDefinition(type="string", required=True),
        },
    )
    await base_context.register_entity_type(person_entity_type)
    await base_context.register_entity_type(company_entity_type)

    # Register relation type
    works_at_relation_type = RelationType(
        name=relation_type,
        from_types=[person_type],
        to_types=[company_type],
        properties={
            "start_date": PropertyDefinition(type="string", required=True),
            "role": PropertyDefinition(type="string", required=True),
        },
    )
    await base_context.register_relation_type(works_at_relation_type)

    try:
        # Start transaction
        await base_context.begin_transaction()

        # Create person entity
        person_id = await base_context.create_entity(person_type, person_props)
        assert person_id is not None

        # Create company entity
        company_id = await base_context.create_entity(company_type, company_props)
        assert company_id is not None

        # Create relation between person and company
        relation_id = await base_context.create_relation(
            relation_type, person_id, company_id, relation_props
        )
        assert relation_id is not None

        # Verify entities and relation exist within transaction
        person = await base_context.get_entity(person_id)
        assert person is not None
        assert person.type == person_type
        assert person.properties == person_props

        company = await base_context.get_entity(company_id)
        assert company is not None
        assert company.type == company_type
        assert company.properties == company_props

        relation = await base_context.get_relation(relation_id)
        assert relation is not None
        assert relation.type == relation_type
        assert relation.from_entity == person_id
        assert relation.to_entity == company_id
        assert relation.properties == relation_props

        # Commit transaction
        await base_context.commit_transaction()

        # Verify entities and relation still exist after commit
        person = await base_context.get_entity(person_id)
        assert person is not None
        company = await base_context.get_entity(company_id)
        assert company is not None
        relation = await base_context.get_relation(relation_id)
        assert relation is not None

    except Exception as e:
        # Rollback transaction on error
        await base_context.rollback_transaction()
        raise e

    # Test rollback behavior
    try:
        await base_context.begin_transaction()

        # Create an entity
        entity_id = await base_context.create_entity(person_type, person_props)
        assert entity_id is not None

        # Simulate an error condition
        raise ValueError("Simulated error")

    except ValueError:
        # Rollback should undo the entity creation
        await base_context.rollback_transaction()

        # Verify entity doesn't exist after rollback
        entity = await base_context.get_entity(entity_id)
        assert entity is None


@pytest.mark.asyncio
async def test_has_entity_type(base_context):
    """Test checking if an entity type exists."""
    # Register an entity type
    entity_type = EntityType(
        name="Person",
        properties={
            "name": PropertyDefinition(type="string", required=True),
            "age": PropertyDefinition(type="integer", required=False),
        },
    )
    await base_context.register_entity_type(entity_type)

    # Test that registered type exists
    assert await base_context.has_entity_type("Person") is True

    # Test that unregistered type does not exist
    assert await base_context.has_entity_type("InvalidType") is False


@pytest.mark.asyncio
async def test_has_relation_type(base_context):
    """Test checking if a relation type exists."""
    # Register entity type first (required for relation type)
    entity_type = EntityType(
        name="Person",
        properties={
            "name": PropertyDefinition(type="string", required=True),
        },
    )
    await base_context.register_entity_type(entity_type)

    # Register a relation type
    relation_type = RelationType(
        name="KNOWS",
        from_types=["Person"],
        to_types=["Person"],
        properties={
            "since": PropertyDefinition(type="integer", required=True),
        },
    )
    await base_context.register_relation_type(relation_type)

    # Test that registered type exists
    assert await base_context.has_relation_type("KNOWS") is True

    # Test that unregistered type does not exist
    assert await base_context.has_relation_type("InvalidType") is False
