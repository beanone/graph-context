"""
Tests for the EntityManager class.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from graph_context.context_base import (
    EntityManager,
    SchemaValidator,
    TransactionManager,
)
from graph_context.event_system import GraphEvent
from graph_context.exceptions import EntityNotFoundError


@pytest.fixture
def mock_store():
    """Mock GraphStore for testing."""
    store = AsyncMock()
    store.get_entity = AsyncMock()
    store.list_entities = AsyncMock()
    store.create_entity = AsyncMock()
    store.update_entity = AsyncMock()
    store.delete_entity = AsyncMock()
    store.get_entities_by_type = AsyncMock()
    store.search_entities = AsyncMock()
    return store


@pytest.fixture
def mock_validator():
    """Mock SchemaValidator for testing."""
    validator = AsyncMock(spec=SchemaValidator)
    validator.validate_entity = MagicMock()
    return validator


@pytest.fixture
def mock_transaction():
    """Mock TransactionManager for testing."""
    transaction = AsyncMock(spec=TransactionManager)
    transaction.check_transaction = MagicMock()
    return transaction


@pytest.fixture
def mock_events():
    """Mock EventSystem for testing."""
    events = AsyncMock()
    events.emit = AsyncMock()
    return events


@pytest.fixture
def entity_manager(mock_store, mock_events, mock_validator, mock_transaction):
    """EntityManager instance for testing."""
    return EntityManager(mock_store, mock_events, mock_validator, mock_transaction)


@pytest.fixture
def sample_entity():
    """Sample entity for testing."""
    entity = MagicMock()
    entity.id = "entity-123"
    entity.type = "person"
    entity.properties = {
        "name": "John Doe",
        "age": 30
    }
    return entity


@pytest.fixture
def updated_entity():
    """Updated entity for testing."""
    entity = MagicMock()
    entity.id = "entity-123"
    entity.type = "person"
    entity.properties = {
        "name": "Jane Doe",
        "age": 32
    }
    return entity


class TestEntityManager:
    """Test cases for EntityManager class."""

    def test_init(self, mock_store, mock_events, mock_validator, mock_transaction):
        """Test EntityManager initialization."""
        manager = EntityManager(mock_store, mock_events, mock_validator, mock_transaction)
        assert manager._store is mock_store
        assert manager._events is mock_events
        assert manager._validator is mock_validator
        assert manager._transaction is mock_transaction

    @pytest.mark.asyncio
    async def test_get_entity_found(self, entity_manager, mock_store, sample_entity):
        """Test get method when entity exists."""
        entity_id = "entity-123"
        mock_store.get_entity.return_value = sample_entity

        result = await entity_manager.get(entity_id)

        mock_store.get_entity.assert_called_once_with(entity_id)
        assert result == sample_entity

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, entity_manager, mock_store):
        """Test get method when entity doesn't exist."""
        entity_id = "nonexistent"
        mock_store.get_entity.return_value = None

        result = await entity_manager.get(entity_id)

        mock_store.get_entity.assert_called_once_with(entity_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_create_entity_success(self, entity_manager, mock_store, mock_validator,
                                          mock_transaction, mock_events, sample_entity):
        """Test create method successful execution."""
        entity_type = "person"
        entity_properties = {"name": "John Doe", "age": 30}
        validated_props = {"name": "John Doe", "age": 30}
        entity_id = "entity-123"

        mock_validator.validate_entity.return_value = validated_props
        mock_store.create_entity.return_value = entity_id

        result = await entity_manager.create(entity_type, entity_properties)

        mock_transaction.check_transaction.assert_called_once()
        mock_validator.validate_entity.assert_called_once_with(entity_type, entity_properties)
        mock_store.create_entity.assert_called_once_with(entity_type, validated_props)
        mock_events.emit.assert_called_once_with(
            GraphEvent.ENTITY_WRITE,
            entity_id=entity_id,
            entity_type=entity_type
        )
        assert result == entity_id

    @pytest.mark.asyncio
    async def test_update_entity_success(self, entity_manager, mock_store, mock_validator,
                                          mock_transaction, mock_events, sample_entity):
        """Test update method successful execution."""
        entity_id = "entity-123"
        entity_properties = {"name": "Jane Doe", "age": 32}
        validated_props = {"name": "Jane Doe", "age": 32}

        mock_store.get_entity.return_value = sample_entity
        mock_validator.validate_entity.return_value = validated_props
        mock_store.update_entity.return_value = True

        result = await entity_manager.update(entity_id, entity_properties)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_entity.assert_called_once_with(entity_id)
        mock_validator.validate_entity.assert_called_once_with(sample_entity.type, entity_properties)
        mock_store.update_entity.assert_called_once_with(entity_id, validated_props)
        mock_events.emit.assert_called_once_with(
            GraphEvent.ENTITY_WRITE,
            entity_id=entity_id,
            entity_type=sample_entity.type
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_update_entity_verifies_changes(self, entity_manager, mock_store, mock_validator,
                                          mock_transaction, mock_events, sample_entity, updated_entity):
        """Test update method actually changes the entity properties."""
        entity_id = "entity-123"
        original_properties = {"name": "John Doe", "age": 30}
        new_properties = {"name": "Jane Doe", "age": 32}
        validated_props = {"name": "Jane Doe", "age": 32}

        # Setup initial entity
        mock_store.get_entity.side_effect = [sample_entity, updated_entity]
        mock_validator.validate_entity.return_value = validated_props
        mock_store.update_entity.return_value = True

        # Perform update
        result = await entity_manager.update(entity_id, new_properties)
        assert result is True

        # Verify the entity is updated by fetching it again
        updated = await entity_manager.get(entity_id)
        assert updated.properties["name"] == "Jane Doe"
        assert updated.properties["age"] == 32
        assert mock_store.get_entity.call_count == 2

    @pytest.mark.asyncio
    async def test_update_entity_failed_operation(self, entity_manager, mock_store, mock_validator,
                                          mock_transaction, mock_events, sample_entity):
        """Test update method when update operation fails."""
        entity_id = "entity-123"
        entity_properties = {"name": "Jane Doe", "age": 32}
        validated_props = {"name": "Jane Doe", "age": 32}

        mock_store.get_entity.return_value = sample_entity
        mock_validator.validate_entity.return_value = validated_props
        # Simulate failed update operation
        mock_store.update_entity.return_value = False

        result = await entity_manager.update(entity_id, entity_properties)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_entity.assert_called_once_with(entity_id)
        mock_validator.validate_entity.assert_called_once_with(sample_entity.type, entity_properties)
        mock_store.update_entity.assert_called_once_with(entity_id, validated_props)
        # Event should not be emitted on failed update
        mock_events.emit.assert_not_called()
        assert result is False

    @pytest.mark.asyncio
    async def test_update_entity_not_found(self, entity_manager, mock_store, mock_transaction, mock_validator, mock_events):
        """Test update method when entity doesn't exist."""
        entity_id = "nonexistent"
        entity_properties = {"name": "Jane Doe"}
        mock_store.get_entity.return_value = None

        with pytest.raises(EntityNotFoundError, match=f"Entity not found: {entity_id}"):
            await entity_manager.update(entity_id, entity_properties)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_entity.assert_called_once_with(entity_id)
        mock_validator.validate_entity.assert_not_called()
        mock_store.update_entity.assert_not_called()
        mock_events.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_entity_success(self, entity_manager, mock_store, mock_transaction,
                                          mock_events, sample_entity):
        """Test delete method successful execution."""
        entity_id = "entity-123"
        mock_store.get_entity.return_value = sample_entity
        mock_store.delete_entity.return_value = True

        result = await entity_manager.delete(entity_id)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_entity.assert_called_once_with(entity_id)
        mock_store.delete_entity.assert_called_once_with(entity_id)
        mock_events.emit.assert_called_once_with(
            GraphEvent.ENTITY_DELETE,
            entity_id=entity_id,
            entity_type=sample_entity.type
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_entity_verify_deleted(self, entity_manager, mock_store, mock_transaction,
                                               mock_events, sample_entity):
        """Test delete method actually removes the entity."""
        entity_id = "entity-123"

        # First call returns the entity, second call (after delete) returns None
        mock_store.get_entity.side_effect = [sample_entity, None]
        mock_store.delete_entity.return_value = True

        # Delete the entity
        result = await entity_manager.delete(entity_id)
        assert result is True

        # Verify the entity was deleted by trying to fetch it again
        deleted_entity = await entity_manager.get(entity_id)
        assert deleted_entity is None
        assert mock_store.get_entity.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_entity_failed_operation(self, entity_manager, mock_store, mock_transaction,
                                                 mock_events, sample_entity):
        """Test delete method when delete operation fails."""
        entity_id = "entity-123"
        mock_store.get_entity.return_value = sample_entity
        # Simulate failed delete operation
        mock_store.delete_entity.return_value = False

        result = await entity_manager.delete(entity_id)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_entity.assert_called_once_with(entity_id)
        mock_store.delete_entity.assert_called_once_with(entity_id)
        # Event should not be emitted on failed delete
        mock_events.emit.assert_not_called()
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_entity_not_found(self, entity_manager, mock_store, mock_transaction, mock_events):
        """Test delete method when entity doesn't exist."""
        entity_id = "nonexistent"
        mock_store.get_entity.return_value = None

        result = await entity_manager.delete(entity_id)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_entity.assert_called_once_with(entity_id)
        mock_store.delete_entity.assert_not_called()
        mock_events.emit.assert_not_called()
        assert result is False