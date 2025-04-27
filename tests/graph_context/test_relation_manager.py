"""
Tests for the RelationManager class.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from graph_context.event_system import GraphEvent
from graph_context.exceptions import EntityNotFoundError
from graph_context.manager import RelationManager, TransactionManager
from graph_context.types.type_base import Entity, Relation
from graph_context.validation import SchemaValidator


@pytest.fixture
def mock_store():
    """Mock GraphStore for testing."""
    store = AsyncMock()
    store.get_relation = AsyncMock()
    store.get_entity = AsyncMock()
    store.create_relation = AsyncMock()
    store.update_relation = AsyncMock()
    store.delete_relation = AsyncMock()
    store.query_relations = AsyncMock()
    return store


@pytest.fixture
def mock_validator():
    """Mock SchemaValidator for testing."""
    validator = AsyncMock(spec=SchemaValidator)
    validator.validate_relation = MagicMock()
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
def relation_manager(mock_store, mock_events, mock_validator, mock_transaction):
    """RelationManager instance for testing."""
    return RelationManager(mock_store, mock_events, mock_validator, mock_transaction)


@pytest.fixture
def sample_from_entity():
    """Sample from entity for testing."""
    entity = MagicMock(spec=Entity)
    entity.id = "entity-123"
    entity.type = "person"
    return entity


@pytest.fixture
def sample_to_entity():
    """Sample to entity for testing."""
    entity = MagicMock(spec=Entity)
    entity.id = "entity-456"
    entity.type = "document"
    return entity


@pytest.fixture
def sample_relation():
    """Sample relation for testing."""
    relation = MagicMock(spec=Relation)
    relation.id = "relation-123"
    relation.type = "authored"
    relation.from_entity = "entity-123"
    relation.to_entity = "entity-456"
    relation.properties = {"year": 2023, "is_primary_author": True}
    return relation


@pytest.fixture
def updated_relation():
    """Updated relation for testing."""
    relation = MagicMock(spec=Relation)
    relation.id = "relation-123"
    relation.type = "authored"
    relation.from_entity = "entity-123"
    relation.to_entity = "entity-456"
    relation.properties = {"year": 2024, "is_primary_author": False}
    return relation


class TestRelationManager:
    """Test cases for RelationManager class."""

    def test_init(self, mock_store, mock_events, mock_validator, mock_transaction):
        """Test RelationManager initialization."""
        manager = RelationManager(mock_store, mock_events, mock_validator, mock_transaction)
        assert manager._store is mock_store
        assert manager._events is mock_events
        assert manager._validator is mock_validator
        assert manager._transaction is mock_transaction

    @pytest.mark.asyncio
    async def test_get_relation_found(self, relation_manager, mock_store, sample_relation):
        """Test get method when relation exists."""
        relation_id = "relation-123"
        mock_store.get_relation.return_value = sample_relation

        result = await relation_manager.get(relation_id)

        mock_store.get_relation.assert_called_once_with(relation_id)
        assert result is sample_relation

    @pytest.mark.asyncio
    async def test_get_relation_not_found(self, relation_manager, mock_store):
        """Test get method when relation doesn't exist."""
        relation_id = "nonexistent"
        mock_store.get_relation.return_value = None

        result = await relation_manager.get(relation_id)

        mock_store.get_relation.assert_called_once_with(relation_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_create_relation_success(
        self,
        relation_manager,
        mock_store,
        mock_validator,
        mock_transaction,
        mock_events,
        sample_from_entity,
        sample_to_entity,
    ):
        """Test create method successful execution."""
        relation_type = "authored"
        from_entity = "entity-123"
        to_entity = "entity-456"
        properties = {"year": 2023}
        validated_props = {"year": 2023, "is_primary_author": True}
        relation_id = "relation-123"

        mock_store.get_entity.side_effect = [sample_from_entity, sample_to_entity]
        mock_validator.validate_relation.return_value = validated_props
        mock_store.create_relation.return_value = relation_id

        result = await relation_manager.create(relation_type, from_entity, to_entity, properties)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_entity.assert_any_call(from_entity)
        mock_store.get_entity.assert_any_call(to_entity)
        mock_validator.validate_relation.assert_called_once_with(
            relation_type, sample_from_entity.type, sample_to_entity.type, properties
        )
        mock_store.create_relation.assert_called_once_with(relation_type, from_entity, to_entity, validated_props)
        mock_events.emit.assert_called_once_with(
            GraphEvent.RELATION_WRITE,
            relation_id=relation_id,
            relation_type=relation_type,
            from_entity=from_entity,
            to_entity=to_entity,
        )
        assert result == relation_id

    @pytest.mark.asyncio
    async def test_create_relation_from_entity_not_found(
        self,
        relation_manager,
        mock_store,
        mock_transaction,
        mock_validator,
        mock_events,
    ):
        """Test create method when from_entity doesn't exist."""
        relation_type = "authored"
        from_entity = "nonexistent"
        to_entity = "entity-456"
        properties = {"year": 2023}

        mock_store.get_entity.return_value = None

        with pytest.raises(EntityNotFoundError, match=f"From entity not found: {from_entity}"):
            await relation_manager.create(relation_type, from_entity, to_entity, properties)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_entity.assert_called_once_with(from_entity)
        mock_validator.validate_relation.assert_not_called()
        mock_store.create_relation.assert_not_called()
        mock_events.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_relation_to_entity_not_found(
        self,
        relation_manager,
        mock_store,
        mock_transaction,
        sample_from_entity,
        mock_validator,
        mock_events,
    ):
        """Test create method when to_entity doesn't exist."""
        relation_type = "authored"
        from_entity = "entity-123"
        to_entity = "nonexistent"
        properties = {"year": 2023}

        mock_store.get_entity.side_effect = [sample_from_entity, None]

        with pytest.raises(EntityNotFoundError, match=f"To entity not found: {to_entity}"):
            await relation_manager.create(relation_type, from_entity, to_entity, properties)

        mock_transaction.check_transaction.assert_called_once()
        assert mock_store.get_entity.call_count == 2
        mock_validator.validate_relation.assert_not_called()
        mock_store.create_relation.assert_not_called()
        mock_events.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_relation_success(
        self,
        relation_manager,
        mock_store,
        mock_validator,
        mock_transaction,
        mock_events,
        sample_relation,
        sample_from_entity,
        sample_to_entity,
    ):
        """Test update method successful execution."""
        relation_id = "relation-123"
        properties = {"year": 2024, "is_primary_author": False}
        validated_props = {"year": 2024, "is_primary_author": False}

        mock_store.get_relation.return_value = sample_relation
        mock_store.get_entity.side_effect = [sample_from_entity, sample_to_entity]
        mock_validator.validate_relation.return_value = validated_props
        mock_store.update_relation.return_value = True

        result = await relation_manager.update(relation_id, properties)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_relation.assert_called_once_with(relation_id)
        mock_store.get_entity.assert_any_call(sample_relation.from_entity)
        mock_store.get_entity.assert_any_call(sample_relation.to_entity)
        mock_validator.validate_relation.assert_called_once_with(
            sample_relation.type,
            sample_from_entity.type,
            sample_to_entity.type,
            properties,
        )
        mock_store.update_relation.assert_called_once_with(relation_id, validated_props)
        mock_events.emit.assert_called_once_with(
            GraphEvent.RELATION_WRITE,
            relation_id=relation_id,
            relation_type=sample_relation.type,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_update_relation_verifies_changes(
        self,
        relation_manager,
        mock_store,
        mock_validator,
        mock_transaction,
        mock_events,
        sample_relation,
        updated_relation,
        sample_from_entity,
        sample_to_entity,
    ):
        """Test update method actually changes the relation properties."""
        relation_id = "relation-123"
        properties = {"year": 2024, "is_primary_author": False}
        validated_props = {"year": 2024, "is_primary_author": False}

        # Setup initial and updated relations
        mock_store.get_relation.side_effect = [sample_relation, updated_relation]
        mock_store.get_entity.side_effect = [sample_from_entity, sample_to_entity]
        mock_validator.validate_relation.return_value = validated_props
        mock_store.update_relation.return_value = True

        # Perform update
        result = await relation_manager.update(relation_id, properties)
        assert result is True

        # Verify the relation is updated by fetching it again
        updated = await relation_manager.get(relation_id)
        assert updated.properties["year"] == 2024
        assert updated.properties["is_primary_author"] is False
        assert mock_store.get_relation.call_count == 2

    @pytest.mark.asyncio
    async def test_update_relation_failed_operation(
        self,
        relation_manager,
        mock_store,
        mock_validator,
        mock_transaction,
        mock_events,
        sample_relation,
        sample_from_entity,
        sample_to_entity,
    ):
        """Test update method when update operation fails."""
        relation_id = "relation-123"
        properties = {"year": 2024, "is_primary_author": False}
        validated_props = {"year": 2024, "is_primary_author": False}

        mock_store.get_relation.return_value = sample_relation
        mock_store.get_entity.side_effect = [sample_from_entity, sample_to_entity]
        mock_validator.validate_relation.return_value = validated_props
        # Simulate failed update operation
        mock_store.update_relation.return_value = False

        result = await relation_manager.update(relation_id, properties)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_relation.assert_called_once_with(relation_id)
        mock_validator.validate_relation.assert_called_once_with(
            sample_relation.type,
            sample_from_entity.type,
            sample_to_entity.type,
            properties,
        )
        mock_store.update_relation.assert_called_once_with(relation_id, validated_props)
        # Event should not be emitted on failed update
        mock_events.emit.assert_not_called()
        assert result is False

    @pytest.mark.asyncio
    async def test_update_relation_not_found(
        self,
        relation_manager,
        mock_store,
        mock_transaction,
        mock_validator,
        mock_events,
    ):
        """Test update method when relation doesn't exist."""
        relation_id = "nonexistent"
        properties = {"year": 2024}

        mock_store.get_relation.return_value = None

        result = await relation_manager.update(relation_id, properties)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_relation.assert_called_once_with(relation_id)
        mock_validator.validate_relation.assert_not_called()
        mock_store.update_relation.assert_not_called()
        mock_events.emit.assert_not_called()
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_relation_success(
        self,
        relation_manager,
        mock_store,
        mock_transaction,
        mock_events,
        sample_relation,
    ):
        """Test delete method successful execution."""
        relation_id = "relation-123"
        mock_store.get_relation.return_value = sample_relation
        mock_store.delete_relation.return_value = True

        result = await relation_manager.delete(relation_id)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_relation.assert_called_once_with(relation_id)
        mock_store.delete_relation.assert_called_once_with(relation_id)
        mock_events.emit.assert_called_once_with(
            GraphEvent.RELATION_DELETE,
            relation_id=relation_id,
            relation_type=sample_relation.type,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_relation_verify_deleted(
        self,
        relation_manager,
        mock_store,
        mock_transaction,
        mock_events,
        sample_relation,
    ):
        """Test delete method actually removes the relation."""
        relation_id = "relation-123"

        # First call returns the relation, second call (after delete) returns None
        mock_store.get_relation.side_effect = [sample_relation, None]
        mock_store.delete_relation.return_value = True

        # Delete the relation
        result = await relation_manager.delete(relation_id)
        assert result is True

        # Verify the relation was deleted by trying to fetch it again
        deleted_relation = await relation_manager.get(relation_id)
        assert deleted_relation is None
        assert mock_store.get_relation.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_relation_failed_operation(
        self,
        relation_manager,
        mock_store,
        mock_transaction,
        mock_events,
        sample_relation,
    ):
        """Test delete method when delete operation fails."""
        relation_id = "relation-123"
        mock_store.get_relation.return_value = sample_relation
        # Simulate failed delete operation
        mock_store.delete_relation.return_value = False

        result = await relation_manager.delete(relation_id)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_relation.assert_called_once_with(relation_id)
        mock_store.delete_relation.assert_called_once_with(relation_id)
        # Event should not be emitted on failed delete
        mock_events.emit.assert_not_called()
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_relation_not_found(self, relation_manager, mock_store, mock_transaction, mock_events):
        """Test delete method when relation doesn't exist."""
        relation_id = "nonexistent"
        mock_store.get_relation.return_value = None

        result = await relation_manager.delete(relation_id)

        mock_transaction.check_transaction.assert_called_once()
        mock_store.get_relation.assert_called_once_with(relation_id)
        mock_store.delete_relation.assert_not_called()
        mock_events.emit.assert_not_called()
        assert result is False
