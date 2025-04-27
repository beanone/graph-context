"""
Entity management for the graph-context module.

This module provides the EntityManager class for managing entity CRUD operations
in the graph context.
"""

from typing import Any

from ..event_system import EventSystem, GraphEvent
from ..exceptions import EntityNotFoundError
from ..interfaces.store import GraphStore
from ..types.type_base import Entity
from ..validation import SchemaValidator
from .transaction_manager import TransactionManager


class EntityManager:
    """
    Manages entity CRUD operations.

    This class encapsulates entity-related operations to provide a consistent
    interface for working with entities.
    """

    def __init__(
        self,
        store: GraphStore,
        events: EventSystem,
        validator: SchemaValidator,
        transaction: TransactionManager,
    ) -> None:
        """
        Initialize the entity manager.

        Args:
            store: The graph store to perform operations on
            events: Event system for emitting entity events
            validator: Schema validator for validating entities
            transaction: Transaction manager for transaction checks
        """
        self._store = store
        self._events = events
        self._validator = validator
        self._transaction = transaction

    async def get(self, entity_id: str) -> Entity | None:
        """
        Get an entity by ID.

        Args:
            entity_id: ID of the entity to retrieve

        Returns:
            The entity if found, None otherwise
        """
        entity = await self._store.get_entity(entity_id)

        if entity:
            await self._events.emit(GraphEvent.ENTITY_READ, entity_id=entity_id, entity_type=entity.type)

        return entity

    async def create(self, entity_type: str, properties: dict[str, Any]) -> str:
        """
        Create a new entity.

        Args:
            entity_type: Type of entity to create
            properties: Properties for the new entity

        Returns:
            ID of the created entity

        Raises:
            TransactionError: If no transaction is active
            ValidationError: If properties don't match schema
            SchemaError: If entity type is not registered
        """
        self._transaction.check_transaction()
        validated_props = self._validator.validate_entity(entity_type, properties)

        entity_id = await self._store.create_entity(entity_type, validated_props)

        await self._events.emit(GraphEvent.ENTITY_WRITE, entity_id=entity_id, entity_type=entity_type)

        return entity_id

    async def update(self, entity_id: str, properties: dict[str, Any]) -> bool:
        """
        Update an existing entity.

        Args:
            entity_id: ID of the entity to update
            properties: New properties for the entity

        Returns:
            True if update was successful, False otherwise

        Raises:
            TransactionError: If no transaction is active
            EntityNotFoundError: If entity doesn't exist
            ValidationError: If properties don't match schema
        """
        self._transaction.check_transaction()

        # Get current entity to validate type
        entity = await self._store.get_entity(entity_id)
        if not entity:
            raise EntityNotFoundError(f"Entity not found: {entity_id}")

        validated_props = self._validator.validate_entity(entity.type, properties)

        success = await self._store.update_entity(entity_id, validated_props)

        if success:
            await self._events.emit(GraphEvent.ENTITY_WRITE, entity_id=entity_id, entity_type=entity.type)

        return success

    async def delete(self, entity_id: str) -> bool:
        """
        Delete an entity.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            TransactionError: If no transaction is active
        """
        self._transaction.check_transaction()

        # Get current entity for event
        entity = await self._store.get_entity(entity_id)
        if not entity:
            return False

        success = await self._store.delete_entity(entity_id)

        if success:
            await self._events.emit(GraphEvent.ENTITY_DELETE, entity_id=entity_id, entity_type=entity.type)

        return success
