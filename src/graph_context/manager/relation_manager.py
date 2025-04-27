"""
Relation management for the graph-context module.

This module provides the RelationManager class for managing relation CRUD operations
in the graph context.
"""

from typing import Any

from ..event_system import EventSystem, GraphEvent
from ..exceptions import EntityNotFoundError
from ..interfaces.store import GraphStore
from ..types.type_base import Relation
from ..validation import SchemaValidator
from .transaction_manager import TransactionManager


class RelationManager:
    """
    Manages relation CRUD operations.

    This class encapsulates relation-related operations to provide a consistent
    interface for working with relations.
    """

    def __init__(
        self,
        store: GraphStore,
        events: EventSystem,
        validator: SchemaValidator,
        transaction: TransactionManager,
    ) -> None:
        """
        Initialize the relation manager.

        Args:
            store: The graph store to perform operations on
            events: Event system for emitting relation events
            validator: Schema validator for validating relations
            transaction: Transaction manager for transaction checks
        """
        self._store = store
        self._events = events
        self._validator = validator
        self._transaction = transaction

    async def get(self, relation_id: str) -> Relation | None:
        """
        Get a relation by ID.

        Args:
            relation_id: ID of the relation to retrieve

        Returns:
            The relation if found, None otherwise
        """
        relation = await self._store.get_relation(relation_id)

        if relation:
            await self._events.emit(
                GraphEvent.RELATION_READ,
                relation_id=relation_id,
                relation_type=relation.type,
            )

        return relation

    async def create(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a new relation.

        Args:
            relation_type: Type of relation to create
            from_entity: ID of the source entity
            to_entity: ID of the target entity
            properties: Properties for the new relation (optional)

        Returns:
            ID of the created relation

        Raises:
            TransactionError: If no transaction is active
            EntityNotFoundError: If either entity doesn't exist
            ValidationError: If properties don't match schema
            SchemaError: If relation type is not registered
        """
        self._transaction.check_transaction()

        # Get entity types for validation
        from_entity_obj = await self._store.get_entity(from_entity)
        if not from_entity_obj:
            raise EntityNotFoundError(f"From entity not found: {from_entity}")

        to_entity_obj = await self._store.get_entity(to_entity)
        if not to_entity_obj:
            raise EntityNotFoundError(f"To entity not found: {to_entity}")

        validated_props = self._validator.validate_relation(
            relation_type, from_entity_obj.type, to_entity_obj.type, properties or {}
        )

        relation_id = await self._store.create_relation(
            relation_type, from_entity, to_entity, validated_props
        )

        await self._events.emit(
            GraphEvent.RELATION_WRITE,
            relation_id=relation_id,
            relation_type=relation_type,
            from_entity=from_entity,
            to_entity=to_entity,
        )

        return relation_id

    async def update(self, relation_id: str, properties: dict[str, Any]) -> bool:
        """
        Update an existing relation.

        Args:
            relation_id: ID of the relation to update
            properties: New properties for the relation

        Returns:
            True if update was successful, False otherwise

        Raises:
            TransactionError: If no transaction is active
            EntityNotFoundError: If either entity no longer exists
            ValidationError: If properties don't match schema
        """
        self._transaction.check_transaction()

        # Get current relation to validate type
        relation = await self._store.get_relation(relation_id)
        if not relation:
            return False

        # Get entity types for validation
        from_entity = await self._store.get_entity(relation.from_entity)
        if not from_entity:
            raise EntityNotFoundError(f"From entity not found: {relation.from_entity}")

        to_entity = await self._store.get_entity(relation.to_entity)
        if not to_entity:
            raise EntityNotFoundError(f"To entity not found: {relation.to_entity}")

        validated_props = self._validator.validate_relation(
            relation.type, from_entity.type, to_entity.type, properties
        )

        success = await self._store.update_relation(relation_id, validated_props)

        if success:
            await self._events.emit(
                GraphEvent.RELATION_WRITE,
                relation_id=relation_id,
                relation_type=relation.type,
            )

        return success

    async def delete(self, relation_id: str) -> bool:
        """
        Delete a relation.

        Args:
            relation_id: ID of the relation to delete

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            TransactionError: If no transaction is active
        """
        self._transaction.check_transaction()

        # Get current relation for event
        relation = await self._store.get_relation(relation_id)
        if not relation:
            return False

        success = await self._store.delete_relation(relation_id)

        if success:
            await self._events.emit(
                GraphEvent.RELATION_DELETE,
                relation_id=relation_id,
                relation_type=relation.type,
            )

        return success
