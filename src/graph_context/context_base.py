"""
Base implementation for the graph-context module.

This module provides common functionality that can be used by specific graph
context implementations.
"""
from typing import Any

from .exceptions import (
    SchemaError,
    TransactionError,
    ValidationError,
    EntityNotFoundError,
    GraphContextError
)
from .interface import GraphContext
from .types.type_base import EntityType, RelationType, Entity, Relation, QuerySpec, TraversalSpec
from .types.validators import validate_property_value
from .event_system import EventSystem, GraphEvent
from .store import GraphStoreFactory


class BaseGraphContext(GraphContext):
    """
    Base implementation of the GraphContext interface.

    This class provides common functionality for validating entities and relations
    against their schema definitions, while delegating actual storage operations
    to a configured GraphStore implementation.
    """

    def __init__(self) -> None:
        """Initialize the base graph context."""
        self._store = GraphStoreFactory.create()
        self._entity_types: dict[str, EntityType] = {}
        self._relation_types: dict[str, RelationType] = {}
        self._in_transaction: bool = False
        self._events = EventSystem()

    async def cleanup(self) -> None:
        """
        Clean up the graph context.

        This method should be called when the context is no longer needed.
        It cleans up internal state and type registries.

        Raises:
            GraphContextError: If cleanup fails
        """
        # Rollback any active transaction
        if self._in_transaction:
            await self.rollback_transaction()

        # Clear type registries
        self._entity_types.clear()
        self._relation_types.clear()

        # Reset transaction state
        self._in_transaction = False

    async def register_entity_type(self, entity_type: EntityType) -> None:
        """
        Register an entity type in the schema.

        Args:
            entity_type: Entity type to register

        Raises:
            SchemaError: If an entity type with the same name already exists
        """
        if entity_type.name in self._entity_types:
            raise SchemaError(
                f"Entity type already exists: {entity_type.name}",
                schema_type=entity_type.name
            )
        self._entity_types[entity_type.name] = entity_type
        await self._events.emit(
            GraphEvent.SCHEMA_MODIFIED,
            operation="register_entity_type",
            entity_type=entity_type.name
        )
        await self._events.emit(
            GraphEvent.TYPE_MODIFIED,
            entity_type=entity_type.name,
            operation="register"
        )

    async def register_relation_type(self, relation_type: RelationType) -> None:
        """
        Register a relation type in the schema.

        Args:
            relation_type: Relation type to register

        Raises:
            SchemaError: If a relation type with the same name already exists or
                        if any of the referenced entity types do not exist
        """
        if relation_type.name in self._relation_types:
            raise SchemaError(
                f"Relation type already exists: {relation_type.name}",
                schema_type=relation_type.name
            )

        # Validate that referenced entity types exist
        for entity_type in relation_type.from_types:
            if entity_type not in self._entity_types:
                raise SchemaError(
                    f"Unknown entity type in from_types: {entity_type}",
                    schema_type=relation_type.name,
                    field="from_types"
                )

        for entity_type in relation_type.to_types:
            if entity_type not in self._entity_types:
                raise SchemaError(
                    f"Unknown entity type in to_types: {entity_type}",
                    schema_type=relation_type.name,
                    field="to_types"
                )

        self._relation_types[relation_type.name] = relation_type
        await self._events.emit(
            GraphEvent.SCHEMA_MODIFIED,
            operation="register_relation_type",
            relation_type=relation_type.name
        )
        await self._events.emit(
            GraphEvent.TYPE_MODIFIED,
            relation_type=relation_type.name,
            operation="register"
        )

    def validate_entity(
        self,
        entity_type: str,
        properties: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate entity properties against the schema.

        Args:
            entity_type: Type of entity to validate
            properties: Properties to validate

        Returns:
            Validated and potentially coerced properties

        Raises:
            ValidationError: If properties do not match schema
            SchemaError: If entity type is not registered
        """
        if entity_type not in self._entity_types:
            raise SchemaError(
                f"Unknown entity type: {entity_type}",
                schema_type=entity_type
            )

        type_def = self._entity_types[entity_type]
        validated_props = {}

        # Check required properties
        for prop_name, prop_def in type_def.properties.items():
            if prop_name not in properties:
                if prop_def.required:
                    raise ValidationError(
                        f"Missing required property: {prop_name}",
                        field=prop_name
                    )
                if prop_def.default is not None:
                    validated_props[prop_name] = prop_def.default
                continue

            # Validate property value
            try:
                validated_props[prop_name] = validate_property_value(
                    properties[prop_name],
                    prop_def
                )
            except ValidationError as e:
                raise ValidationError(
                    str(e),
                    field=prop_name
                ) from e

        # Check for unknown properties
        for prop_name in properties:
            if prop_name not in type_def.properties:
                raise ValidationError(
                    f"Unknown property: {prop_name}",
                    field=prop_name
                )

        return validated_props

    def validate_relation(
        self,
        relation_type: str,
        from_entity_type: str,
        to_entity_type: str,
        properties: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Validate relation properties and types against the schema.

        Args:
            relation_type: Type of relation to validate
            from_entity_type: Type of source entity
            to_entity_type: Type of target entity
            properties: Properties to validate (optional)

        Returns:
            Validated and potentially coerced properties

        Raises:
            ValidationError: If properties do not match schema
            SchemaError: If relation type is not registered
        """
        if relation_type not in self._relation_types:
            raise SchemaError(
                f"Unknown relation type: {relation_type}",
                schema_type=relation_type
            )

        type_def = self._relation_types[relation_type]

        # Validate entity types
        if from_entity_type not in type_def.from_types:
            raise ValidationError(
                f"Invalid from_entity_type: {from_entity_type}",
                field="from_entity_type"
            )

        if to_entity_type not in type_def.to_types:
            raise ValidationError(
                f"Invalid to_entity_type: {to_entity_type}",
                field="to_entity_type"
            )

        # Validate properties if provided
        if properties is None:
            properties = {}

        validated_props = {}

        # Check required properties
        for prop_name, prop_def in type_def.properties.items():
            if prop_name not in properties:
                if prop_def.required:
                    raise ValidationError(
                        f"Missing required property: {prop_name}",
                        field=prop_name
                    )
                if prop_def.default is not None:
                    validated_props[prop_name] = prop_def.default
                continue

            # Validate property value
            try:
                validated_props[prop_name] = validate_property_value(
                    properties[prop_name],
                    prop_def
                )
            except ValidationError as e:
                raise ValidationError(
                    str(e),
                    field=prop_name
                ) from e

        # Check for unknown properties
        for prop_name in properties:
            if prop_name not in type_def.properties:
                raise ValidationError(
                    f"Unknown property: {prop_name}",
                    field=prop_name
                )

        return validated_props

    def _check_transaction(self, required: bool = True) -> None:
        """
        Check transaction state.

        Args:
            required: Whether a transaction is required

        Raises:
            TransactionError: If transaction state does not match requirement
        """
        if required and not self._in_transaction:
            raise TransactionError("Operation requires an active transaction")
        elif not required and self._in_transaction:
            raise TransactionError("Operation cannot be performed in a transaction")

    async def begin_transaction(self) -> None:
        """Begin a new transaction."""
        if self._in_transaction:
            raise TransactionError("Transaction already in progress")

        await self._store.begin_transaction()
        self._in_transaction = True

        await self._events.emit(GraphEvent.TRANSACTION_BEGIN)

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")

        await self._store.commit_transaction()
        self._in_transaction = False

        await self._events.emit(GraphEvent.TRANSACTION_COMMIT)

    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")

        await self._store.rollback_transaction()
        self._in_transaction = False

        await self._events.emit(GraphEvent.TRANSACTION_ROLLBACK)

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        entity = await self._store.get_entity(entity_id)

        if entity:
            await self._events.emit(
                GraphEvent.ENTITY_READ,
                entity_id=entity_id,
                entity_type=entity.type
            )

        return entity

    async def create_entity(self, entity_type: str, properties: dict[str, Any]) -> str:
        """Create a new entity."""
        self._check_transaction()
        validated_props = self.validate_entity(entity_type, properties)

        entity_id = await self._store.create_entity(entity_type, validated_props)

        await self._events.emit(
            GraphEvent.ENTITY_WRITE,
            entity_id=entity_id,
            entity_type=entity_type
        )

        return entity_id

    async def update_entity(self, entity_id: str, properties: dict[str, Any]) -> bool:
        """Update an existing entity."""
        self._check_transaction()

        # Get current entity to validate type
        entity = await self._store.get_entity(entity_id)
        if not entity:
            raise EntityNotFoundError(f"Entity not found: {entity_id}")

        validated_props = self.validate_entity(entity.type, properties)

        success = await self._store.update_entity(entity_id, validated_props)

        if success:
            await self._events.emit(
                GraphEvent.ENTITY_WRITE,
                entity_id=entity_id,
                entity_type=entity.type
            )

        return success

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        self._check_transaction()

        # Get current entity for event
        entity = await self._store.get_entity(entity_id)
        if not entity:
            return False

        success = await self._store.delete_entity(entity_id)

        if success:
            await self._events.emit(
                GraphEvent.ENTITY_DELETE,
                entity_id=entity_id,
                entity_type=entity.type
            )

        return success

    async def get_relation(self, relation_id: str) -> Relation | None:
        """Get a relation by ID."""
        relation = await self._store.get_relation(relation_id)

        if relation:
            await self._events.emit(
                GraphEvent.RELATION_READ,
                relation_id=relation_id,
                relation_type=relation.type
            )

        return relation

    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: dict[str, Any] | None = None
    ) -> str:
        """Create a new relation."""
        self._check_transaction()

        # Get entity types for validation
        from_entity_obj = await self._store.get_entity(from_entity)
        if not from_entity_obj:
            raise EntityNotFoundError(f"From entity not found: {from_entity}")

        to_entity_obj = await self._store.get_entity(to_entity)
        if not to_entity_obj:
            raise EntityNotFoundError(f"To entity not found: {to_entity}")

        validated_props = self.validate_relation(
            relation_type,
            from_entity_obj.type,
            to_entity_obj.type,
            properties or {}
        )

        relation_id = await self._store.create_relation(
            relation_type,
            from_entity,
            to_entity,
            validated_props
        )

        await self._events.emit(
            GraphEvent.RELATION_WRITE,
            relation_id=relation_id,
            relation_type=relation_type,
            from_entity=from_entity,
            to_entity=to_entity
        )

        return relation_id

    async def update_relation(
        self,
        relation_id: str,
        properties: dict[str, Any]
    ) -> bool:
        """Update an existing relation."""
        self._check_transaction()

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

        validated_props = self.validate_relation(
            relation.type,
            from_entity.type,
            to_entity.type,
            properties
        )

        success = await self._store.update_relation(relation_id, validated_props)

        if success:
            await self._events.emit(
                GraphEvent.RELATION_WRITE,
                relation_id=relation_id,
                relation_type=relation.type
            )

        return success

    async def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation."""
        self._check_transaction()

        # Get current relation for event
        relation = await self._store.get_relation(relation_id)
        if not relation:
            return False

        success = await self._store.delete_relation(relation_id)

        if success:
            await self._events.emit(
                GraphEvent.RELATION_DELETE,
                relation_id=relation_id,
                relation_type=relation.type
            )

        return success

    async def query(self, query_spec: QuerySpec) -> list[Entity]:
        """Execute a query against the graph."""
        results = await self._store.query(query_spec)

        await self._events.emit(
            GraphEvent.QUERY_EXECUTED,
            query_spec=query_spec
        )

        return results

    async def traverse(
        self,
        start_entity: str,
        traversal_spec: TraversalSpec
    ) -> list[Entity]:
        """Traverse the graph starting from a given entity."""
        results = await self._store.traverse(start_entity, traversal_spec)

        await self._events.emit(
            GraphEvent.TRAVERSAL_EXECUTED,
            start_entity=start_entity,
            traversal_spec=traversal_spec
        )

        return results