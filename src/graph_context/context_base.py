"""
Base implementation for the graph-context module.

This module provides common functionality that can be used by specific graph
context implementations.
"""

from typing import Any

from .event_system import EventSystem, GraphEvent
from .exceptions import SchemaError
from .factory import GraphStoreFactory
from .interface import GraphContext
from .manager import EntityManager, QueryManager, RelationManager, TransactionManager
from .types.type_base import (
    Entity,
    EntityType,
    QuerySpec,
    Relation,
    RelationType,
    TraversalSpec,
)
from .validation import SchemaValidator


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
        self._events = EventSystem()
        self._validator = SchemaValidator(self._entity_types, self._relation_types)
        self._transaction = TransactionManager(self._store, self._events)
        self._entity_manager = EntityManager(
            self._store, self._events, self._validator, self._transaction
        )
        self._relation_manager = RelationManager(
            self._store, self._events, self._validator, self._transaction
        )
        self._query_manager = QueryManager(self._store, self._events)

    async def cleanup(self) -> None:
        """
        Clean up the graph context.

        This method should be called when the context is no longer needed.
        It cleans up internal state and type registries.

        Raises:
            GraphContextError: If cleanup fails
        """
        # Rollback any active transaction
        if self._transaction.is_in_transaction():
            await self._transaction.rollback_transaction()

        # Clear type registries
        self._entity_types.clear()
        self._relation_types.clear()

    async def has_entity_type(self, entity_type: str) -> bool:
        """
        Check if an entity type exists in the schema.

        Args:
            entity_type: Name of the entity type to check

        Returns:
            True if the entity type exists, False otherwise
        """
        return entity_type in self._entity_types

    async def has_relation_type(self, relation_type: str) -> bool:
        """
        Check if a relation type exists in the schema.

        Args:
            relation_type: Name of the relation type to check

        Returns:
            True if the relation type exists, False otherwise
        """
        return relation_type in self._relation_types

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
                schema_type=entity_type.name,
            )
        self._entity_types[entity_type.name] = entity_type
        await self._events.emit(
            GraphEvent.SCHEMA_MODIFIED,
            operation="register_entity_type",
            entity_type=entity_type.name,
        )
        await self._events.emit(
            GraphEvent.TYPE_MODIFIED, entity_type=entity_type.name, operation="register"
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
                schema_type=relation_type.name,
            )

        # Validate that referenced entity types exist
        for entity_type in relation_type.from_types:
            if entity_type not in self._entity_types:
                raise SchemaError(
                    f"Unknown entity type in from_types: {entity_type}",
                    schema_type=relation_type.name,
                    field="from_types",
                )

        for entity_type in relation_type.to_types:
            if entity_type not in self._entity_types:
                raise SchemaError(
                    f"Unknown entity type in to_types: {entity_type}",
                    schema_type=relation_type.name,
                    field="to_types",
                )

        self._relation_types[relation_type.name] = relation_type
        await self._events.emit(
            GraphEvent.SCHEMA_MODIFIED,
            operation="register_relation_type",
            relation_type=relation_type.name,
        )
        await self._events.emit(
            GraphEvent.TYPE_MODIFIED,
            relation_type=relation_type.name,
            operation="register",
        )

    def validate_entity(
        self, entity_type: str, properties: dict[str, Any]
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
        return self._validator.validate_entity(entity_type, properties)

    def validate_relation(
        self,
        relation_type: str,
        from_entity_type: str,
        to_entity_type: str,
        properties: dict[str, Any] | None = None,
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
        return self._validator.validate_relation(
            relation_type, from_entity_type, to_entity_type, properties
        )

    async def begin_transaction(self) -> None:
        """Begin a new transaction."""
        await self._transaction.begin_transaction()

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        await self._transaction.commit_transaction()

    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        await self._transaction.rollback_transaction()

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return await self._entity_manager.get(entity_id)

    async def create_entity(self, entity_type: str, properties: dict[str, Any]) -> str:
        """Create a new entity."""
        return await self._entity_manager.create(entity_type, properties)

    async def update_entity(self, entity_id: str, properties: dict[str, Any]) -> bool:
        """Update an existing entity."""
        return await self._entity_manager.update(entity_id, properties)

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        return await self._entity_manager.delete(entity_id)

    async def get_relation(self, relation_id: str) -> Relation | None:
        """Get a relation by ID."""
        return await self._relation_manager.get(relation_id)

    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create a new relation."""
        return await self._relation_manager.create(
            relation_type, from_entity, to_entity, properties
        )

    async def update_relation(
        self, relation_id: str, properties: dict[str, Any]
    ) -> bool:
        """Update an existing relation."""
        return await self._relation_manager.update(relation_id, properties)

    async def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation."""
        return await self._relation_manager.delete(relation_id)

    async def query(self, query_spec: QuerySpec) -> list[Entity]:
        """Execute a query against the graph."""
        return await self._query_manager.query(query_spec)

    async def traverse(
        self, start_entity: str, traversal_spec: TraversalSpec
    ) -> list[Entity]:
        """Traverse the graph starting from a given entity."""
        return await self._query_manager.traverse(start_entity, traversal_spec)
