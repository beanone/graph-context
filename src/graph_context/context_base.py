"""
Base implementation for the graph-context module.

This module provides common functionality that can be used by specific graph
context implementations.
"""
from abc import abstractmethod
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


class BaseGraphContext(GraphContext):
    """
    Base implementation of the GraphContext interface.

    This class provides common functionality for validating entities and relations
    against their schema definitions. Specific implementations should inherit from
    this class and implement the abstract methods for their particular backend.
    """

    def __init__(self) -> None:
        """Initialize the base graph context."""
        self._entity_types: dict[str, EntityType] = {}
        self._relation_types: dict[str, RelationType] = {}
        self._in_transaction: bool = False
        self._events = EventSystem()

    @abstractmethod
    async def _get_entity_impl(self, entity_id: str) -> Entity | None:
        """Implementation method to get an entity."""
        pass

    @abstractmethod
    async def _create_entity_impl(self, entity_type: str, properties: dict[str, Any]) -> str:
        """Implementation method to create an entity."""
        pass

    @abstractmethod
    async def _update_entity_impl(self, entity_id: str, properties: dict[str, Any]) -> bool:
        """Implementation method to update an entity."""
        pass

    @abstractmethod
    async def _delete_entity_impl(self, entity_id: str) -> bool:
        """Implementation method to delete an entity."""
        pass

    @abstractmethod
    async def _get_relation_impl(self, relation_id: str) -> Relation | None:
        """Implementation method to get a relation."""
        pass

    @abstractmethod
    async def _create_relation_impl(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: dict[str, Any]
    ) -> str:
        """Implementation method to create a relation."""
        pass

    @abstractmethod
    async def _update_relation_impl(self, relation_id: str, properties: dict[str, Any]) -> bool:
        """Implementation method to update a relation."""
        pass

    @abstractmethod
    async def _delete_relation_impl(self, relation_id: str) -> bool:
        """Implementation method to delete a relation."""
        pass

    @abstractmethod
    async def _query_impl(self, query_spec: QuerySpec) -> list[Entity]:
        """Implementation method to execute a query."""
        pass

    @abstractmethod
    async def _traverse_impl(self, start_entity: str, traversal_spec: TraversalSpec) -> list[Entity]:
        """Implementation method to execute a traversal."""
        pass

    async def cleanup(self) -> None:
        """
        Clean up the graph context.

        This method should be called when the context is no longer needed.
        It cleans up internal state and type registries.

        Note: Specific implementations should override this method to add
        their own cleanup logic (e.g., closing connections, cleaning up
        backend resources) while still calling super().cleanup().

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
            entity_type: Type of the entity
            properties: Properties to validate

        Returns:
            Dictionary of validated property values

        Raises:
            SchemaError: If the entity type is not defined in the schema
            ValidationError: If property validation fails
        """
        type_def = self._entity_types.get(entity_type)
        if not type_def:
            raise SchemaError(
                f"Unknown entity type: {entity_type}",
                schema_type=entity_type
            )

        validated: dict[str, Any] = {}
        for name, prop_def in type_def.properties.items():
            if name in properties:
                validated[name] = validate_property_value(
                    properties[name],
                    prop_def
                )
            elif prop_def.required:
                raise ValidationError(
                    f"Required property missing: {name}",
                    field=name,
                    constraint="required"
                )
            elif prop_def.default is not None:
                validated[name] = prop_def.default

        # Check for unknown properties
        unknown = set(properties.keys()) - set(type_def.properties.keys())
        if unknown:
            raise ValidationError(
                f"Unknown properties: {', '.join(unknown)}",
                field=next(iter(unknown)),
                constraint="unknown"
            )

        return validated

    def validate_relation(
        self,
        relation_type: str,
        from_entity_type: str,
        to_entity_type: str,
        properties: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Validate relation properties and entity types against the schema.

        Args:
            relation_type: Type of the relation
            from_entity_type: Type of the source entity
            to_entity_type: Type of the target entity
            properties: Optional properties to validate

        Returns:
            Dictionary of validated property values

        Raises:
            SchemaError: If the relation type is not defined in the schema or if
                        the entity types are not compatible
            ValidationError: If property validation fails
        """
        type_def = self._relation_types.get(relation_type)
        if not type_def:
            raise SchemaError(
                f"Unknown relation type: {relation_type}",
                schema_type=relation_type
            )

        if from_entity_type not in type_def.from_types:
            raise SchemaError(
                f"Invalid source entity type for relation {relation_type}: "
                f"{from_entity_type}",
                schema_type=relation_type,
                field="from_types"
            )

        if to_entity_type not in type_def.to_types:
            raise SchemaError(
                f"Invalid target entity type for relation {relation_type}: "
                f"{to_entity_type}",
                schema_type=relation_type,
                field="to_types"
            )

        if not properties:
            return {}

        validated: dict[str, Any] = {}
        for name, prop_def in type_def.properties.items():
            if name in properties:
                validated[name] = validate_property_value(
                    properties[name],
                    prop_def
                )
            elif prop_def.required:
                raise ValidationError(
                    f"Required property missing: {name}",
                    field=name,
                    constraint="required"
                )
            elif prop_def.default is not None:
                validated[name] = prop_def.default

        # Check for unknown properties
        unknown = set(properties.keys()) - set(type_def.properties.keys())
        if unknown:
            raise ValidationError(
                f"Unknown properties: {', '.join(unknown)}",
                field=next(iter(unknown)),
                constraint="unknown"
            )

        return validated

    def _check_transaction(self, required: bool = True) -> None:
        """
        Check if a transaction is in progress.

        Args:
            required: If True, raises an error if no transaction is in progress.
                     If False, raises an error if a transaction is in progress.

        Raises:
            TransactionError: If the transaction state is not as required
        """
        if required and not self._in_transaction:
            raise TransactionError(
                "No transaction in progress",
                state="none"
            )
        elif not required and self._in_transaction:
            raise TransactionError(
                "Transaction already in progress",
                state="active"
            )

    async def begin_transaction(self) -> None:
        """Begin a new transaction."""
        if self._in_transaction:
            raise TransactionError("Transaction already in progress")
        self._in_transaction = True
        await self._events.emit(
            GraphEvent.TRANSACTION_BEGIN
        )

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")
        self._in_transaction = False
        await self._events.emit(
            GraphEvent.TRANSACTION_COMMIT
        )

    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")
        self._in_transaction = False
        await self._events.emit(
            GraphEvent.TRANSACTION_ROLLBACK
        )

    async def get_entity(self, entity_id: str) -> Entity | None:
        """
        Retrieve an entity by ID.

        Args:
            entity_id: ID of the entity to retrieve

        Returns:
            The entity if found, None otherwise

        Raises:
            GraphContextError: If the operation fails
        """
        result = await self._get_entity_impl(entity_id)
        if result:
            await self._events.emit(
                GraphEvent.ENTITY_READ,
                entity_id=entity_id,
                entity_type=result.get('type'),
                result=result
            )
        return result

    async def create_entity(self, entity_type: str, properties: dict[str, Any]) -> str:
        """
        Create a new entity in the graph.

        Args:
            entity_type: Type of the entity to create
            properties: Dictionary of property values

        Returns:
            The ID of the created entity

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the entity type is not defined in the schema
            GraphContextError: If the operation fails
        """
        validated = self.validate_entity(entity_type, properties)
        entity_id = await self._create_entity_impl(entity_type, validated)
        await self._events.emit(
            GraphEvent.ENTITY_WRITE,
            entity_id=entity_id,
            entity_type=entity_type,
            properties=validated
        )
        return entity_id

    async def update_entity(self, entity_id: str, properties: dict[str, Any]) -> bool:
        """
        Update an existing entity.

        Args:
            entity_id: ID of the entity to update
            properties: Dictionary of property values to update

        Returns:
            True if the entity was updated, False if not found

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the updated properties violate the schema
            GraphContextError: If the operation fails
        """
        # Get existing entity to validate type
        entity = await self._get_entity_impl(entity_id)
        if not entity:
            return False

        validated = self.validate_entity(entity['type'], properties)
        success = await self._update_entity_impl(entity_id, validated)
        if success:
            await self._events.emit(
                GraphEvent.ENTITY_WRITE,
                entity_id=entity_id,
                entity_type=entity['type'],
                properties=validated
            )
        return success

    async def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity from the graph.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if the entity was deleted, False if not found

        Raises:
            GraphContextError: If the operation fails
        """
        # Get entity type before deletion for event
        entity = await self._get_entity_impl(entity_id)
        if not entity:
            return False

        success = await self._delete_entity_impl(entity_id)
        if success:
            await self._events.emit(
                GraphEvent.ENTITY_DELETE,
                entity_id=entity_id,
                entity_type=entity['type']
            )
        return success

    async def get_relation(self, relation_id: str) -> Relation | None:
        """
        Retrieve a relation by ID.

        Args:
            relation_id: ID of the relation to retrieve

        Returns:
            The relation if found, None otherwise

        Raises:
            GraphContextError: If the operation fails
        """
        result = await self._get_relation_impl(relation_id)
        if result:
            await self._events.emit(
                GraphEvent.RELATION_READ,
                relation_id=relation_id,
                relation_type=result.get('type'),
                result=result
            )
        return result

    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: dict[str, Any] | None = None
    ) -> str:
        """
        Create a new relation between entities.

        Args:
            relation_type: Type of the relation to create
            from_entity: ID of the source entity
            to_entity: ID of the target entity
            properties: Optional dictionary of property values

        Returns:
            The ID of the created relation

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the relation type is not defined in the schema
            EntityNotFoundError: If either entity does not exist
            GraphContextError: If the operation fails
        """
        # Get entity types for validation
        from_entity_obj = await self._get_entity_impl(from_entity)
        to_entity_obj = await self._get_entity_impl(to_entity)
        if not from_entity_obj or not to_entity_obj:
            raise EntityNotFoundError("Source or target entity not found")

        validated = self.validate_relation(
            relation_type,
            from_entity_obj['type'],
            to_entity_obj['type'],
            properties or {}
        )

        relation_id = await self._create_relation_impl(
            relation_type,
            from_entity,
            to_entity,
            validated
        )

        await self._events.emit(
            GraphEvent.RELATION_WRITE,
            relation_id=relation_id,
            relation_type=relation_type,
            from_entity=from_entity,
            to_entity=to_entity,
            properties=validated
        )
        return relation_id

    async def update_relation(
        self,
        relation_id: str,
        properties: dict[str, Any]
    ) -> bool:
        """
        Update an existing relation.

        Args:
            relation_id: ID of the relation to update
            properties: Dictionary of property values to update

        Returns:
            True if the relation was updated, False if not found

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the updated properties violate the schema
            GraphContextError: If the operation fails
        """
        relation = await self._get_relation_impl(relation_id)
        if not relation:
            return False

        validated = self.validate_relation(
            relation['type'],
            relation['from_type'],
            relation['to_type'],
            properties
        )

        success = await self._update_relation_impl(relation_id, validated)
        if success:
            await self._events.emit(
                GraphEvent.RELATION_WRITE,
                relation_id=relation_id,
                relation_type=relation['type'],
                properties=validated
            )
        return success

    async def delete_relation(self, relation_id: str) -> bool:
        """
        Delete a relation from the graph.

        Args:
            relation_id: ID of the relation to delete

        Returns:
            True if the relation was deleted, False if not found

        Raises:
            GraphContextError: If the operation fails
        """
        relation = await self._get_relation_impl(relation_id)
        if not relation:
            return False

        success = await self._delete_relation_impl(relation_id)
        if success:
            await self._events.emit(
                GraphEvent.RELATION_DELETE,
                relation_id=relation_id,
                relation_type=relation['type']
            )
        return success

    async def query(self, query_spec: QuerySpec) -> list[Entity]:
        """
        Execute a query against the graph.

        Args:
            query_spec: Specification of the query to execute

        Returns:
            List of entities matching the query

        Raises:
            ValidationError: If the query specification is invalid
            GraphContextError: If the operation fails
        """
        results = await self._query_impl(query_spec)
        await self._events.emit(
            GraphEvent.QUERY_EXECUTED,
            query_spec=query_spec,
            results=results
        )
        return results

    async def traverse(
        self,
        start_entity: str,
        traversal_spec: TraversalSpec
    ) -> list[Entity]:
        """
        Traverse the graph starting from a given entity.

        Args:
            start_entity: ID of the entity to start traversal from
            traversal_spec: Specification of the traversal

        Returns:
            List of entities found during traversal

        Raises:
            EntityNotFoundError: If the start entity does not exist
            ValidationError: If the traversal specification is invalid
            GraphContextError: If the operation fails
        """
        results = await self._traverse_impl(start_entity, traversal_spec)
        await self._events.emit(
            GraphEvent.TRAVERSAL_EXECUTED,
            start_entity=start_entity,
            traversal_spec=traversal_spec,
            results=results
        )
        return results