"""
Base implementation for the graph-context module.

This module provides common functionality that can be used by specific graph
context implementations.
"""
from typing import Any

from .exceptions import SchemaError, TransactionError, ValidationError
from .interface import GraphContext
from .types.type_base import EntityType, RelationType
from .types.validators import validate_property_value


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

    def register_entity_type(self, entity_type: EntityType) -> None:
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

    def register_relation_type(self, relation_type: RelationType) -> None:
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
        self._check_transaction(required=False)
        self._in_transaction = True

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        self._check_transaction(required=True)
        self._in_transaction = False

    async def rollback_transaction(self) -> None:
        """Roll back the current transaction."""
        self._check_transaction(required=True)
        self._in_transaction = False