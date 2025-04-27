"""
Schema validation for entities and relations.

This module provides the SchemaValidator class for validating entities and relations
against their type definitions in the schema.
"""

from typing import Any

from ..exceptions import SchemaError, ValidationError
from ..types.type_base import EntityType, RelationType
from ..types.validators import validate_property_value


class SchemaValidator:
    """
    Handles schema validation for entities and relations.

    This class encapsulates the validation logic to ensure entities and relations
    conform to their type definitions in the schema.
    """

    def __init__(
        self,
        entity_types: dict[str, EntityType],
        relation_types: dict[str, RelationType],
    ) -> None:
        """
        Initialize the validator with type registries.

        Args:
            entity_types: Dictionary of entity types
            relation_types: Dictionary of relation types
        """
        self._entity_types = entity_types
        self._relation_types = relation_types

    def validate_entity(self, entity_type: str, properties: dict[str, Any]) -> dict[str, Any]:
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
            raise SchemaError(f"Unknown entity type: {entity_type}", schema_type=entity_type)

        type_def = self._entity_types[entity_type]
        validated_props = {}

        # Check required properties
        for prop_name, prop_def in type_def.properties.items():
            if prop_name not in properties:
                if prop_def.required:
                    raise ValidationError(f"Missing required property: {prop_name}", field=prop_name)
                if prop_def.default is not None:
                    validated_props[prop_name] = prop_def.default
                continue

            # Validate property value
            try:
                validated_props[prop_name] = validate_property_value(properties[prop_name], prop_def)
            except ValidationError as e:
                raise ValidationError(str(e), field=prop_name) from e

        # Check for unknown properties
        for prop_name in properties:
            if prop_name not in type_def.properties:
                raise ValidationError(f"Unknown property: {prop_name}", field=prop_name)

        return validated_props

    def validate_relation(  # noqa: C901
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
        if relation_type not in self._relation_types:
            raise SchemaError(f"Unknown relation type: {relation_type}", schema_type=relation_type)

        type_def = self._relation_types[relation_type]

        # Validate entity types
        if from_entity_type not in type_def.from_types:
            raise ValidationError(
                f"Invalid from_entity_type: {from_entity_type}",
                field="from_entity_type",
            )

        if to_entity_type not in type_def.to_types:
            raise ValidationError(f"Invalid to_entity_type: {to_entity_type}", field="to_entity_type")

        # Validate properties if provided
        if properties is None:
            properties = {}

        validated_props = {}

        # Check required properties
        for prop_name, prop_def in type_def.properties.items():
            if prop_name not in properties:
                if prop_def.required:
                    raise ValidationError(f"Missing required property: {prop_name}", field=prop_name)
                if prop_def.default is not None:
                    validated_props[prop_name] = prop_def.default
                continue

            # Validate property value
            try:
                validated_props[prop_name] = validate_property_value(properties[prop_name], prop_def)
            except ValidationError as e:
                raise ValidationError(str(e), field=prop_name) from e

        # Check for unknown properties
        for prop_name in properties:
            if prop_name not in type_def.properties:
                raise ValidationError(f"Unknown property: {prop_name}", field=prop_name)

        return validated_props
