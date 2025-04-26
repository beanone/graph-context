"""
Type validation logic for the graph-context module.

This module provides validation functions for checking property values against
their defined types and constraints.
"""
import math
from datetime import datetime
from typing import Any, Optional, Union
from uuid import UUID

from ..exceptions import ValidationError
from .type_base import PropertyDefinition, PropertyType

# Add the validation map before the validate_property_value function
_PROPERTY_VALIDATORS = {
    PropertyType.STRING: lambda v, c: validate_string(v, c),
    PropertyType.INTEGER: lambda v, c: validate_number(v, PropertyType.INTEGER, c),
    PropertyType.FLOAT: lambda v, c: validate_number(v, PropertyType.FLOAT, c),
    PropertyType.BOOLEAN: lambda v, c: validate_boolean(v),
    PropertyType.DATETIME: lambda v, c: validate_datetime(v, c),
    PropertyType.UUID: lambda v, c: validate_uuid(v),
    PropertyType.LIST: lambda v, c: validate_list(v, c),
    PropertyType.DICT: lambda v, c: validate_dict(v, c),
}


def validate_string(value: Any, constraints: Optional[dict[str, Any]] = None) -> str:
    """
    Validate a string value against its constraints.

    Args:
        value: Value to validate
        constraints: Optional dictionary of constraints

    Returns:
        The validated string value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError("Value must be a string", value=value, constraint="type")

    if constraints:
        if "min_length" in constraints and len(value) < constraints["min_length"]:
            raise ValidationError(
                f"min_length constraint: String length " f"must be at least {constraints['min_length']}",
                value=value,
                constraint="min_length",
            )

        if "max_length" in constraints and len(value) > constraints["max_length"]:
            raise ValidationError(
                f"max_length constraint: String length must " f"be at most {constraints['max_length']}",
                value=value,
                constraint="max_length",
            )

        if "pattern" in constraints and not constraints["pattern"].match(value):
            raise ValidationError(
                "pattern constraint: String must match the specified pattern", value=value, constraint="pattern"
            )

    return value


def validate_number(
    value: Any, property_type: PropertyType, constraints: Optional[dict[str, Any]] = None
) -> Union[int, float]:
    """
    Validate a numeric value against its constraints.

    Args:
        value: Value to validate
        property_type: Type of the property (INTEGER or FLOAT)
        constraints: Optional dictionary of constraints

    Returns:
        The validated numeric value

    Raises:
        ValidationError: If validation fails
    """
    if property_type == PropertyType.INTEGER:
        if not isinstance(value, int):
            raise ValidationError("Value must be an integer", value=value, constraint="type")
    elif property_type == PropertyType.FLOAT:
        value = _validate_float_value(value)

    if constraints:
        if "minimum" in constraints and value < constraints["minimum"]:
            raise ValidationError(
                f"minimum constraint: Value must be at least {constraints['minimum']}",
                value=value,
                constraint="minimum",
            )

        if "maximum" in constraints and value > constraints["maximum"]:
            raise ValidationError(
                f"maximum constraint: Value must be at most {constraints['maximum']}", value=value, constraint="maximum"
            )

    return value


def _validate_float_value(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value is None:
        raise ValidationError("Value must be a number", value=value, constraint="type")
    value = float(value)
    # Validate special float values
    if math.isnan(value):
        raise ValidationError("NaN values are not allowed", value=value, constraint="type")
    if math.isinf(value):
        raise ValidationError("Infinite values are not allowed", value=value, constraint="type")

    return value


def validate_boolean(value: Any) -> bool:
    """
    Validate a boolean value.

    Args:
        value: Value to validate

    Returns:
        The validated boolean value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, bool):
        raise ValidationError("Value must be a boolean", value=value, constraint="type")
    return value


def validate_datetime(value: Any, constraints: Optional[dict[str, Any]] = None) -> datetime:
    """
    Validate a datetime value against its constraints.

    Args:
        value: Value to validate
        constraints: Optional dictionary of constraints

    Returns:
        The validated datetime value

    Raises:
        ValidationError: If validation fails
    """
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as e:
            raise ValidationError("Invalid datetime format", value=value, constraint="format") from e

    if not isinstance(value, datetime):
        raise ValidationError("Value must be a datetime", value=value, constraint="type")

    if constraints:
        if "min_date" in constraints and value < constraints["min_date"]:
            raise ValidationError(
                f"min_date constraint: Date must be after {constraints['min_date']}", value=value, constraint="min_date"
            )

        if "max_date" in constraints and value > constraints["max_date"]:
            raise ValidationError(
                f"max_date constraint: Date must be before {constraints['max_date']}",
                value=value,
                constraint="max_date",
            )

    return value


def validate_uuid(value: Any) -> UUID:
    """
    Validate a UUID value.

    Args:
        value: Value to validate

    Returns:
        The validated UUID value

    Raises:
        ValidationError: If validation fails
    """
    if isinstance(value, str):
        try:
            value = UUID(value)
        except ValueError as e:
            raise ValidationError("Invalid UUID format", value=value, constraint="format") from e

    if not isinstance(value, UUID):
        raise ValidationError("Value must be a UUID", value=value, constraint="type")

    return value


def validate_list(value: Any, constraints: Optional[dict[str, Any]] = None) -> list[Any]:
    """
    Validate a list value against its constraints.

    Args:
        value: Value to validate
        constraints: Optional dictionary of constraints

    Returns:
        The validated list value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, list):
        raise ValidationError("Value must be a list", value=value, constraint="type")

    if constraints:
        if "min_items" in constraints and len(value) < constraints["min_items"]:
            raise ValidationError(
                f"min_items constraint: List must have at " f"least {constraints['min_items']} items",
                value=value,
                constraint="min_items",
            )

        if "max_items" in constraints and len(value) > constraints["max_items"]:
            raise ValidationError(
                f"max_items constraint: List must have " f"at most {constraints['max_items']} items",
                value=value,
                constraint="max_items",
            )

        if "item_type" in constraints:
            item_type = constraints["item_type"]
            item_constraints = constraints.get("item_constraints")
            for i, item in enumerate(value):
                try:
                    validator = _PROPERTY_VALIDATORS.get(item_type)
                    if validator is None:
                        raise ValidationError(f"Unsupported property type: {item_type}", constraint="type")
                    value[i] = validator(item, item_constraints)
                except ValidationError as e:
                    raise ValidationError(
                        f"Invalid item at index {i}: {e!s}", value=item, constraint=e.details.get("constraint")
                    ) from e

    return value


def validate_dict(value: Any, constraints: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Validate a dictionary value against its constraints.

    Args:
        value: Value to validate
        constraints: Optional dictionary of constraints

    Returns:
        The validated dictionary value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, dict):
        raise ValidationError("Value must be a dictionary", value=value, constraint="type")

    if constraints:
        # Check for additional properties if specified
        if constraints.get("additional_properties") is False:
            allowed_props = set(constraints.get("properties", {}).keys())
            actual_props = set(value.keys())
            extra_props = actual_props - allowed_props
            if extra_props:
                raise ValidationError(
                    f"Additional properties are not allowed: {', '.join(extra_props)}",
                    value=value,
                    constraint="additional_properties",
                )

        if "properties" in constraints:
            _validate_constraints(value, constraints)

    return value


def _validate_constraints(value, constraints):
    properties = constraints["properties"]
    for prop_name, prop_def in properties.items():
        if prop_def.get("required", False) and prop_name not in value:
            raise ValidationError(
                f"required constraint: Property '{prop_name}' is missing", field=prop_name, constraint="required"
            )

        if prop_name in value:
            prop_type = prop_def["type"]
            prop_constraints = prop_def.get("constraints")
            try:
                validator = _PROPERTY_VALIDATORS.get(prop_type)
                if validator is None:
                    raise ValidationError(f"Unsupported property type: {prop_type}", constraint="type")
                value[prop_name] = validator(value[prop_name], prop_constraints)
            except ValidationError as e:
                raise ValidationError(
                    f"Invalid value for property '{prop_name}': {e!s}",
                    field=prop_name,
                    value=value[prop_name],
                    constraint=e.details.get("constraint"),
                ) from e


def validate_property_value(value: Any, property_def: PropertyDefinition) -> Any:
    """
    Validate a property value against its definition.

    Args:
        value: Value to validate
        property_def: Property definition to validate against

    Returns:
        The validated property value

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if property_def.required:
            raise ValidationError("required constraint: Property value is required", constraint="required")
        return property_def.default

    try:
        validator = _PROPERTY_VALIDATORS.get(property_def.type)
        if validator is None:
            raise ValidationError(f"Unsupported property type: {property_def.type}", constraint="type")
        return validator(value, property_def.constraints)
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Validation failed: {e!s}", value=value, constraint="type") from e
