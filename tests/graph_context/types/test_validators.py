"""Tests for type validation logic."""
import pytest
from datetime import datetime, UTC
from uuid import UUID

from graph_context.exceptions import ValidationError
from graph_context.types.type_base import PropertyDefinition, PropertyType
from graph_context.types.validators import (
    validate_string,
    validate_number,
    validate_boolean,
    validate_datetime,
    validate_uuid,
    validate_list,
    validate_dict,
    validate_property_value
)


def test_validate_string():
    """Test string validation."""
    # Basic validation
    assert validate_string("test") == "test"

    # Length constraints
    constraints = {"min_length": 3, "max_length": 5}
    assert validate_string("test", constraints) == "test"

    with pytest.raises(ValidationError) as exc_info:
        validate_string("ab", constraints)
    assert "min_length" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        validate_string("too long", constraints)
    assert "max_length" in str(exc_info.value)

    # Type validation
    with pytest.raises(ValidationError) as exc_info:
        validate_string(123)
    assert "must be a string" in str(exc_info.value)


def test_validate_number():
    """Test number validation."""
    # Integer validation
    assert validate_number(42, PropertyType.INTEGER) == 42

    with pytest.raises(ValidationError) as exc_info:
        validate_number(3.14, PropertyType.INTEGER)
    assert "must be an integer" in str(exc_info.value)

    # Float validation
    assert validate_number(3.14, PropertyType.FLOAT) == 3.14
    assert validate_number(42, PropertyType.FLOAT) == 42.0

    with pytest.raises(ValidationError) as exc_info:
        validate_number("42", PropertyType.FLOAT)
    assert "must be a number" in str(exc_info.value)

    # Range constraints
    constraints = {"minimum": 0, "maximum": 100}
    assert validate_number(42, PropertyType.INTEGER, constraints) == 42

    with pytest.raises(ValidationError) as exc_info:
        validate_number(-1, PropertyType.INTEGER, constraints)
    assert "minimum" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        validate_number(101, PropertyType.INTEGER, constraints)
    assert "maximum" in str(exc_info.value)


def test_validate_boolean():
    """Test boolean validation."""
    assert validate_boolean(True) is True
    assert validate_boolean(False) is False

    with pytest.raises(ValidationError) as exc_info:
        validate_boolean(1)
    assert "must be a boolean" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        validate_boolean("true")
    assert "must be a boolean" in str(exc_info.value)


def test_validate_datetime():
    """Test datetime validation."""
    now = datetime.now(UTC)
    assert validate_datetime(now) == now

    # String parsing
    dt_str = "2024-01-01T12:00:00"
    dt = validate_datetime(dt_str)
    assert isinstance(dt, datetime)
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 1

    with pytest.raises(ValidationError) as exc_info:
        validate_datetime("invalid date")
    assert "format" in str(exc_info.value)

    # Range constraints
    min_date = datetime(2024, 1, 1, tzinfo=UTC)
    max_date = datetime(2024, 12, 31, tzinfo=UTC)
    constraints = {"min_date": min_date, "max_date": max_date}

    valid_date = datetime(2024, 6, 1, tzinfo=UTC)
    assert validate_datetime(valid_date, constraints) == valid_date

    with pytest.raises(ValidationError) as exc_info:
        validate_datetime(datetime(2023, 12, 31, tzinfo=UTC), constraints)
    assert "after" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        validate_datetime(datetime(2025, 1, 1, tzinfo=UTC), constraints)
    assert "before" in str(exc_info.value)


def test_validate_uuid():
    """Test UUID validation."""
    uuid_str = "550e8400-e29b-41d4-a716-446655440000"
    uuid = UUID(uuid_str)

    # UUID object
    assert validate_uuid(uuid) == uuid

    # UUID string
    assert validate_uuid(uuid_str) == uuid

    with pytest.raises(ValidationError) as exc_info:
        validate_uuid("invalid-uuid")
    assert "format" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        validate_uuid(123)
    assert "must be a UUID" in str(exc_info.value)


def test_validate_list():
    """Test list validation."""
    # Basic validation
    assert validate_list([1, 2, 3]) == [1, 2, 3]

    # Length constraints
    constraints = {"min_items": 2, "max_items": 4}
    assert validate_list([1, 2, 3], constraints) == [1, 2, 3]

    with pytest.raises(ValidationError) as exc_info:
        validate_list([1], constraints)
    assert "min_items" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        validate_list([1, 2, 3, 4, 5], constraints)
    assert "max_items" in str(exc_info.value)

    # Item type validation
    constraints = {
        "item_type": PropertyType.INTEGER,
        "item_constraints": {"minimum": 0}
    }
    assert validate_list([1, 2, 3], constraints) == [1, 2, 3]

    with pytest.raises(ValidationError) as exc_info:
        validate_list([1, "2", 3], constraints)
    assert "must be an integer" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        validate_list([1, -1, 3], constraints)
    assert "minimum" in str(exc_info.value)


def test_validate_dict():
    """Test dictionary validation."""
    # Basic validation
    assert validate_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    # Property validation
    constraints = {
        "properties": {
            "name": {
                "type": PropertyType.STRING,
                "required": True
            },
            "age": {
                "type": PropertyType.INTEGER,
                "constraints": {"minimum": 0}
            }
        }
    }

    valid_dict = {"name": "Alice", "age": 30}
    assert validate_dict(valid_dict, constraints) == valid_dict

    with pytest.raises(ValidationError) as exc_info:
        validate_dict({"age": 30}, constraints)
    assert "required" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        validate_dict({"name": "Alice", "age": -1}, constraints)
    assert "minimum" in str(exc_info.value)


def test_validate_property_value():
    """Test property value validation."""
    # Test with required property
    prop_def = PropertyDefinition(
        type=PropertyType.STRING,
        required=True
    )

    assert validate_property_value("test", prop_def) == "test"

    with pytest.raises(ValidationError) as exc_info:
        validate_property_value(None, prop_def)
    assert "required" in str(exc_info.value)

    # Test with default value
    prop_def = PropertyDefinition(
        type=PropertyType.INTEGER,
        default=42
    )

    assert validate_property_value(None, prop_def) == 42
    assert validate_property_value(123, prop_def) == 123

    # Test with constraints
    prop_def = PropertyDefinition(
        type=PropertyType.STRING,
        constraints={"min_length": 3}
    )

    assert validate_property_value("test", prop_def) == "test"

    with pytest.raises(ValidationError) as exc_info:
        validate_property_value("ab", prop_def)
    assert "min_length" in str(exc_info.value)

    # Test with unsupported type
    with pytest.raises(ValidationError) as exc_info:
        # Create a property definition with a valid type first
        prop_def = PropertyDefinition(type=PropertyType.STRING)
        # Then modify the type to be invalid (this bypasses Pydantic validation)
        prop_def.type = "unsupported"  # type: ignore
        validate_property_value("test", prop_def)
    assert "Unsupported property type" in str(exc_info.value)