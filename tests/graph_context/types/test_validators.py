"""Tests for type validation logic."""
import re
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest

from graph_context.exceptions import ValidationError
from graph_context.types.type_base import PropertyDefinition, PropertyType
from graph_context.types.validators import (
    validate_boolean,
    validate_datetime,
    validate_dict,
    validate_list,
    validate_number,
    validate_property_value,
    validate_string,
    validate_uuid,
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

    # Test non-string, non-datetime input
    with pytest.raises(ValidationError) as exc_info:
        validate_datetime(12345)
    assert "Value must be a datetime" in str(exc_info.value)
    assert exc_info.value.details["constraint"] == "type"

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


def test_validate_string_extended():
    """Test extended string validation scenarios."""
    # Pattern matching
    constraints = {"pattern": re.compile(r"^[A-Z][a-z]+$")}
    assert validate_string("Hello", constraints) == "Hello"

    with pytest.raises(ValidationError) as exc_info:
        validate_string("hello", constraints)
    assert "pattern" in str(exc_info.value)

    # Empty string
    assert validate_string("") == ""

    # Unicode strings
    unicode_str = "Hello 世界"
    assert validate_string(unicode_str) == unicode_str

    # Very long string
    long_str = "a" * 1000
    constraints = {"max_length": 100}
    with pytest.raises(ValidationError) as exc_info:
        validate_string(long_str, constraints)
    assert "max_length" in str(exc_info.value)


def test_validate_number_extended():
    """Test extended number validation scenarios."""
    # Edge cases for integers
    assert validate_number(0, PropertyType.INTEGER) == 0
    assert validate_number(-0, PropertyType.INTEGER) == 0

    # Large numbers
    large_int = 2**31 - 1
    assert validate_number(large_int, PropertyType.INTEGER) == large_int

    # Float edge cases
    assert validate_number(0.0, PropertyType.FLOAT) == 0.0
    assert validate_number(-0.0, PropertyType.FLOAT) == -0.0

    # Scientific notation
    assert validate_number(1e-10, PropertyType.FLOAT) == 1e-10
    assert validate_number(1E10, PropertyType.FLOAT) == 1E10

    # NaN and Infinity handling
    with pytest.raises(ValidationError):
        validate_number(float('nan'), PropertyType.FLOAT)

    with pytest.raises(ValidationError):
        validate_number(float('inf'), PropertyType.FLOAT)

    with pytest.raises(ValidationError):
        validate_number(float('-inf'), PropertyType.FLOAT)


def test_validate_datetime_extended():
    """Test extended datetime validation scenarios."""
    # Different timezone handling
    utc_now = datetime.now(timezone.utc)
    est = timezone(timedelta(hours=-5))
    est_now = utc_now.astimezone(est)

    assert validate_datetime(est_now) == est_now

    # Different datetime formats
    formats = [
        "2024-01-01T12:00:00",
        "2024-01-01T12:00:00Z",
        "2024-01-01T12:00:00+00:00",
        "2024-01-01T12:00:00-05:00"
    ]

    for dt_str in formats:
        dt = validate_datetime(dt_str)
        assert isinstance(dt, datetime)

    # Microsecond precision
    dt_with_micro = datetime(2024, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
    assert validate_datetime(dt_with_micro) == dt_with_micro

    # Invalid formats
    invalid_formats = [
        "2024/01/01",
        "12:00:00",
        "2024-13-01T12:00:00",
        "2024-01-32T12:00:00"
    ]

    for invalid_dt in invalid_formats:
        with pytest.raises(ValidationError):
            validate_datetime(invalid_dt)


def test_validate_list_extended():
    """Test extended list validation scenarios."""
    # Nested list validation
    constraints = {
        "item_type": PropertyType.LIST,
        "item_constraints": {
            "item_type": PropertyType.INTEGER,
            "min_items": 1
        }
    }

    valid_nested = [[1, 2], [3, 4]]
    assert validate_list(valid_nested, constraints) == valid_nested

    with pytest.raises(ValidationError):
        validate_list([[1, 2], []], constraints)

    # Mixed type validation
    mixed_constraints = {
        "item_type": PropertyType.STRING,
        "allow_mixed_types": True
    }

    mixed_list = ["string", 123, True]
    with pytest.raises(ValidationError):
        validate_list(mixed_list, mixed_constraints)

    # Empty list
    assert validate_list([]) == []

    constraints = {"min_items": 1}
    with pytest.raises(ValidationError):
        validate_list([], constraints)


def test_validate_dict_extended():
    """Test extended dictionary validation scenarios."""
    # Nested dictionary validation
    constraints = {
        "properties": {
            "user": {
                "type": PropertyType.DICT,
                "required": True,
                "constraints": {
                    "properties": {
                        "name": {"type": PropertyType.STRING, "required": True},
                        "age": {"type": PropertyType.INTEGER, "required": False}
                    }
                }
            }
        }
    }

    valid_nested = {
        "user": {
            "name": "Alice",
            "age": 30
        }
    }
    assert validate_dict(valid_nested, constraints) == valid_nested

    with pytest.raises(ValidationError):
        validate_dict({"user": {"age": 30}}, constraints)

    # Additional properties
    constraints = {
        "properties": {
            "name": {"type": PropertyType.STRING, "required": True}
        },
        "additional_properties": False
    }

    with pytest.raises(ValidationError):
        validate_dict({"name": "Alice", "extra": "value"}, constraints)

    # Empty dictionary
    assert validate_dict({}) == {}

    constraints = {
        "properties": {
            "required_prop": {"type": PropertyType.STRING, "required": True}
        }
    }
    with pytest.raises(ValidationError):
        validate_dict({}, constraints)


def test_validate_property_value_extended():
    """Test extended property value validation scenarios."""
    # Complex nested property
    prop_def = PropertyDefinition(
        type=PropertyType.DICT,
        required=True,
        constraints={
            "properties": {
                "list_prop": {
                    "type": PropertyType.LIST,
                    "required": True,
                    "constraints": {
                        "item_type": PropertyType.STRING,
                        "min_items": 1
                    }
                }
            }
        }
    )

    valid_value = {
        "list_prop": ["item1", "item2"]
    }
    assert validate_property_value(valid_value, prop_def) == valid_value

    invalid_value = {
        "list_prop": []
    }
    with pytest.raises(ValidationError):
        validate_property_value(invalid_value, prop_def)

    # Default value with constraints
    prop_def = PropertyDefinition(
        type=PropertyType.STRING,
        default="default",
        constraints={"min_length": 3}
    )

    assert validate_property_value(None, prop_def) == "default"

    with pytest.raises(ValidationError):
        validate_property_value("ab", prop_def)


def test_validate_datetime_conversion_error():
    """Test datetime validation with invalid string format."""
    with pytest.raises(ValidationError, match="Invalid datetime format"):
        validate_datetime("not-a-datetime")


def test_validate_list_nested_validation_errors():
    """Test list validation with nested validation errors."""
    constraints = {
        "item_type": PropertyType.DICT,
        "item_constraints": {
            "properties": {
                "name": {"type": PropertyType.STRING, "required": True},
                "age": {"type": PropertyType.INTEGER}
            }
        }
    }
    invalid_list = [
        {"name": "Alice", "age": 30},  # Valid
        {"age": 25},  # Missing required name
        {"name": "Charlie", "age": "invalid"}  # Invalid age type
    ]

    with pytest.raises(ValidationError, match="Invalid item at index 1"):
        validate_list(invalid_list, constraints)

    # Test other property types in list validation
    bool_constraints = {"item_type": PropertyType.BOOLEAN}
    with pytest.raises(ValidationError, match="Invalid item at index 0"):
        validate_list(["not-a-bool"], bool_constraints)

    uuid_constraints = {"item_type": PropertyType.UUID}
    with pytest.raises(ValidationError, match="Invalid item at index 0"):
        validate_list(["not-a-uuid"], uuid_constraints)


def test_validate_dict_nested_all_types():
    """Test dictionary validation with all possible property types."""
    constraints = {
        "properties": {
            "string_prop": {"type": PropertyType.STRING},
            "int_prop": {"type": PropertyType.INTEGER},
            "float_prop": {"type": PropertyType.FLOAT},
            "bool_prop": {"type": PropertyType.BOOLEAN},
            "datetime_prop": {"type": PropertyType.DATETIME},
            "uuid_prop": {"type": PropertyType.UUID},
            "list_prop": {"type": PropertyType.LIST},
            "dict_prop": {"type": PropertyType.DICT}
        }
    }

    valid_dict = {
        "string_prop": "test",
        "int_prop": 42,
        "float_prop": 3.14,
        "bool_prop": True,
        "datetime_prop": datetime.now(),
        "uuid_prop": UUID("550e8400-e29b-41d4-a716-446655440000"),
        "list_prop": [1, 2, 3],
        "dict_prop": {"key": "value"}
    }

    # This should pass
    result = validate_dict(valid_dict, constraints)
    assert isinstance(result, dict)

    # Test invalid values for each type
    for prop, invalid_value in [
        ("string_prop", 123),
        ("int_prop", "not-an-int"),
        ("float_prop", "not-a-float"),
        ("bool_prop", "not-a-bool"),
        ("datetime_prop", "not-a-datetime"),
        ("uuid_prop", "not-a-uuid"),
        ("list_prop", "not-a-list"),
        ("dict_prop", "not-a-dict")
    ]:
        invalid_dict = valid_dict.copy()
        invalid_dict[prop] = invalid_value
        with pytest.raises(ValidationError):
            validate_dict(invalid_dict, constraints)


def test_validate_property_value_edge_cases():
    """Test property value validation with edge cases."""
    # Test None value with required property
    required_def = PropertyDefinition(
        type=PropertyType.STRING,
        required=True
    )
    with pytest.raises(ValidationError, match="Property value is required"):
        validate_property_value(None, required_def)

    # Test None value with optional property and default
    optional_def = PropertyDefinition(
        type=PropertyType.STRING,
        required=False,
        default="default_value"
    )
    result = validate_property_value(None, optional_def)
    assert result == "default_value"

    # Test validation error propagation
    list_def = PropertyDefinition(
        type=PropertyType.LIST,
        constraints={
            "item_type": PropertyType.INTEGER
        }
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_property_value(["not-an-int"], list_def)
    assert "Invalid item at index 0" in str(exc_info.value)

    # Test with an invalid property type (this will test the else branch)
    class MockPropertyDefinition:
        def __init__(self):
            self.type = "INVALID_TYPE"
            self.required = True

    mock_def = MockPropertyDefinition()
    with pytest.raises(ValidationError, match="Unsupported property type"):
        validate_property_value("test", mock_def)


def test_validate_dict_additional_properties_with_empty_properties():
    """Test dictionary validation with additional
       properties check but no defined properties."""
    constraints = {
        "additional_properties": False,
        "properties": {}
    }

    with pytest.raises(ValidationError, match="Additional properties are not allowed"):
        validate_dict({"extra": "property"}, constraints)


def test_validate_list_with_all_property_types():
    """Test list validation with all possible property types as items."""
    for prop_type in PropertyType:
        constraints = {"item_type": prop_type}
        if prop_type == PropertyType.STRING:
            validate_list(["test"], constraints)
        elif prop_type == PropertyType.INTEGER:
            validate_list([42], constraints)
        elif prop_type == PropertyType.FLOAT:
            validate_list([3.14], constraints)
        elif prop_type == PropertyType.BOOLEAN:
            validate_list([True], constraints)
        elif prop_type == PropertyType.DATETIME:
            validate_list([datetime.now()], constraints)
        elif prop_type == PropertyType.UUID:
            validate_list([UUID("550e8400-e29b-41d4-a716-446655440000")], constraints)
        elif prop_type == PropertyType.LIST:
            validate_list([[1, 2, 3]], constraints)
        elif prop_type == PropertyType.DICT:
            validate_list([{"key": "value"}], constraints)


def test_validate_list_with_all_property_types():
    """Test list validation with all property types."""
    constraints = {
        "item_type": PropertyType.LIST,
        "item_constraints": {
            "item_type": PropertyType.DICT,
            "item_constraints": {
                "properties": {
                    "name": {"type": PropertyType.STRING, "required": True},
                    "age": {"type": PropertyType.INTEGER},
                    "active": {"type": PropertyType.BOOLEAN}
                }
            }
        }
    }
    value = [[{"name": "Alice", "age": 30, "active": True}]]
    assert validate_list(value, constraints) == value

def test_validate_float_value():
    """Test internal float validation function."""
    from graph_context.types.validators import _validate_float_value

    # Test valid float values
    assert _validate_float_value(3.14) == 3.14
    assert _validate_float_value(-2.5) == -2.5
    assert _validate_float_value(0.0) == 0.0

    # Test integer conversion to float
    assert _validate_float_value(42) == 42.0
    assert _validate_float_value(-17) == -17.0
    assert _validate_float_value(0) == 0.0


def test_validate_fload_value_with_exceptions():
    """Test float validation with exceptions."""
    from graph_context.types.validators import _validate_float_value

    # Test NaN rejection
    with pytest.raises(ValidationError) as exc_info:
        _validate_float_value(float('nan'))
    assert "NaN values are not allowed" in str(exc_info.value)

    # Test infinity rejection
    with pytest.raises(ValidationError) as exc_info:
        _validate_float_value(float('inf'))
    assert "Infinite values are not allowed" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        _validate_float_value(float('-inf'))
    assert "Infinite values are not allowed" in str(exc_info.value)

    # Test invalid type rejection
    invalid_values = ["3.14", [1.0], {"value": 2.0}, None, True]
    for invalid_value in invalid_values:
        with pytest.raises(ValidationError) as exc_info:
            _validate_float_value(invalid_value)
        assert "Value must be a number" in str(exc_info.value)


def test_validate_float_special_values():
    """Test validation of special float values."""
    # Test NaN
    with pytest.raises(ValidationError) as exc_info:
        validate_number(float('nan'), PropertyType.FLOAT)
    assert "NaN values are not allowed" in str(exc_info.value)

    # Test positive infinity
    with pytest.raises(ValidationError) as exc_info:
        validate_number(float('inf'), PropertyType.FLOAT)
    assert "Infinite values are not allowed" in str(exc_info.value)

    # Test negative infinity
    with pytest.raises(ValidationError) as exc_info:
        validate_number(float('-inf'), PropertyType.FLOAT)
    assert "Infinite values are not allowed" in str(exc_info.value)


def test_validate_datetime_timezone_handling():
    """Test datetime validation with timezone handling."""
    # Test datetime with timezone
    dt_with_tz = datetime.now(timezone(timedelta(hours=2)))
    assert validate_datetime(dt_with_tz) == dt_with_tz

    # Test datetime string with timezone
    dt_str_with_tz = "2024-01-01T12:00:00+02:00"
    result = validate_datetime(dt_str_with_tz)
    assert result.tzinfo is not None
    assert result.utcoffset() == timedelta(hours=2)


def test_validate_dict_additional_properties():
    """Test dictionary validation with additional properties."""
    constraints = {
        "properties": {
            "name": {"type": PropertyType.STRING, "required": True}
        },
        "additional_properties": False
    }

    # Test with no additional properties
    valid_dict = {"name": "test"}
    assert validate_dict(valid_dict, constraints) == valid_dict

    # Test with additional properties when not allowed
    with pytest.raises(ValidationError) as exc_info:
        validate_dict({"name": "test", "extra": "value"}, constraints)
    assert "Additional properties are not allowed" in str(exc_info.value)

    # Test with additional properties when allowed
    constraints["additional_properties"] = True
    assert validate_dict({"name": "test", "extra": "value"}, constraints) == {"name": "test", "extra": "value"}


def test_validate_nested_list_validation():
    """Test nested list validation with complex constraints."""
    constraints = {
        "item_type": PropertyType.LIST,
        "item_constraints": {
            "item_type": PropertyType.INTEGER,
            "item_constraints": {"minimum": 0}
        }
    }

    # Valid nested list
    valid_list = [[1, 2], [3, 4]]
    assert validate_list(valid_list, constraints) == valid_list

    # Invalid inner list item
    with pytest.raises(ValidationError) as exc_info:
        validate_list([[1, -2], [3, 4]], constraints)
    assert "minimum" in str(exc_info.value)

    # Invalid inner list type
    with pytest.raises(ValidationError) as exc_info:
        validate_list([[1, "2"], [3, 4]], constraints)
    assert "must be an integer" in str(exc_info.value)


def test_validate_property_value_complex():
    """Test property value validation with complex nested structures."""
    # Test list property with nested dictionary
    list_prop_def = PropertyDefinition(
        type=PropertyType.LIST,
        constraints={
            "item_type": PropertyType.DICT,
            "item_constraints": {
                "properties": {
                    "id": {"type": PropertyType.UUID, "required": True},
                    "data": {
                        "type": PropertyType.DICT,
                        "constraints": {
                            "properties": {
                                "value": {"type": PropertyType.FLOAT},
                                "timestamp": {"type": PropertyType.DATETIME}
                            }
                        }
                    }
                }
            }
        }
    )

    uuid_val = UUID('550e8400-e29b-41d4-a716-446655440000')
    dt_val = datetime.now(UTC)
    value = [{
        "id": uuid_val,
        "data": {
            "value": 42.5,
            "timestamp": dt_val
        }
    }]

    result = validate_property_value(value, list_prop_def)
    assert result == value
    assert isinstance(result[0]["id"], UUID)
    assert isinstance(result[0]["data"]["timestamp"], datetime)


def test_validate_string_pattern():
    """Test string validation with regex pattern."""
    pattern = re.compile(r'^[A-Z][a-z]+$')
    constraints = {"pattern": pattern}

    # Valid pattern
    assert validate_string("Hello", constraints) == "Hello"

    # Invalid pattern
    with pytest.raises(ValidationError) as exc_info:
        validate_string("123", constraints)
    assert "pattern constraint" in str(exc_info.value)


def test_validate_list_empty_constraints():
    """Test list validation with empty constraints."""
    # Test with empty constraints
    value = [1, 2, 3]
    assert validate_list(value, {}) == value

    # Test with None constraints
    assert validate_list(value, None) == value


def test_validate_dict_empty_properties():
    """Test dictionary validation with empty properties."""
    # Test with empty properties
    constraints = {"properties": {}}
    value = {"extra": "value"}
    assert validate_dict(value, constraints) == value

    # Test with additional_properties explicitly set
    constraints = {"properties": {}, "additional_properties": True}
    assert validate_dict(value, constraints) == value


def test_validate_property_value_none():
    """Test property value validation with None value."""
    prop_def = PropertyDefinition(
        type=PropertyType.STRING,
        required=False
    )
    assert validate_property_value(None, prop_def) is None


def test_validate_string_pattern_edge_cases():
    """Test string validation with regex pattern edge cases."""
    # Test with a pattern that doesn't match at all
    pattern = re.compile(r'^[0-9]+$')  # Only digits
    constraints = {"pattern": pattern}

    # Test with a string that doesn't match at all
    with pytest.raises(ValidationError) as exc_info:
        validate_string("abc", constraints)
    assert "pattern constraint" in str(exc_info.value)

    # Test with a string that partially matches but not from start
    with pytest.raises(ValidationError) as exc_info:
        validate_string("abc123", constraints)
    assert "pattern constraint" in str(exc_info.value)

    # Test with empty string
    with pytest.raises(ValidationError) as exc_info:
        validate_string("", constraints)
    assert "pattern constraint" in str(exc_info.value)

    # Test with valid pattern
    assert validate_string("123", constraints) == "123"

def test_validate_constraints_required_property():
    """Test _validate_constraints with missing required property."""
    from graph_context.types.validators import _validate_constraints, ValidationError
    from graph_context.types.type_base import PropertyType

    constraints = {
        "properties": {
            "name": {
                "type": PropertyType.STRING,
                "required": True
            }
        }
    }
    value = {}

    with pytest.raises(ValidationError) as exc_info:
        _validate_constraints(value, constraints)
    assert "required constraint: Property 'name' is missing" in str(exc_info.value)
    assert exc_info.value.details["field"] == "name"
    assert exc_info.value.details["constraint"] == "required"


def test_validate_constraints_invalid_type():
    """Test _validate_constraints with invalid property type."""
    from graph_context.types.validators import _validate_constraints, ValidationError
    from graph_context.types.type_base import PropertyType

    constraints = {
        "properties": {
            "age": {
                "type": "INVALID_TYPE"
            }
        }
    }
    value = {"age": 25}

    with pytest.raises(ValidationError) as exc_info:
        _validate_constraints(value, constraints)
    assert "Unsupported property type" in str(exc_info.value)
    assert exc_info.value.details["constraint"] == "type"


def test_validate_constraints_validation_error():
    """Test _validate_constraints propagating validation error from property validator."""
    from graph_context.types.validators import _validate_constraints, ValidationError
    from graph_context.types.type_base import PropertyType

    constraints = {
        "properties": {
            "age": {
                "type": PropertyType.INTEGER,
                "constraints": {"minimum": 0}
            }
        }
    }
    value = {"age": -1}

    with pytest.raises(ValidationError) as exc_info:
        _validate_constraints(value, constraints)
    assert "Invalid value for property 'age'" in str(exc_info.value)
    assert exc_info.value.details["field"] == "age"
    assert exc_info.value.details["value"] == -1
    assert exc_info.value.details["constraint"] == "minimum"


def test_validate_constraints_successful():
    """Test _validate_constraints with successful validation."""
    from graph_context.types.validators import _validate_constraints
    from graph_context.types.type_base import PropertyType

    constraints = {
        "properties": {
            "name": {
                "type": PropertyType.STRING,
                "constraints": {"min_length": 2}
            },
            "age": {
                "type": PropertyType.INTEGER,
                "constraints": {"minimum": 0}
            }
        }
    }
    value = {"name": "Alice", "age": 25}

    # Function modifies dictionary in-place
    _validate_constraints(value, constraints)
    assert value["name"] == "Alice"
    assert value["age"] == 25


def test_validate_constraints_optional_property():
    """Test _validate_constraints with optional property."""
    from graph_context.types.validators import _validate_constraints
    from graph_context.types.type_base import PropertyType

    constraints = {
        "properties": {
            "name": {
                "type": PropertyType.STRING,
                "required": True
            },
            "age": {
                "type": PropertyType.INTEGER,
                "required": False
            }
        }
    }
    value = {"name": "Alice"}  # age is optional

    # Should not raise any error for missing optional property
    _validate_constraints(value, constraints)
    assert value["name"] == "Alice"
    assert "age" not in value