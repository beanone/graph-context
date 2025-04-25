"""
Tests for the SchemaValidator class.
"""
import pytest

from graph_context.context_base import SchemaValidator
from graph_context.types.type_base import EntityType, RelationType, PropertyType
from graph_context.types.type_base import PropertyDefinition
from graph_context.exceptions import SchemaError, ValidationError


@pytest.fixture
def entity_types():
    """Sample entity types for testing."""
    return {
        "person": EntityType(
            name="person",
            properties={
                "name": PropertyDefinition(type=PropertyType.STRING, required=True),
                "age": PropertyDefinition(type=PropertyType.INTEGER, required=False),
                "email": PropertyDefinition(
                    type=PropertyType.STRING,
                    required=False,
                    constraints={"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
                ),
                "active": PropertyDefinition(
                    type=PropertyType.BOOLEAN,
                    required=False,
                    default=True
                )
            }
        ),
        "document": EntityType(
            name="document",
            properties={
                "title": PropertyDefinition(type=PropertyType.STRING, required=True),
                "content": PropertyDefinition(type=PropertyType.STRING, required=False)
            }
        )
    }


@pytest.fixture
def relation_types(entity_types):
    """Sample relation types for testing."""
    return {
        "authored": RelationType(
            name="authored",
            from_types=["person"],
            to_types=["document"],
            properties={
                "year": PropertyDefinition(type=PropertyType.INTEGER, required=False),
                "is_primary_author": PropertyDefinition(
                    type=PropertyType.BOOLEAN,
                    required=False,
                    default=True
                )
            }
        ),
        "likes": RelationType(
            name="likes",
            from_types=["person"],
            to_types=["document", "person"],
            properties={}
        )
    }


@pytest.fixture
def validator(entity_types, relation_types):
    """SchemaValidator instance with test types."""
    return SchemaValidator(entity_types, relation_types)


class TestSchemaValidator:
    """Test cases for SchemaValidator class."""

    def test_init(self, entity_types, relation_types):
        """Test SchemaValidator initialization."""
        validator = SchemaValidator(entity_types, relation_types)
        assert validator._entity_types is entity_types
        assert validator._relation_types is relation_types

    def test_validate_entity_success(self, validator):
        """Test successful entity validation."""
        # Test with all required properties
        props = validator.validate_entity("person", {"name": "John Doe", "age": 30})
        assert props == {"name": "John Doe", "age": 30, "active": True}

        # Test with only required properties
        props = validator.validate_entity("person", {"name": "Jane Doe"})
        assert props == {"name": "Jane Doe", "active": True}

    def test_validate_entity_unknown_type(self, validator):
        """Test entity validation with unknown type."""
        with pytest.raises(SchemaError) as exc_info:
            validator.validate_entity("unknown", {"name": "Test"})
        assert "Unknown entity type: unknown" in str(exc_info.value)

    def test_validate_entity_missing_required(self, validator):
        """Test entity validation with missing required property."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_entity("person", {"age": 30})
        assert "Missing required property: name" in str(exc_info.value)

    def test_validate_entity_unknown_property(self, validator):
        """Test entity validation with unknown property."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_entity("person", {"name": "Test", "unknown": "value"})
        assert "Unknown property: unknown" in str(exc_info.value)

    def test_validate_entity_default_values(self, validator):
        """Test entity validation with default values."""
        props = validator.validate_entity("person", {"name": "Test"})
        assert props["active"] is True  # Default value should be set

    def test_validate_relation_success(self, validator):
        """Test successful relation validation."""
        # Test with all properties
        props = validator.validate_relation(
            "authored",
            "person",
            "document",
            {"year": 2023, "is_primary_author": False}
        )
        assert props == {"year": 2023, "is_primary_author": False}

        # Test with no properties, using defaults
        props = validator.validate_relation("authored", "person", "document")
        assert props == {"is_primary_author": True}

    def test_validate_relation_unknown_type(self, validator):
        """Test relation validation with unknown type."""
        with pytest.raises(SchemaError) as exc_info:
            validator.validate_relation("unknown", "person", "document")
        assert "Unknown relation type: unknown" in str(exc_info.value)

    def test_validate_relation_invalid_from_type(self, validator):
        """Test relation validation with invalid from type."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_relation("authored", "document", "document")
        assert "Invalid from_entity_type: document" in str(exc_info.value)

    def test_validate_relation_invalid_to_type(self, validator):
        """Test relation validation with invalid to type."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_relation("authored", "person", "person")
        assert "Invalid to_entity_type: person" in str(exc_info.value)

    def test_validate_relation_missing_required(self, validator, relation_types):
        """Test relation validation with missing required property."""
        # Modify relation_types to have a required property
        relation_types["authored"].properties["year"] = PropertyDefinition(
            type=PropertyType.INTEGER,
            required=True
        )

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_relation("authored", "person", "document", {})
        assert "Missing required property: year" in str(exc_info.value)

    def test_validate_relation_unknown_property(self, validator):
        """Test relation validation with unknown property."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_relation(
                "authored",
                "person",
                "document",
                {"unknown": "value"}
            )
        assert "Unknown property: unknown" in str(exc_info.value)

    def test_validate_relation_default_values(self, validator):
        """Test relation validation with default values."""
        props = validator.validate_relation("authored", "person", "document", {})
        assert props["is_primary_author"] is True  # Default value should be set