"""Tests for base type definitions."""
from datetime import UTC, datetime

import pytest

from graph_context.types.type_base import (
    Entity,
    EntityType,
    PropertyDefinition,
    PropertyType,
    QueryCondition,
    QueryOperator,
    QuerySpec,
    Relation,
    RelationType,
    TraversalDirection,
    TraversalSpec,
)


def test_property_type_values():
    """Test that PropertyType enum has the expected values."""
    assert PropertyType.STRING == "string"
    assert PropertyType.INTEGER == "integer"
    assert PropertyType.FLOAT == "float"
    assert PropertyType.BOOLEAN == "boolean"
    assert PropertyType.DATETIME == "datetime"
    assert PropertyType.UUID == "uuid"
    assert PropertyType.LIST == "list"
    assert PropertyType.DICT == "dict"


def test_property_definition():
    """Test PropertyDefinition model."""
    # Test required property
    prop = PropertyDefinition(type=PropertyType.STRING, required=True)
    assert prop.type == PropertyType.STRING
    assert prop.required is True
    assert prop.default is None
    assert prop.description is None
    assert prop.constraints is None

    # Test with all fields
    prop = PropertyDefinition(
        type=PropertyType.INTEGER,
        required=False,
        default=42,
        description="Age in years",
        constraints={"minimum": 0, "maximum": 150},
    )
    assert prop.type == PropertyType.INTEGER
    assert prop.required is False
    assert prop.default == 42
    assert prop.description == "Age in years"
    assert prop.constraints == {"minimum": 0, "maximum": 150}


def test_entity_type():
    """Test EntityType model."""
    entity_type = EntityType(
        name="Person",
        description="A human being",
        properties={
            "name": PropertyDefinition(type=PropertyType.STRING, required=True),
            "age": PropertyDefinition(type=PropertyType.INTEGER),
        },
        indexes=["name"],
    )

    assert entity_type.name == "Person"
    assert entity_type.description == "A human being"
    assert len(entity_type.properties) == 2
    assert entity_type.properties["name"].required is True
    assert entity_type.properties["age"].required is False
    assert entity_type.indexes == ["name"]


def test_relation_type():
    """Test RelationType model."""
    relation_type = RelationType(
        name="KNOWS",
        description="Person knows another person",
        properties={"since": PropertyDefinition(type=PropertyType.DATETIME)},
        from_types=["Person"],
        to_types=["Person"],
        indexes=["since"],
    )

    assert relation_type.name == "KNOWS"
    assert relation_type.description == "Person knows another person"
    assert len(relation_type.properties) == 1
    assert relation_type.properties["since"].type == PropertyType.DATETIME
    assert relation_type.from_types == ["Person"]
    assert relation_type.to_types == ["Person"]
    assert relation_type.indexes == ["since"]


def test_entity():
    """Test Entity model."""
    now = datetime.now(UTC)
    entity = Entity(id="123", type="Person", properties={"name": "Alice", "age": 30}, created_at=now, updated_at=now)

    assert entity.id == "123"
    assert entity.type == "Person"
    assert entity.properties["name"] == "Alice"
    assert entity.properties["age"] == 30
    assert isinstance(entity.created_at, datetime)
    assert isinstance(entity.updated_at, datetime)


def test_relation():
    """Test Relation model."""
    now = datetime.now(UTC)
    relation = Relation(
        id="456",
        type="KNOWS",
        from_entity="123",
        to_entity="789",
        properties={"since": now},
        created_at=now,
        updated_at=now,
    )

    assert relation.id == "456"
    assert relation.type == "KNOWS"
    assert relation.from_entity == "123"
    assert relation.to_entity == "789"
    assert relation.properties["since"] == now
    assert isinstance(relation.created_at, datetime)
    assert isinstance(relation.updated_at, datetime)


def test_query_operator():
    """Test QueryOperator enum."""
    assert QueryOperator.EQUALS == "eq"
    assert QueryOperator.NOT_EQUALS == "ne"
    assert QueryOperator.GREATER_THAN == "gt"
    assert QueryOperator.GREATER_THAN_OR_EQUAL == "gte"
    assert QueryOperator.LESS_THAN == "lt"
    assert QueryOperator.LESS_THAN_OR_EQUAL == "lte"
    assert QueryOperator.IN == "in"
    assert QueryOperator.NOT_IN == "not_in"
    assert QueryOperator.CONTAINS == "contains"
    assert QueryOperator.NOT_CONTAINS == "not_contains"
    assert QueryOperator.STARTS_WITH == "starts_with"
    assert QueryOperator.ENDS_WITH == "ends_with"
    assert QueryOperator.REGEX == "regex"


def test_query_condition():
    """Test QueryCondition model."""
    condition = QueryCondition(field="age", operator=QueryOperator.GREATER_THAN, value=18)

    assert condition.field == "age"
    assert condition.operator == QueryOperator.GREATER_THAN
    assert condition.value == 18


def test_query_spec():
    """Test QuerySpec model."""
    spec = QuerySpec(
        entity_type="Person",
        conditions=[QueryCondition(field="age", operator=QueryOperator.GREATER_THAN, value=18)],
        limit=10,
        offset=0,
    )

    assert spec.entity_type == "Person"
    assert len(spec.conditions) == 1
    assert spec.conditions[0].field == "age"
    assert spec.limit == 10
    assert spec.offset == 0


def test_traversal_direction():
    """Test TraversalDirection enum."""
    assert TraversalDirection.OUTBOUND == "outbound"
    assert TraversalDirection.INBOUND == "inbound"
    assert TraversalDirection.ANY == "any"


def test_traversal_spec():
    """Test TraversalSpec model."""
    spec = TraversalSpec(
        max_depth=3,
        relation_types=["KNOWS", "WORKS_WITH"],
        direction=TraversalDirection.OUTBOUND,
        conditions=[QueryCondition(field="active", operator=QueryOperator.EQUALS, value=True)],
    )

    assert spec.max_depth == 3
    assert spec.relation_types == ["KNOWS", "WORKS_WITH"]
    assert spec.direction == TraversalDirection.OUTBOUND
    assert len(spec.conditions) == 1
    assert spec.conditions[0].field == "active"


def test_invalid_traversal_depth():
    """Test that TraversalSpec validates max_depth."""
    with pytest.raises(ValueError):
        TraversalSpec(max_depth=0)
