"""
Base type definitions for the graph-context module.

This module defines the core types used throughout the graph-context component,
including entity and relation types, property types, and validation rules.
"""
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PropertyType(str, Enum):
    """Enumeration of supported property types."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    UUID = "uuid"
    LIST = "list"
    DICT = "dict"


class PropertyDefinition(BaseModel):
    """Definition of a property in the schema."""

    type: PropertyType
    required: bool = False
    default: Any | None = None
    description: str | None = None
    constraints: dict[str, Any] | None = None


class EntityType(BaseModel):
    """Definition of an entity type in the schema."""

    name: str = Field(..., description="Name of the entity type")
    description: str | None = None
    properties: dict[str, PropertyDefinition] = Field(
        default_factory=dict, description="Dictionary of property definitions"
    )
    indexes: list[str] = Field(default_factory=list, description="List of property names to index")


class RelationType(BaseModel):
    """Definition of a relation type in the schema."""

    name: str = Field(..., description="Name of the relation type")
    description: str | None = None
    properties: dict[str, PropertyDefinition] = Field(
        default_factory=dict, description="Dictionary of property definitions"
    )
    from_types: list[str] = Field(..., description="List of allowed entity types for the source")
    to_types: list[str] = Field(..., description="List of allowed entity types for the target")
    indexes: list[str] = Field(default_factory=list, description="List of property names to index")


class Entity(BaseModel):
    """Representation of an entity in the graph."""

    id: str = Field(..., description="Unique identifier of the entity")
    type: str = Field(..., description="Type of the entity")
    properties: dict[str, Any] = Field(default_factory=dict, description="Dictionary of property values")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Timestamp of entity creation")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Timestamp of last update")


class Relation(BaseModel):
    """Representation of a relation in the graph."""

    id: str = Field(..., description="Unique identifier of the relation")
    type: str = Field(..., description="Type of the relation")
    from_entity: str = Field(..., description="ID of the source entity")
    to_entity: str = Field(..., description="ID of the target entity")
    properties: dict[str, Any] = Field(default_factory=dict, description="Dictionary of property values")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Timestamp of relation creation"
    )
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Timestamp of last update")


class QueryOperator(str, Enum):
    """Enumeration of supported query operators."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"


class QueryCondition(BaseModel):
    """Representation of a query condition."""

    field: str = Field(..., description="Field name to query")
    operator: QueryOperator = Field(..., description="Query operator")
    value: Any = Field(..., description="Value to compare against")


class QuerySpec(BaseModel):
    """Specification for a graph query."""

    entity_type: str | None = Field(None, description="Type of entities to query")
    conditions: list[QueryCondition] = Field(default_factory=list, description="List of query conditions")
    limit: int | None = Field(None, description="Maximum number of results to return")
    offset: int | None = Field(None, description="Number of results to skip")


class TraversalDirection(str, Enum):
    """Enumeration of traversal directions."""

    OUTBOUND = "outbound"
    INBOUND = "inbound"
    ANY = "any"


class TraversalSpec(BaseModel):
    """Specification for a graph traversal."""

    max_depth: int = Field(..., description="Maximum depth of traversal", ge=1)
    relation_types: list[str] | None = Field(None, description="List of relation types to traverse")
    direction: TraversalDirection = Field(TraversalDirection.ANY, description="Direction of traversal")
    conditions: list[QueryCondition] | None = Field(None, description="Conditions to filter traversed entities")
