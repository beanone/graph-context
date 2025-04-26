"""
Type definitions for the graph-context module.

This package provides the core type definitions used throughout the graph-context
component, including entity and relation types, property types, and validation
rules.
"""

from .type_base import (
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
from .validators import validate_property_value

__all__ = [
    # Base types
    "Entity",
    "EntityType",
    "Relation",
    "RelationType",
    "PropertyType",
    "PropertyDefinition",
    "QueryOperator",
    "QueryCondition",
    "QuerySpec",
    "TraversalDirection",
    "TraversalSpec",
    # Validators
    "validate_property_value",
]
