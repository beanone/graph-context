"""
Graph Context component for Knowledge Graph Assisted Research IDE.

This package provides a unified interface for graph operations, with support for
different backend implementations and strict type safety.
"""

from .context_base import BaseGraphContext
from .exceptions import (
    EntityNotFoundError,
    GraphContextError,
    QueryError,
    RelationNotFoundError,
    SchemaError,
    TransactionError,
    ValidationError,
)
from .interface import GraphContext
from .types.type_base import (
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

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "GraphContext",
    "BaseGraphContext",
    # Exceptions
    "GraphContextError",
    "EntityNotFoundError",
    "RelationNotFoundError",
    "ValidationError",
    "SchemaError",
    "TransactionError",
    "QueryError",
    # Types
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
]
