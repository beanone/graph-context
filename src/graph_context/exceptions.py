"""
Exceptions for the graph-context module.

This module defines all custom exceptions that can be raised by the graph-context
component during its operations.
"""
from typing import Any, Optional


class GraphContextError(Exception):
    """Base exception for all graph context errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary containing additional error details
        """
        super().__init__(message)
        self.details = details or {}


class EntityNotFoundError(GraphContextError):
    """Raised when an entity cannot be found."""

    def __init__(self, entity_id: str, entity_type: Optional[str] = None) -> None:
        """
        Initialize the exception.

        Args:
            entity_id: ID of the entity that could not be found
            entity_type: Optional type of the entity that could not be found
        """
        details = {"entity_id": entity_id}
        if entity_type:
            details["entity_type"] = entity_type
        super().__init__(
            f"Entity with ID '{entity_id}'" +
            (f" and type '{entity_type}'" if entity_type else "") +
            " not found",
            details
        )


class EntityTypeNotFoundError(GraphContextError):
    """Raised when an entity type cannot be found."""

    def __init__(self, entity_type: str) -> None:
        """
        Initialize the exception.

        Args:
            entity_type: Name of the entity type that could not be found
        """
        super().__init__(
            f"Entity type '{entity_type}' not found",
            {"entity_type": entity_type}
        )


class RelationNotFoundError(GraphContextError):
    """Raised when a relation cannot be found."""

    def __init__(
        self,
        relation_id: str,
        relation_type: Optional[str] = None,
        from_entity: Optional[str] = None,
        to_entity: Optional[str] = None
    ) -> None:
        """
        Initialize the exception.

        Args:
            relation_id: ID of the relation that could not be found
            relation_type: Optional type of the relation
            from_entity: Optional ID of the source entity
            to_entity: Optional ID of the target entity
        """
        details = {"relation_id": relation_id}
        if relation_type:
            details["relation_type"] = relation_type
        if from_entity:
            details["from_entity"] = from_entity
        if to_entity:
            details["to_entity"] = to_entity

        super().__init__(
            f"Relation with ID '{relation_id}'" +
            (f" and type '{relation_type}'" if relation_type else "") +
            " not found",
            details
        )


class RelationTypeNotFoundError(GraphContextError):
    """Raised when a relation type cannot be found."""

    def __init__(self, relation_type: str) -> None:
        """
        Initialize the exception.

        Args:
            relation_type: Name of the relation type that could not be found
        """
        super().__init__(
            f"Relation type '{relation_type}' not found",
            {"relation_type": relation_type}
        )


class ValidationError(GraphContextError):
    """Raised when entity or relation validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraint: Optional[str] = None
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Description of the validation error
            field: Optional name of the field that failed validation
            value: Optional value that failed validation
            constraint: Optional description of the constraint that was violated
        """
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        if constraint:
            details["constraint"] = constraint

        super().__init__(message, details)


class DuplicateEntityError(GraphContextError):
    """Raised when attempting to create an entity that already exists."""

    def __init__(self, entity_id: str, entity_type: Optional[str] = None) -> None:
        """
        Initialize the exception.

        Args:
            entity_id: ID of the duplicate entity
            entity_type: Optional type of the duplicate entity
        """
        details = {"entity_id": entity_id}
        if entity_type:
            details["entity_type"] = entity_type
        super().__init__(
            f"Entity with ID '{entity_id}'" +
            (f" and type '{entity_type}'" if entity_type else "") +
            " already exists",
            details
        )


class DuplicateRelationError(GraphContextError):
    """Raised when attempting to create a relation that already exists."""

    def __init__(self, relation_id: str, relation_type: Optional[str] = None) -> None:
        """
        Initialize the exception.

        Args:
            relation_id: ID of the duplicate relation
            relation_type: Optional type of the duplicate relation
        """
        details = {"relation_id": relation_id}
        if relation_type:
            details["relation_type"] = relation_type
        super().__init__(
            f"Relation with ID '{relation_id}'" +
            (f" and type '{relation_type}'" if relation_type else "") +
            " already exists",
            details
        )


class SchemaError(GraphContextError):
    """Raised when there are schema-related issues."""

    def __init__(
        self,
        message: str,
        schema_type: Optional[str] = None,
        field: Optional[str] = None
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Description of the schema error
            schema_type: Optional type name in the schema
            field: Optional field name that caused the error
        """
        details = {}
        if schema_type:
            details["schema_type"] = schema_type
        if field:
            details["field"] = field

        super().__init__(message, details)


class TransactionError(GraphContextError):
    """Raised when there are transaction-related issues."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        state: Optional[str] = None
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Description of the transaction error
            operation: Optional name of the operation that failed
            state: Optional state of the transaction when the error occurred
        """
        details = {}
        if operation:
            details["operation"] = operation
        if state:
            details["state"] = state

        super().__init__(message, details)


class QueryError(GraphContextError):
    """Raised when there are query-related issues."""

    def __init__(
        self,
        message: str,
        query_spec: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Description of the query error
            query_spec: Optional query specification that caused the error
        """
        details = {}
        if query_spec:
            details["query_spec"] = query_spec

        super().__init__(message, details)


class BackendError(GraphContextError):
    """Raised when there are backend-related issues."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        backend_error: Optional[Exception] = None
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Description of the backend error
            operation: Optional name of the operation that failed
            backend_error: Optional original backend exception
        """
        details = {}
        if operation:
            details["operation"] = operation
        if backend_error:
            details["backend_error"] = str(backend_error)

        super().__init__(message, details)