"""Tests for custom exceptions."""

from graph_context.exceptions import (
    BackendError,
    DuplicateEntityError,
    DuplicateRelationError,
    EntityNotFoundError,
    EntityTypeNotFoundError,
    GraphContextError,
    QueryError,
    RelationNotFoundError,
    RelationTypeNotFoundError,
    SchemaError,
    TransactionError,
    ValidationError,
)


def test_graph_context_error():
    """Test GraphContextError base exception."""
    msg = "Base error message"
    exc = GraphContextError(msg)
    assert str(exc) == msg
    assert isinstance(exc, Exception)

    # Test with details
    details = {"key": "value"}
    exc = GraphContextError(msg, details)
    assert exc.details == details


def test_validation_error():
    """Test ValidationError exception."""
    msg = "Invalid value"
    exc = ValidationError(msg)
    assert str(exc) == msg
    assert isinstance(exc, GraphContextError)

    # Test with all optional parameters
    exc = ValidationError(
        msg,
        field="name",
        value="test",
        constraint="min_length"
    )
    assert exc.details["field"] == "name"
    assert exc.details["value"] == "test"
    assert exc.details["constraint"] == "min_length"


def test_entity_not_found_error():
    """Test EntityNotFoundError exception."""
    entity_id = "123"
    entity_type = "Person"
    exc = EntityNotFoundError(entity_id, entity_type)
    assert str(exc) == f"Entity with ID '{entity_id}' and type '{entity_type}' not found"
    assert isinstance(exc, GraphContextError)

    # Test without entity_type
    exc = EntityNotFoundError(entity_id)
    assert str(exc) == f"Entity with ID '{entity_id}' not found"


def test_entity_type_not_found_error():
    """Test EntityTypeNotFoundError exception."""
    entity_type = "Person"
    exc = EntityTypeNotFoundError(entity_type)
    assert str(exc) == f"Entity type '{entity_type}' not found"
    assert isinstance(exc, GraphContextError)


def test_relation_not_found_error():
    """Test RelationNotFoundError exception."""
    relation_id = "456"
    relation_type = "KNOWS"
    exc = RelationNotFoundError(relation_id, relation_type)
    assert str(exc) == f"Relation with ID '{relation_id}' and type '{relation_type}' not found"
    assert isinstance(exc, GraphContextError)

    # Test with all optional parameters
    exc = RelationNotFoundError(
        relation_id,
        relation_type="KNOWS",
        from_entity="123",
        to_entity="456"
    )
    assert exc.details["relation_id"] == relation_id
    assert exc.details["relation_type"] == "KNOWS"
    assert exc.details["from_entity"] == "123"
    assert exc.details["to_entity"] == "456"


def test_relation_type_not_found_error():
    """Test RelationTypeNotFoundError exception."""
    relation_type = "KNOWS"
    exc = RelationTypeNotFoundError(relation_type)
    assert str(exc) == f"Relation type '{relation_type}' not found"
    assert isinstance(exc, GraphContextError)


def test_duplicate_entity_error():
    """Test DuplicateEntityError exception."""
    entity_id = "123"
    entity_type = "Person"
    exc = DuplicateEntityError(entity_id, entity_type)
    assert str(exc) == f"Entity with ID '{entity_id}' and type '{entity_type}' already exists"
    assert isinstance(exc, GraphContextError)


def test_duplicate_relation_error():
    """Test DuplicateRelationError exception."""
    relation_id = "456"
    relation_type = "KNOWS"
    exc = DuplicateRelationError(relation_id, relation_type)
    assert str(exc) == f"Relation with ID '{relation_id}' and type '{relation_type}' already exists"
    assert isinstance(exc, GraphContextError)


def test_transaction_error():
    """Test TransactionError exception."""
    msg = "Transaction failed"
    exc = TransactionError(msg)
    assert str(exc) == msg
    assert isinstance(exc, GraphContextError)

    # Test with optional parameters
    exc = TransactionError(msg, operation="commit", state="pending")
    assert exc.details["operation"] == "commit"
    assert exc.details["state"] == "pending"


def test_backend_error():
    """Test BackendError exception."""
    msg = "Backend operation failed"
    exc = BackendError(msg)
    assert str(exc) == msg
    assert isinstance(exc, GraphContextError)

    # Test with optional parameters
    backend_error = ValueError("Original error")
    exc = BackendError(msg, operation="query", backend_error=backend_error)
    assert exc.details["operation"] == "query"
    assert str(exc.details["backend_error"]) == str(backend_error)  # Compare string representations


def test_schema_error():
    """Test SchemaError exception."""
    msg = "Schema validation failed"
    exc = SchemaError(msg)
    assert str(exc) == msg
    assert isinstance(exc, GraphContextError)

    # Test with optional parameters
    exc = SchemaError(msg, schema_type="entity", field="name")
    assert exc.details["schema_type"] == "entity"
    assert exc.details["field"] == "name"


def test_query_error():
    """Test QueryError exception."""
    msg = "Query execution failed"
    exc = QueryError(msg)
    assert str(exc) == msg
    assert isinstance(exc, GraphContextError)

    # Test with query_spec
    query_spec = {"type": "entity", "filter": {"name": "test"}}
    exc = QueryError(msg, query_spec=query_spec)
    assert exc.details["query_spec"] == query_spec