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
    exc = ValidationError(msg, field="name", value="test", constraint="min_length")
    assert exc.details["field"] == "name"
    assert exc.details["value"] == "test"
    assert exc.details["constraint"] == "min_length"

    # Test with some optional parameters omitted
    exc = ValidationError(msg, field="name")
    assert exc.details["field"] == "name"
    assert "value" not in exc.details
    assert "constraint" not in exc.details


def test_entity_not_found_error():
    """Test EntityNotFoundError exception."""
    entity_id = "123"
    entity_type = "Person"

    # Test with both parameters
    exc = EntityNotFoundError(entity_id, entity_type)
    assert str(exc) == f"Entity with ID '{entity_id}' and type '{entity_type}' not found"
    assert isinstance(exc, GraphContextError)
    assert exc.details["entity_id"] == entity_id
    assert exc.details["entity_type"] == entity_type

    # Test without entity_type
    exc = EntityNotFoundError(entity_id)
    assert str(exc) == f"Entity with ID '{entity_id}' not found"
    assert exc.details["entity_id"] == entity_id
    assert "entity_type" not in exc.details

    # Test with None entity_type
    exc = EntityNotFoundError(entity_id, None)
    assert str(exc) == f"Entity with ID '{entity_id}' not found"
    assert exc.details["entity_id"] == entity_id
    assert "entity_type" not in exc.details


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

    # Test with minimal parameters
    exc = RelationNotFoundError(relation_id)
    assert str(exc) == f"Relation with ID '{relation_id}' not found"
    assert isinstance(exc, GraphContextError)
    assert exc.details["relation_id"] == relation_id

    # Test with relation_type only
    exc = RelationNotFoundError(relation_id, relation_type)
    assert str(exc) == f"Relation with ID '{relation_id}' and type '{relation_type}' not found"
    assert exc.details["relation_id"] == relation_id
    assert exc.details["relation_type"] == relation_type

    # Test with all optional parameters
    exc = RelationNotFoundError(relation_id, relation_type="KNOWS", from_entity="123", to_entity="456")
    assert exc.details["relation_id"] == relation_id
    assert exc.details["relation_type"] == "KNOWS"
    assert exc.details["from_entity"] == "123"
    assert exc.details["to_entity"] == "456"

    # Test with None values
    exc = RelationNotFoundError(relation_id, None)
    assert str(exc) == f"Relation with ID '{relation_id}' not found"
    assert "relation_type" not in exc.details


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

    # Test with both parameters
    exc = DuplicateEntityError(entity_id, entity_type)
    assert str(exc) == f"Entity with ID '{entity_id}' and type '{entity_type}' already exists"
    assert isinstance(exc, GraphContextError)

    # Test without entity_type
    exc = DuplicateEntityError(entity_id)
    assert str(exc) == f"Entity with ID '{entity_id}' already exists"

    # Test with None entity_type
    exc = DuplicateEntityError(entity_id, None)
    assert str(exc) == f"Entity with ID '{entity_id}' already exists"


def test_duplicate_relation_error():
    """Test DuplicateRelationError exception."""
    relation_id = "456"
    relation_type = "KNOWS"

    # Test with both parameters
    exc = DuplicateRelationError(relation_id, relation_type)
    assert str(exc) == f"Relation with ID '{relation_id}' and type '{relation_type}' already exists"
    assert isinstance(exc, GraphContextError)

    # Test without relation_type
    exc = DuplicateRelationError(relation_id)
    assert str(exc) == f"Relation with ID '{relation_id}' already exists"

    # Test with None relation_type
    exc = DuplicateRelationError(relation_id, None)
    assert str(exc) == f"Relation with ID '{relation_id}' already exists"


def test_transaction_error():
    """Test TransactionError exception."""
    msg = "Transaction failed"

    # Test with message only
    exc = TransactionError(msg)
    assert str(exc) == msg
    assert isinstance(exc, GraphContextError)
    assert not exc.details

    # Test with all optional parameters
    exc = TransactionError(msg, operation="commit", state="pending")
    assert exc.details["operation"] == "commit"
    assert exc.details["state"] == "pending"

    # Test with operation only
    exc = TransactionError(msg, operation="commit")
    assert exc.details["operation"] == "commit"
    assert "state" not in exc.details

    # Test with state only
    exc = TransactionError(msg, state="pending")
    assert exc.details["state"] == "pending"
    assert "operation" not in exc.details


def test_backend_error():
    """Test BackendError exception."""
    msg = "Backend operation failed"

    # Test with message only
    exc = BackendError(msg)
    assert str(exc) == msg
    assert isinstance(exc, GraphContextError)
    assert not exc.details

    # Test with all optional parameters
    backend_error = ValueError("Original error")
    exc = BackendError(msg, operation="query", backend_error=backend_error)
    assert exc.details["operation"] == "query"
    assert str(exc.details["backend_error"]) == str(backend_error)

    # Test with operation only
    exc = BackendError(msg, operation="query")
    assert exc.details["operation"] == "query"
    assert "backend_error" not in exc.details

    # Test with backend_error only
    exc = BackendError(msg, backend_error=backend_error)
    assert str(exc.details["backend_error"]) == str(backend_error)
    assert "operation" not in exc.details


def test_schema_error():
    """Test SchemaError exception."""
    msg = "Schema validation failed"

    # Test with message only
    exc = SchemaError(msg)
    assert str(exc) == msg
    assert isinstance(exc, GraphContextError)
    assert not exc.details

    # Test with all optional parameters
    exc = SchemaError(msg, schema_type="entity", field="name")
    assert exc.details["schema_type"] == "entity"
    assert exc.details["field"] == "name"

    # Test with schema_type only
    exc = SchemaError(msg, schema_type="entity")
    assert exc.details["schema_type"] == "entity"
    assert "field" not in exc.details

    # Test with field only
    exc = SchemaError(msg, field="name")
    assert exc.details["field"] == "name"
    assert "schema_type" not in exc.details


def test_query_error():
    """Test QueryError exception."""
    msg = "Query execution failed"

    # Test with message only
    exc = QueryError(msg)
    assert str(exc) == msg
    assert isinstance(exc, GraphContextError)
    assert not exc.details

    # Test with query_spec
    query_spec = {"type": "entity", "filter": {"name": "test"}}
    exc = QueryError(msg, query_spec=query_spec)
    assert exc.details["query_spec"] == query_spec

    # Test with None query_spec
    exc = QueryError(msg, query_spec=None)
    assert not exc.details
