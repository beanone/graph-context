"""Tests for custom exceptions."""
import pytest

from graph_context.exceptions import (
    GraphContextError,
    ValidationError,
    EntityNotFoundError,
    EntityTypeNotFoundError,
    RelationNotFoundError,
    RelationTypeNotFoundError,
    DuplicateEntityError,
    DuplicateRelationError,
    TransactionError,
    BackendError
)


def test_graph_context_error():
    """Test GraphContextError base exception."""
    msg = "Base error message"
    exc = GraphContextError(msg)
    assert str(exc) == msg
    assert isinstance(exc, Exception)


def test_validation_error():
    """Test ValidationError exception."""
    msg = "Invalid value"
    exc = ValidationError(msg)
    assert str(exc) == msg
    assert isinstance(exc, GraphContextError)


def test_entity_not_found_error():
    """Test EntityNotFoundError exception."""
    entity_id = "123"
    entity_type = "Person"
    exc = EntityNotFoundError(entity_id, entity_type)
    assert str(exc) == f"Entity with ID '{entity_id}' and type '{entity_type}' not found"
    assert isinstance(exc, GraphContextError)


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


def test_backend_error():
    """Test BackendError exception."""
    msg = "Backend operation failed"
    exc = BackendError(msg)
    assert str(exc) == msg
    assert isinstance(exc, GraphContextError)