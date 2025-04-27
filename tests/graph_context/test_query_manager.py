"""
Tests for the QueryManager class.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from graph_context.event_system import GraphEvent
from graph_context.manager import QueryManager
from graph_context.types.type_base import Entity, QuerySpec, TraversalSpec


@pytest.fixture
def mock_store():
    """Mock GraphStore for testing."""
    store = AsyncMock()
    store.query = AsyncMock()
    store.traverse = AsyncMock()
    return store


@pytest.fixture
def mock_events():
    """Mock EventSystem for testing."""
    events = AsyncMock()
    events.emit = AsyncMock()
    return events


@pytest.fixture
def query_manager(mock_store, mock_events):
    """QueryManager instance for testing."""
    return QueryManager(mock_store, mock_events)


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [MagicMock(spec=Entity), MagicMock(spec=Entity)]


class TestQueryManager:
    """Test cases for QueryManager class."""

    def test_init(self, mock_store, mock_events):
        """Test QueryManager initialization."""
        manager = QueryManager(mock_store, mock_events)
        assert manager._store is mock_store
        assert manager._events is mock_events

    @pytest.mark.asyncio
    async def test_query_success(
        self, query_manager, mock_store, mock_events, sample_entities
    ):
        """Test query method successful execution."""
        query_spec = MagicMock(spec=QuerySpec)
        mock_store.query.return_value = sample_entities

        result = await query_manager.query(query_spec)

        mock_store.query.assert_called_once_with(query_spec)
        mock_events.emit.assert_called_once_with(
            GraphEvent.QUERY_EXECUTED, query_spec=query_spec
        )
        assert result == sample_entities

    @pytest.mark.asyncio
    async def test_traverse_success(
        self, query_manager, mock_store, mock_events, sample_entities
    ):
        """Test traverse method successful execution."""
        start_entity = "entity-123"
        traversal_spec = MagicMock(spec=TraversalSpec)
        mock_store.traverse.return_value = sample_entities

        result = await query_manager.traverse(start_entity, traversal_spec)

        mock_store.traverse.assert_called_once_with(start_entity, traversal_spec)
        mock_events.emit.assert_called_once_with(
            GraphEvent.TRAVERSAL_EXECUTED,
            start_entity=start_entity,
            traversal_spec=traversal_spec,
        )
        assert result == sample_entities
