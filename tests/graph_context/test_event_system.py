"""Unit tests for the event system."""

import pytest
from pydantic import ValidationError
from typing import List

from graph_context.event_system import (
    EventSystem,
    GraphEvent,
    EventContext,
    EventHandler
)

@pytest.fixture
def event_system() -> EventSystem:
    """Create a fresh event system for each test."""
    return EventSystem()

@pytest.fixture
def received_events() -> List[EventContext]:
    """Track events received by handlers."""
    return []

@pytest.fixture
def handler(received_events: List[EventContext]) -> EventHandler:
    """Create a handler that records received events."""
    async def _handler(context: EventContext) -> None:
        received_events.append(context)
    return _handler

@pytest.fixture
def error_handler() -> EventHandler:
    """Create a handler that raises an exception."""
    async def _handler(_: EventContext) -> None:
        raise RuntimeError("Test error")
    return _handler

class TestGraphEvent:
    """Tests for the GraphEvent enum."""

    def test_event_values(self):
        """Test that all events have the expected string values."""
        assert GraphEvent.ENTITY_READ.value == "entity:read"
        assert GraphEvent.ENTITY_WRITE.value == "entity:write"
        assert GraphEvent.RELATION_READ.value == "relation:read"
        assert GraphEvent.RELATION_WRITE.value == "relation:write"
        assert GraphEvent.QUERY_EXECUTED.value == "query:executed"
        assert GraphEvent.SCHEMA_MODIFIED.value == "schema:modified"

class TestEventContext:
    """Tests for the EventContext class."""

    def test_create_with_required_fields(self):
        """Test creating context with only required fields."""
        context = EventContext(event=GraphEvent.ENTITY_READ)
        assert context.event == GraphEvent.ENTITY_READ
        assert context.data == {}

    def test_create_with_data(self):
        """Test creating context with additional data."""
        data = {"entity_id": "123", "type": "user"}
        context = EventContext(event=GraphEvent.ENTITY_READ, data=data)
        assert context.event == GraphEvent.ENTITY_READ
        assert context.data == data

    def test_context_immutable_after_creation(self):
        """Test that context and its data cannot be modified after creation."""
        initial_data = {"id": "123"}
        context = EventContext(event=GraphEvent.ENTITY_READ, data=initial_data.copy())

        # Test event immutability
        with pytest.raises(ValidationError) as exc_info:
            context.event = GraphEvent.ENTITY_WRITE
        assert "Instance is frozen" in str(exc_info.value)

        # Test data immutability
        with pytest.raises(ValidationError) as exc_info:
            context.data = {"new": "data"}
        assert "Instance is frozen" in str(exc_info.value)

        # Verify the original data is unchanged
        assert context.data == initial_data
        assert initial_data == {"id": "123"}  # Ensure original dict wasn't modified

class TestEventSystem:
    """Tests for the EventSystem class."""

    def test_initial_state(self, event_system: EventSystem):
        """Test initial state of event system."""
        assert event_system._enabled is True
        assert len(event_system._handlers) == 0

    @pytest.mark.asyncio
    async def test_subscribe_and_emit(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: List[EventContext]
    ):
        """Test basic subscribe and emit functionality."""
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler)
        await event_system.emit(
            GraphEvent.ENTITY_READ,
            entity_id="123"
        )

        assert len(received_events) == 1
        assert received_events[0].event == GraphEvent.ENTITY_READ
        assert received_events[0].data == {"entity_id": "123"}

    @pytest.mark.asyncio
    async def test_multiple_handlers(
        self,
        event_system: EventSystem,
        received_events: List[EventContext]
    ):
        """Test multiple handlers for same event."""
        async def handler1(ctx: EventContext) -> None:
            received_events.append(ctx)

        async def handler2(ctx: EventContext) -> None:
            received_events.append(ctx)

        await event_system.subscribe(GraphEvent.ENTITY_READ, handler1)
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler2)
        await event_system.emit(GraphEvent.ENTITY_READ)

        assert len(received_events) == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: List[EventContext]
    ):
        """Test unsubscribing a handler."""
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler)
        await event_system.unsubscribe(GraphEvent.ENTITY_READ, handler)
        await event_system.emit(GraphEvent.ENTITY_READ)

        assert len(received_events) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(
        self,
        event_system: EventSystem,
        handler: EventHandler
    ):
        """Test unsubscribing a handler that wasn't subscribed."""
        # Should not raise an exception
        await event_system.unsubscribe(GraphEvent.ENTITY_READ, handler)

    @pytest.mark.asyncio
    async def test_disable_enable(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: List[EventContext]
    ):
        """Test disabling and enabling event emission."""
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler)

        event_system.disable()
        await event_system.emit(GraphEvent.ENTITY_READ)
        assert len(received_events) == 0

        event_system.enable()
        await event_system.emit(GraphEvent.ENTITY_READ)
        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_handler_error_isolation(
        self,
        event_system: EventSystem,
        error_handler: EventHandler,
        handler: EventHandler,
        received_events: List[EventContext]
    ):
        """Test that handler errors don't affect other handlers."""
        await event_system.subscribe(GraphEvent.ENTITY_READ, error_handler)
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler)

        # Should not raise an exception and second handler should still be called
        await event_system.emit(GraphEvent.ENTITY_READ)
        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_handler_execution_order(
        self,
        event_system: EventSystem,
        received_events: List[EventContext]
    ):
        """Test that handlers are executed in subscription order."""
        order: List[int] = []

        async def handler1(_: EventContext) -> None:
            order.append(1)

        async def handler2(_: EventContext) -> None:
            order.append(2)

        await event_system.subscribe(GraphEvent.ENTITY_READ, handler1)
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler2)
        await event_system.emit(GraphEvent.ENTITY_READ)

        assert order == [1, 2]

    @pytest.mark.asyncio
    async def test_different_events(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: List[EventContext]
    ):
        """Test that handlers only receive their subscribed events."""
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler)

        # Emit different event
        await event_system.emit(GraphEvent.ENTITY_WRITE)
        assert len(received_events) == 0

        # Emit subscribed event
        await event_system.emit(GraphEvent.ENTITY_READ)
        assert len(received_events) == 1