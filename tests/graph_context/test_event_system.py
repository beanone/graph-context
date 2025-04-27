"""Unit tests for the event system."""

from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from graph_context.event_system import (
    EventContext,
    EventHandler,
    EventMetadata,
    EventSystem,
    GraphEvent,
)


@pytest.fixture
def event_system() -> EventSystem:
    """Create a fresh event system for each test."""
    return EventSystem()


@pytest.fixture
def received_events() -> list[EventContext]:
    """Track events received by handlers."""
    return []


@pytest.fixture
def handler(received_events: list[EventContext]) -> EventHandler:
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
        assert GraphEvent.ENTITY_DELETE.value == "entity:delete"
        assert GraphEvent.ENTITY_BULK_WRITE.value == "entity:bulk_write"
        assert GraphEvent.ENTITY_BULK_DELETE.value == "entity:bulk_delete"
        assert GraphEvent.RELATION_READ.value == "relation:read"
        assert GraphEvent.RELATION_WRITE.value == "relation:write"
        assert GraphEvent.RELATION_DELETE.value == "relation:delete"
        assert GraphEvent.RELATION_BULK_WRITE.value == "relation:bulk_write"
        assert GraphEvent.RELATION_BULK_DELETE.value == "relation:bulk_delete"
        assert GraphEvent.QUERY_EXECUTED.value == "query:executed"
        assert GraphEvent.TRAVERSAL_EXECUTED.value == "traversal:executed"
        assert GraphEvent.SCHEMA_MODIFIED.value == "schema:modified"
        assert GraphEvent.TYPE_MODIFIED.value == "type:modified"


class TestEventMetadata:
    """Tests for the EventMetadata class."""

    def test_create_with_defaults(self):
        """Test creating metadata with default values."""
        metadata = EventMetadata()
        assert metadata.entity_type is None
        assert metadata.relation_type is None
        assert metadata.affected_types == set()
        assert isinstance(metadata.operation_id, str)
        assert isinstance(UUID(metadata.operation_id), UUID)  # Valid UUID
        assert isinstance(metadata.timestamp, datetime)
        assert metadata.query_spec is None
        assert metadata.traversal_spec is None
        assert metadata.is_bulk is False
        assert metadata.affected_count is None

    def test_create_with_all_fields(self):
        """Test creating metadata with all fields specified."""
        metadata = EventMetadata(
            entity_type="person",
            relation_type="knows",
            affected_types={"person", "organization"},
            operation_id="test-op-id",
            timestamp=datetime(2024, 1, 1),
            query_spec={"type": "person"},
            traversal_spec={"depth": 2},
            is_bulk=True,
            affected_count=10,
        )
        assert metadata.entity_type == "person"
        assert metadata.relation_type == "knows"
        assert metadata.affected_types == {"person", "organization"}
        assert metadata.operation_id == "test-op-id"
        assert metadata.timestamp == datetime(2024, 1, 1)
        assert metadata.query_spec == {"type": "person"}
        assert metadata.traversal_spec == {"depth": 2}
        assert metadata.is_bulk is True
        assert metadata.affected_count == 10

    def test_metadata_immutable(self):
        """Test that metadata is immutable after creation."""
        metadata = EventMetadata(entity_type="person")

        with pytest.raises(ValidationError) as exc_info:
            metadata.entity_type = "organization"
        assert "Instance is frozen" in str(exc_info.value)


class TestEventContext:
    """Tests for the EventContext class."""

    def test_create_with_required_fields(self):
        """Test creating context with only required fields."""
        context = EventContext(event=GraphEvent.ENTITY_READ)
        assert context.event == GraphEvent.ENTITY_READ
        assert context.data == {}
        assert isinstance(context.metadata, EventMetadata)

    def test_create_with_metadata_and_data(self):
        """Test creating context with metadata and data."""
        metadata = EventMetadata(entity_type="person")
        data = {"entity_id": "123", "type": "person"}
        context = EventContext(
            event=GraphEvent.ENTITY_READ, metadata=metadata, data=data
        )
        assert context.event == GraphEvent.ENTITY_READ
        assert context.metadata == metadata
        assert context.data == data

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
        received_events: list[EventContext],
    ):
        """Test basic subscribe and emit functionality."""
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler)
        await event_system.emit(GraphEvent.ENTITY_READ, entity_id="123")

        assert len(received_events) == 1
        assert received_events[0].event == GraphEvent.ENTITY_READ
        assert received_events[0].data == {"entity_id": "123"}

    @pytest.mark.asyncio
    async def test_multiple_handlers(
        self, event_system: EventSystem, received_events: list[EventContext]
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
        received_events: list[EventContext],
    ):
        """Test unsubscribing a handler."""
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler)
        await event_system.unsubscribe(GraphEvent.ENTITY_READ, handler)
        await event_system.emit(GraphEvent.ENTITY_READ)

        assert len(received_events) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(
        self, event_system: EventSystem, handler: EventHandler
    ):
        """Test unsubscribing a handler that wasn't subscribed."""
        # Should not raise an exception
        await event_system.unsubscribe(GraphEvent.ENTITY_READ, handler)

    @pytest.mark.asyncio
    async def test_disable_enable(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
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
        received_events: list[EventContext],
    ):
        """Test that handler errors don't affect other handlers."""
        await event_system.subscribe(GraphEvent.ENTITY_READ, error_handler)
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler)

        # Should not raise an exception and second handler should still be called
        await event_system.emit(GraphEvent.ENTITY_READ)
        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_handler_execution_order(
        self, event_system: EventSystem, received_events: list[EventContext]
    ):
        """Test that handlers are executed in subscription order."""
        order: list[int] = []

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
        received_events: list[EventContext],
    ):
        """Test that handlers only receive their subscribed events."""
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler)

        # Emit different event
        await event_system.emit(GraphEvent.ENTITY_WRITE)
        assert len(received_events) == 0

        # Emit subscribed event
        await event_system.emit(GraphEvent.ENTITY_READ)
        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_bulk_operation_events(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test bulk operation events with metadata."""
        await event_system.subscribe(GraphEvent.ENTITY_BULK_WRITE, handler)

        # Test with explicit metadata
        metadata = EventMetadata(
            entity_type="person",
            is_bulk=True,
            affected_count=5,
            affected_types={"person"},
        )
        entities = [{"id": str(i)} for i in range(5)]

        await event_system.emit(
            GraphEvent.ENTITY_BULK_WRITE, metadata=metadata, entities=entities
        )

        # Test with implicit metadata
        await event_system.emit(
            GraphEvent.ENTITY_BULK_WRITE,
            entities=entities,
            entity_type="person",
            affected_types={"person"},
        )

        assert len(received_events) == 2

        # Check explicit metadata event
        event1 = received_events[0]
        assert event1.event == GraphEvent.ENTITY_BULK_WRITE
        assert event1.metadata.is_bulk is True
        assert event1.metadata.affected_count == 5
        assert len(event1.data["entities"]) == 5

        # Check implicit metadata event
        event2 = received_events[1]
        assert event2.event == GraphEvent.ENTITY_BULK_WRITE
        assert event2.metadata.is_bulk is True
        assert event2.metadata.affected_count == 5
        assert len(event2.data["entities"]) == 5

    @pytest.mark.asyncio
    async def test_query_events(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test query execution events with metadata."""
        await event_system.subscribe(GraphEvent.QUERY_EXECUTED, handler)

        query_spec = {"type": "person", "filter": {"age": {"gt": 18}}}

        # Test with explicit metadata
        metadata = EventMetadata(entity_type="person", query_spec=query_spec)
        await event_system.emit(
            GraphEvent.QUERY_EXECUTED,
            metadata=metadata,
            results=[{"id": "1"}, {"id": "2"}],
        )

        # Test with implicit metadata
        await event_system.emit(
            GraphEvent.QUERY_EXECUTED,
            query_spec=query_spec,
            entity_type="person",
            results=[{"id": "1"}, {"id": "2"}],
        )

        assert len(received_events) == 2

        # Check both events
        for event in received_events:
            assert event.event == GraphEvent.QUERY_EXECUTED
            assert event.metadata.query_spec == query_spec
            assert len(event.data["results"]) == 2

    @pytest.mark.asyncio
    async def test_traversal_events(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test traversal execution events with metadata."""
        await event_system.subscribe(GraphEvent.TRAVERSAL_EXECUTED, handler)

        traversal_spec = {"start": "1", "relation": "knows", "depth": 2}

        # Test with explicit metadata
        metadata = EventMetadata(entity_type="person", traversal_spec=traversal_spec)
        await event_system.emit(
            GraphEvent.TRAVERSAL_EXECUTED,
            metadata=metadata,
            path=[{"id": "1"}, {"id": "2"}],
        )

        # Test with implicit metadata
        await event_system.emit(
            GraphEvent.TRAVERSAL_EXECUTED,
            traversal_spec=traversal_spec,
            entity_type="person",
            path=[{"id": "1"}, {"id": "2"}],
        )

        assert len(received_events) == 2

        # Check both events
        for event in received_events:
            assert event.event == GraphEvent.TRAVERSAL_EXECUTED
            assert event.metadata.traversal_spec == traversal_spec
            assert len(event.data["path"]) == 2

    @pytest.mark.asyncio
    async def test_schema_modification_events(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test schema modification events with metadata."""
        await event_system.subscribe(GraphEvent.SCHEMA_MODIFIED, handler)
        await event_system.subscribe(GraphEvent.TYPE_MODIFIED, handler)

        # Test schema modification with explicit metadata
        schema_metadata = EventMetadata(affected_types={"person", "organization"})
        await event_system.emit(
            GraphEvent.SCHEMA_MODIFIED,
            metadata=schema_metadata,
            changes={"added": ["person"], "modified": ["organization"]},
        )

        # Test schema modification with implicit metadata
        await event_system.emit(
            GraphEvent.SCHEMA_MODIFIED,
            affected_types={"person", "organization"},
            changes={"added": ["person"], "modified": ["organization"]},
        )

        # Test type modification with explicit metadata
        type_metadata = EventMetadata(entity_type="person", affected_types={"person"})
        await event_system.emit(
            GraphEvent.TYPE_MODIFIED,
            metadata=type_metadata,
            changes={"properties": {"age": "int"}},
        )

        # Test type modification with implicit metadata
        await event_system.emit(
            GraphEvent.TYPE_MODIFIED,
            entity_type="person",
            affected_types={"person"},
            changes={"properties": {"age": "int"}},
        )

        assert len(received_events) == 4

        # Check schema modification events
        for i in range(2):
            event = received_events[i]
            assert event.event == GraphEvent.SCHEMA_MODIFIED
            assert "person" in event.metadata.affected_types
            assert "organization" in event.metadata.affected_types

        # Check type modification events
        for i in range(2, 4):
            event = received_events[i]
            assert event.event == GraphEvent.TYPE_MODIFIED
            assert event.metadata.entity_type == "person"
            assert "person" in event.metadata.affected_types

    @pytest.mark.asyncio
    async def test_metadata_from_data(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test metadata creation from different data fields."""
        await event_system.subscribe(GraphEvent.RELATION_WRITE, handler)

        # Test relation_type
        await event_system.emit(
            GraphEvent.RELATION_WRITE,
            relation_type="knows",
            relation={"id": "1", "type": "knows"},
        )

        # Test affected_types without explicit metadata
        await event_system.emit(
            GraphEvent.RELATION_WRITE,
            relation_type="knows",
            affected_types={"person", "organization"},
            relation={"id": "2", "type": "knows"},
        )

        assert len(received_events) == 2

        # Check relation_type handling
        event1 = received_events[0]
        assert event1.metadata.relation_type == "knows"
        assert event1.metadata.affected_types == set()

        # Check affected_types handling
        event2 = received_events[1]
        assert event2.metadata.relation_type == "knows"
        assert event2.metadata.affected_types == {"person", "organization"}

    @pytest.mark.asyncio
    async def test_bulk_operation_with_relations(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test bulk operation events with relations."""
        await event_system.subscribe(GraphEvent.RELATION_BULK_WRITE, handler)
        relations = [
            {"id": "1", "type": "KNOWS", "from": "a", "to": "b"},
            {"id": "2", "type": "WORKS_AT", "from": "c", "to": "d"},
        ]
        await event_system.emit(
            GraphEvent.RELATION_BULK_WRITE, relations=relations, relation_type="MIXED"
        )

        assert len(received_events) == 1
        context = received_events[0]
        assert context.event == GraphEvent.RELATION_BULK_WRITE
        assert context.metadata.is_bulk is True
        assert context.metadata.affected_count == 2
        assert context.metadata.relation_type == "MIXED"

    @pytest.mark.asyncio
    async def test_metadata_with_affected_types(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test event metadata with affected types."""
        await event_system.subscribe(GraphEvent.SCHEMA_MODIFIED, handler)
        affected_types = {"Person", "Organization"}
        await event_system.emit(
            GraphEvent.SCHEMA_MODIFIED, affected_types=affected_types
        )

        assert len(received_events) == 1
        context = received_events[0]
        assert context.metadata.affected_types == affected_types

    @pytest.mark.asyncio
    async def test_query_with_empty_spec(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test query events without query spec."""
        await event_system.subscribe(GraphEvent.QUERY_EXECUTED, handler)
        await event_system.emit(GraphEvent.QUERY_EXECUTED)

        assert len(received_events) == 1
        context = received_events[0]
        assert context.metadata.query_spec is None

    @pytest.mark.asyncio
    async def test_traversal_with_empty_spec(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test traversal events without traversal spec."""
        await event_system.subscribe(GraphEvent.TRAVERSAL_EXECUTED, handler)
        await event_system.emit(GraphEvent.TRAVERSAL_EXECUTED)

        assert len(received_events) == 1
        context = received_events[0]
        assert context.metadata.traversal_spec is None

    @pytest.mark.asyncio
    async def test_enable_disable_state(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test enable/disable state transitions."""
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler)

        # Initial state (enabled)
        await event_system.emit(GraphEvent.ENTITY_READ)
        assert len(received_events) == 1

        # Disable
        event_system.disable()
        await event_system.emit(GraphEvent.ENTITY_READ)
        assert len(received_events) == 1  # No new events

        # Enable
        event_system.enable()
        await event_system.emit(GraphEvent.ENTITY_READ)
        assert len(received_events) == 2  # New event received

    @pytest.mark.asyncio
    async def test_handler_error_with_multiple_handlers(
        self,
        event_system: EventSystem,
        error_handler: EventHandler,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test error handling with multiple handlers."""
        # Add error handler first
        await event_system.subscribe(GraphEvent.ENTITY_READ, error_handler)
        # Add normal handler second
        await event_system.subscribe(GraphEvent.ENTITY_READ, handler)

        # Emit event - error handler should fail but not affect normal handler
        await event_system.emit(GraphEvent.ENTITY_READ)

        # Normal handler should still receive the event
        assert len(received_events) == 1
        assert received_events[0].event == GraphEvent.ENTITY_READ

    @pytest.mark.asyncio
    async def test_bulk_operation_without_count_data(
        self,
        event_system: EventSystem,
        handler: EventHandler,
        received_events: list[EventContext],
    ):
        """Test bulk operation events without entities or relations."""
        await event_system.subscribe(GraphEvent.ENTITY_BULK_WRITE, handler)
        await event_system.emit(GraphEvent.ENTITY_BULK_WRITE)

        assert len(received_events) == 1
        context = received_events[0]
        assert context.metadata.is_bulk is True
        assert context.metadata.affected_count is None
