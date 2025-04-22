"""
Event system for graph operations.

This module provides a minimal pub/sub system for graph operations, allowing features
to react to changes in the graph without coupling to specific implementations.
"""

from enum import Enum
from typing import Any, Callable, Awaitable, Dict
from collections import defaultdict
from pydantic import BaseModel, ConfigDict

# Type alias for event handlers
EventHandler = Callable[["EventContext"], Awaitable[None]]

class GraphEvent(str, Enum):
    """Core graph operation events.

    These events represent the fundamental operations that can occur in the graph,
    without making assumptions about how they will be used.
    """
    # Entity operations
    ENTITY_READ = "entity:read"
    ENTITY_WRITE = "entity:write"

    # Relation operations
    RELATION_READ = "relation:read"
    RELATION_WRITE = "relation:write"

    # Query operations
    QUERY_EXECUTED = "query:executed"

    # Schema operations
    SCHEMA_MODIFIED = "schema:modified"

class EventContext(BaseModel):
    """Context for a graph event.

    Contains the event type and any relevant data about the operation.
    The data field is intentionally generic to allow any operation-specific
    information to be passed without coupling to specific implementations.

    This class is immutable to ensure event data cannot be modified after creation.
    """
    event: GraphEvent
    data: Dict[str, Any]

    model_config = ConfigDict(frozen=True)

    def __init__(self, **data: Any) -> None:
        """Initialize with default empty data dict if none provided."""
        if 'data' not in data:
            data['data'] = {}
        super().__init__(**data)

class EventSystem:
    """Simple pub/sub system for graph operations.

    Provides mechanisms to subscribe to and emit events for graph operations
    without making assumptions about how those events will be used.
    """

    def __init__(self) -> None:
        """Initialize the event system."""
        self._handlers: dict[GraphEvent, list[EventHandler]] = defaultdict(list)
        self._enabled = True

    async def subscribe(self, event: GraphEvent, handler: EventHandler) -> None:
        """Subscribe to a specific graph event.

        Args:
            event: The graph event to subscribe to
            handler: Async function to call when the event occurs
        """
        self._handlers[event].append(handler)

    async def unsubscribe(self, event: GraphEvent, handler: EventHandler) -> None:
        """Unsubscribe from a specific graph event.

        Args:
            event: The graph event to unsubscribe from
            handler: The handler to remove

        Note:
            If the handler is not found, this operation is a no-op.
        """
        try:
            self._handlers[event].remove(handler)
        except ValueError:
            pass  # Handler wasn't registered, ignore

    async def emit(self, event: GraphEvent, **data: Any) -> None:
        """Emit a graph event to all subscribers.

        Args:
            event: The graph event that occurred
            **data: Any relevant data about the operation
        """
        if not self._enabled:
            return

        context = EventContext(event=event, data=data)

        # Execute handlers sequentially to maintain ordering
        # This is important for operations that might depend on each other
        for handler in self._handlers[event]:
            try:
                await handler(context)
            except Exception:
                # Log error but continue processing handlers
                # This prevents one handler from breaking others
                # TODO: Add proper error logging
                continue

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission.

        This can be useful during bulk operations or when
        temporary suppression of events is needed.
        """
        self._enabled = False