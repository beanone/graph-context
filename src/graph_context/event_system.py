"""
Event system for graph operations.

This module provides a minimal pub/sub system for graph operations, allowing features
to react to changes in the graph without coupling to specific implementations.
"""

from enum import Enum
from typing import Any, Callable, Awaitable, Dict, Optional, Set
from collections import defaultdict
from pydantic import BaseModel, ConfigDict, Field
from uuid import uuid4
from datetime import datetime

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
    ENTITY_DELETE = "entity:delete"
    ENTITY_BULK_WRITE = "entity:bulk_write"
    ENTITY_BULK_DELETE = "entity:bulk_delete"

    # Relation operations
    RELATION_READ = "relation:read"
    RELATION_WRITE = "relation:write"
    RELATION_DELETE = "relation:delete"
    RELATION_BULK_WRITE = "relation:bulk_write"
    RELATION_BULK_DELETE = "relation:bulk_delete"

    # Query operations
    QUERY_EXECUTED = "query:executed"
    TRAVERSAL_EXECUTED = "traversal:executed"

    # Schema operations
    SCHEMA_MODIFIED = "schema:modified"
    TYPE_MODIFIED = "type:modified"

class EventMetadata(BaseModel):
    """Metadata for graph events.

    Contains structured information about the operation that can be used
    by event handlers for tasks like caching and logging.
    """
    # Type information
    entity_type: Optional[str] = None
    relation_type: Optional[str] = None
    affected_types: Set[str] = Field(default_factory=set)

    # Operation details
    operation_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Query/traversal metadata
    query_spec: Optional[Dict[str, Any]] = None
    traversal_spec: Optional[Dict[str, Any]] = None

    # Bulk operation metadata
    is_bulk: bool = False
    affected_count: Optional[int] = None

    model_config = ConfigDict(frozen=True)

class EventContext(BaseModel):
    """Context for a graph event.

    Contains the event type and any relevant data about the operation.
    The data field is intentionally generic to allow any operation-specific
    information to be passed without coupling to specific implementations.

    This class is immutable to ensure event data cannot be modified after creation.
    """
    event: GraphEvent
    metadata: EventMetadata
    data: Dict[str, Any]

    model_config = ConfigDict(frozen=True)

    def __init__(self, **data: Any) -> None:
        """Initialize with default empty data dict if none provided."""
        if 'data' not in data:
            data['data'] = {}
        if 'metadata' not in data:
            data['metadata'] = EventMetadata()
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

    async def emit(
        self,
        event: GraphEvent,
        metadata: Optional[EventMetadata] = None,
        **data: Any
    ) -> None:
        """Emit a graph event to all subscribers.

        Args:
            event: The graph event that occurred
            metadata: Optional event metadata. If not provided, default metadata will be created.
            **data: Any relevant data about the operation
        """
        if not self._enabled:
            return

        # If metadata is not provided, create default metadata based on event type and data
        if metadata is None:
            metadata_kwargs = {}

            # Set bulk operation metadata
            if event in {
                GraphEvent.ENTITY_BULK_WRITE,
                GraphEvent.ENTITY_BULK_DELETE,
                GraphEvent.RELATION_BULK_WRITE,
                GraphEvent.RELATION_BULK_DELETE
            }:
                metadata_kwargs["is_bulk"] = True
                if "entities" in data:
                    metadata_kwargs["affected_count"] = len(data["entities"])
                elif "relations" in data:
                    metadata_kwargs["affected_count"] = len(data["relations"])

            # Set query metadata
            if event == GraphEvent.QUERY_EXECUTED and "query_spec" in data:
                metadata_kwargs["query_spec"] = data["query_spec"]

            # Set traversal metadata
            if event == GraphEvent.TRAVERSAL_EXECUTED and "traversal_spec" in data:
                metadata_kwargs["traversal_spec"] = data["traversal_spec"]

            # Set type information
            if "entity_type" in data:
                metadata_kwargs["entity_type"] = data["entity_type"]
            if "relation_type" in data:
                metadata_kwargs["relation_type"] = data["relation_type"]
            if "affected_types" in data:
                metadata_kwargs["affected_types"] = set(data["affected_types"])

            metadata = EventMetadata(**metadata_kwargs)

        context = EventContext(event=event, metadata=metadata, data=data)

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