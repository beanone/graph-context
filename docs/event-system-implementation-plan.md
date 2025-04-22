# Event System Implementation Plan

## Rationale

The event system serves as a lightweight foundation for implementing cross-cutting concerns in the Graph Context component. This design is driven by several key principles:

1. **Minimal Core, Maximum Flexibility**
   - Simple pub/sub mechanism that can support any feature
   - No built-in assumptions about handlers or use cases
   - Easy to extend without modifying core code

2. **Performance First**
   - Minimal overhead for event emission
   - No complex hierarchies or routing
   - Efficient handler execution

3. **Clear Integration Points**
   - Direct integration with base context
   - Simple event emission in core methods
   - Easy to add new events and handlers

4. **Focus on Core Operations**
   - Entity operations (read/write)
   - Relation operations (read/write)
   - Query execution
   - Schema modifications
   - No assumptions about how events will be used

5. **Future Extensibility**
   - Foundation for additional features
   - No knowledge of specific features (caching, logging, etc.)
   - No breaking changes needed for extensions

## Expected Folder Structure

```
graph-context/
├── src/
│   └── graph_context/
│       ├── __init__.py          # Package exports
│       ├── interface.py         # Core interfaces
│       ├── context_base.py      # Base implementations
│       ├── exceptions.py        # Error definitions
│       ├── event_system.py      # Event system implementation
│       └── types/              # Type definitions
│           ├── __init__.py
│           ├── type_base.py
│           └── validators.py
└── tests/
    └── graph_context/
        ├── __init__.py
        ├── test_event_system.py  # Event system tests
        └── types/
            ├── __init__.py
            ├── test_type_base.py
            └── test_validators.py
```

The event system is intentionally kept in a single file (`event_system.py`) to maintain simplicity and reduce complexity. This structure:
- Preserves the existing codebase organization
- Minimizes the impact on the current implementation
- Makes it easy to understand the event system's scope
- Provides clear separation of concerns

## Implementation

### Core Event System
```python
# src/graph_context/event_system.py

from enum import Enum
from typing import Any, Callable, Awaitable, TypeVar
from collections import defaultdict
from pydantic import BaseModel

T = TypeVar('T')

class GraphEvent(str, Enum):
    """Base event types - extendable as needed."""
    ENTITY_READ = "entity:read"
    ENTITY_MODIFIED = "entity:modified"
    RELATION_READ = "relation:read"
    RELATION_MODIFIED = "relation:modified"
    QUERY_EXECUTED = "query:executed"
    SCHEMA_MODIFIED = "schema:modified"

class EventContext(BaseModel):
    """Minimal context with essential metadata."""
    event: GraphEvent
    data: dict[str, Any] = {}  # Flexible payload for any event-specific data

EventHandler = Callable[[EventContext], Awaitable[None]]

class EventSystem:
    """Simple pub/sub system for graph events."""

    def __init__(self):
        self._handlers: dict[GraphEvent, list[EventHandler]] = defaultdict(list)
        self._enabled = True

    async def subscribe(self, event: GraphEvent, handler: EventHandler) -> None:
        """Subscribe to an event."""
        self._handlers[event].append(handler)

    async def unsubscribe(self, event: GraphEvent, handler: EventHandler) -> None:
        """Unsubscribe from an event."""
        self._handlers[event].remove(handler)

    async def emit(self, event: GraphEvent, **data) -> None:
        """Emit an event with data."""
        if not self._enabled:
            return

        context = EventContext(event=event, data=data)
        for handler in self._handlers[event]:
            await handler(context)

    def disable(self) -> None:
        """Disable event emission."""
        self._enabled = False

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True
```

### Integration with Base Context

```python
# src/graph_context/context_base.py

from .event_system import EventSystem, GraphEvent

class BaseGraphContext:
    def __init__(self):
        self._events = EventSystem()

    # Example integration points
    async def get_entity(self, entity_id: str):
        result = await self._get_entity(entity_id)
        await self._events.emit(
            GraphEvent.ENTITY_READ,
            entity_id=entity_id,
            result=result
        )
        return result

    async def update_entity(self, entity_id: str, data: dict):
        result = await self._update_entity(entity_id, data)
        await self._events.emit(
            GraphEvent.ENTITY_MODIFIED,
            entity_id=entity_id,
            changes=data
        )
        return result
```

## Testing Strategy

### Unit Tests
```python
# tests/test_event_system.py

import pytest
from graph_context.event_system import EventSystem, GraphEvent, EventContext

async def test_event_subscription():
    events = EventSystem()
    received = []

    async def handler(ctx: EventContext):
        received.append(ctx)

    await events.subscribe(GraphEvent.ENTITY_READ, handler)
    await events.emit(GraphEvent.ENTITY_READ, entity_id="123")

    assert len(received) == 1
    assert received[0].event == GraphEvent.ENTITY_READ
    assert received[0].data["entity_id"] == "123"

async def test_event_unsubscription():
    events = EventSystem()
    received = []

    async def handler(ctx: EventContext):
        received.append(ctx)

    await events.subscribe(GraphEvent.ENTITY_READ, handler)
    await events.unsubscribe(GraphEvent.ENTITY_READ, handler)
    await events.emit(GraphEvent.ENTITY_READ, entity_id="123")

    assert len(received) == 0

async def test_disabled_events():
    events = EventSystem()
    received = []

    async def handler(ctx: EventContext):
        received.append(ctx)

    await events.subscribe(GraphEvent.ENTITY_READ, handler)
    events.disable()
    await events.emit(GraphEvent.ENTITY_READ, entity_id="123")

    assert len(received) == 0
```

## Success Criteria

1. **Functionality**
   - Events are properly emitted and received
   - Event handlers execute in order
   - Event system can be enabled/disabled
   - Event context contains all necessary data

2. **Performance**
   - Minimal overhead for event emission
   - Efficient handler execution
   - Low memory footprint

3. **Code Quality**
   - Clear and concise implementation
   - Comprehensive test coverage
   - Type safety throughout
   - Proper error handling

## Documentation

### Usage Example
```python
# Example: Creating a cache invalidation handler
async def cache_invalidation_handler(ctx: EventContext):
    if ctx.event == GraphEvent.ENTITY_MODIFIED:
        entity_id = ctx.data["entity_id"]
        await cache.invalidate(f"entity:{entity_id}")

# Registering the handler
await context._events.subscribe(
    GraphEvent.ENTITY_MODIFIED,
    cache_invalidation_handler
)
```

### Adding New Events
To extend the event system with new event types:

1. Add new event to `GraphEvent` enum:
```python
class GraphEvent(str, Enum):
    # ... existing events ...
    NEW_EVENT = "category:action"
```

2. Emit the new event where needed:
```python
await self._events.emit(
    GraphEvent.NEW_EVENT,
    relevant_data="value"
)
```

3. Subscribe to the new event:
```python
async def new_event_handler(ctx: EventContext):
    data = ctx.data["relevant_data"]
    # Handle the event

await events.subscribe(GraphEvent.NEW_EVENT, new_event_handler)
```

## Implementation Schedule

1. Core Implementation (1 day)
   - Event system class
   - Basic event types
   - Context model

2. Testing (1 day)
   - Unit test suite
   - Integration tests
   - Performance validation

3. Integration (1 day)
   - Base context integration
   - Documentation
   - Example handlers

## Next Steps

1. Implement the core event system
2. Write comprehensive tests
3. Integrate with base context
4. Document usage patterns
5. Move forward with caching implementation