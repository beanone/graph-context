# Graph Context Component

## Overview

The Graph Context component is the core abstraction layer for all graph operations in the Knowledge Graph Assisted Research IDE. It serves as the foundational interface between the high-level services and the underlying graph storage backends, providing a consistent API for graph operations regardless of the chosen storage implementation.

## Purpose

The Graph Context component fulfills several critical roles:

1. **Abstraction Layer**: Provides a unified interface for graph operations that can work with different backend implementations (Neo4j, ArangoDB, FileDB)
2. **Type Safety**: Ensures all operations conform to the defined type system and schema
3. **Data Validation**: Validates entities, relations, and their properties before persistence
4. **Query Interface**: Offers a consistent query API across different backend implementations
5. **Transaction Management**: Handles atomic operations and maintains data consistency

## Architecture

### Component Structure

```
graph-context/
├── src/
│   ├── graph_context/
│   │   ├── __init__.py
│   │   ├── interface.py        # Core GraphContext interface
│   │   ├── store.py           # GraphStore interface and factory
│   │   ├── context_base.py    # Base implementation of GraphContext
│   │   ├── event_system.py    # Event system implementation
│   │   ├── exceptions.py      # Context-specific exceptions
│   │   ├── traversal.py       # Graph traversal implementation
│   │   ├── caching/           # Caching system implementation
│   │   │   ├── __init__.py
│   │   │   ├── manager.py     # Cache manager
│   │   │   ├── store.py       # Cache store implementations
│   │   │   └── types.py       # Cache-specific types
│   │   └── types/
│   │       ├── __init__.py
│   │       ├── type_base.py   # Base type definitions
│   │       └── validators.py   # Type validation logic
│   └── __init__.py
└── tests/
    ├── graph_context/
    │   ├── __init__.py
    │   ├── test_interface.py
    │   ├── test_context_base.py
    │   ├── test_store.py
    │   ├── test_event_system.py
    │   └── test_traversal.py
    ├── caching/
    │   ├── __init__.py
    │   ├── test_manager.py
    │   └── test_store.py
    └── types/
        ├── __init__.py
        └── test_type_base.py
```

### Core Interfaces

#### GraphContext Interface

```python
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from .types.type_base import Entity, QuerySpec, Relation, TraversalSpec

T = TypeVar("T")

class GraphContext(ABC):
    """
    Abstract base class defining the core graph operations interface.
    """

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up the graph context.

        This method should be called when the context is no longer needed.
        It should close connections, clean up resources, etc.

        Raises:
            GraphContextError: If cleanup fails
        """
        pass

    @abstractmethod
    async def create_entity(self, entity_type: str, properties: dict[str, Any]) -> Entity:
        """
        Create a new entity in the graph.

        Args:
            entity_type: Type of the entity to create
            properties: Dictionary of property values

        Returns:
            The created entity

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the entity type is not defined in the schema
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Entity | None:
        """
        Retrieve an entity by ID.

        Args:
            entity_id: ID of the entity to retrieve

        Returns:
            The entity if found, None otherwise

        Raises:
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def update_entity(self, entity_id: str, properties: dict[str, Any]) -> Entity | None:
        """
        Update an existing entity.

        Args:
            entity_id: ID of the entity to update
            properties: Dictionary of property values to update

        Returns:
            The updated entity if successful, None if the entity was not found

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the updated properties violate the schema
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity from the graph.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if the entity was deleted, False if it was not found

        Raises:
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def create_relation(
        self, relation_type: str, from_entity: str, to_entity: str, properties: dict[str, Any] | None = None
    ) -> Relation:
        """
        Create a new relation between entities.

        Args:
            relation_type: Type of the relation to create
            from_entity: ID of the source entity
            to_entity: ID of the target entity
            properties: Optional dictionary of property values

        Returns:
            The created relation

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the relation type is not defined in the schema
            EntityNotFoundError: If either entity does not exist
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def get_relation(self, relation_id: str) -> Relation | None:
        """
        Retrieve a relation by ID.

        Args:
            relation_id: ID of the relation to retrieve

        Returns:
            The relation if found, None otherwise

        Raises:
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def update_relation(self, relation_id: str, properties: dict[str, Any]) -> Relation | None:
        """
        Update an existing relation.

        Args:
            relation_id: ID of the relation to update
            properties: Dictionary of property values to update

        Returns:
            The updated relation if successful, None if the relation was not found

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the updated properties violate the schema
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def query(self, query_spec: QuerySpec) -> list[Entity]:
        """
        Execute a query against the graph.

        Args:
            query_spec: Specification of the query to execute

        Returns:
            List of entities matching the query

        Raises:
            ValidationError: If the query specification is invalid
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def traverse(self, start_entity: str, traversal_spec: TraversalSpec) -> list[Entity]:
        """
        Traverse the graph starting from a given entity.

        Args:
            start_entity: ID of the entity to start traversal from
            traversal_spec: Specification of the traversal

        Returns:
            List of entities found during traversal

        Raises:
            EntityNotFoundError: If the start entity does not exist
            ValidationError: If the traversal specification is invalid
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def begin_transaction(self) -> None:
        """
        Begin a new transaction.

        This method should be called before a series of operations that need
        to be executed atomically.

        Raises:
            TransactionError: If a transaction is already in progress or if
                            the operation fails
        """
        pass

    @abstractmethod
    async def commit_transaction(self) -> None:
        """
        Commit the current transaction.

        This method should be called to persist the changes made during the
        current transaction.

        Raises:
            TransactionError: If no transaction is in progress or if the
                            operation fails
        """
        pass

    @abstractmethod
    async def rollback_transaction(self) -> None:
        """
        Roll back the current transaction.

        This method should be called to discard the changes made during the
        current transaction.

        Raises:
            TransactionError: If no transaction is in progress or if the
                            operation fails
        """
        pass
```

### Graph Store Interface

The GraphStore interface defines the contract for actual data persistence:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from .types.type_base import Entity, Relation, QuerySpec, TraversalSpec

class GraphStore(ABC):
    """
    Abstract interface for graph data storage operations.
    Concrete implementations handle the actual persistence of entities and relations.
    """

    @abstractmethod
    async def create_entity(
        self,
        entity_type: str,
        properties: Dict[str, Any]
    ) -> str:
        """Create a new entity in the graph."""
        pass

    @abstractmethod
    async def get_entity(
        self,
        entity_id: str
    ) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        pass

    @abstractmethod
    async def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Update an existing entity."""
        pass

    @abstractmethod
    async def delete_entity(
        self,
        entity_id: str
    ) -> bool:
        """Delete an entity from the graph."""
        pass

    @abstractmethod
    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new relation between entities."""
        pass

    @abstractmethod
    async def get_relation(
        self,
        relation_id: str
    ) -> Optional[Relation]:
        """Retrieve a relation by ID."""
        pass

    @abstractmethod
    async def update_relation(
        self,
        relation_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Update an existing relation."""
        pass

    @abstractmethod
    async def delete_relation(
        self,
        relation_id: str
    ) -> bool:
        """Delete a relation from the graph."""
        pass

    @abstractmethod
    async def query(
        self,
        query_spec: QuerySpec
    ) -> List[Entity]:
        """Execute a query against the graph."""
        pass

    @abstractmethod
    async def traverse(
        self,
        start_entity: str,
        traversal_spec: TraversalSpec
    ) -> List[Entity]:
        """Traverse the graph starting from a given entity."""
        pass

    @abstractmethod
    async def begin_transaction(self) -> None:
        """Begin a storage transaction."""
        pass

    @abstractmethod
    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        pass

    @abstractmethod
    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        pass
```

### Event System

The event system enables features to react to graph operations without coupling to specific implementations:

```python
from collections import defaultdict
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

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

    # Transaction operations
    TRANSACTION_BEGIN = "transaction:begin"
    TRANSACTION_COMMIT = "transaction:commit"
    TRANSACTION_ROLLBACK = "transaction:rollback"

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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

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
        if "data" not in data:
            data["data"] = {}
        if "metadata" not in data:
            data["metadata"] = EventMetadata()
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
        self, event: GraphEvent, metadata: Optional[EventMetadata] = None, **data: Any
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
            metadata_kwargs = self.create_metadata(event, data)
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

    def create_metadata(self, event: GraphEvent, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for an event based on event type and data."""
        metadata_kwargs = {}

        # Set bulk operation metadata
        if event in {
            GraphEvent.ENTITY_BULK_WRITE,
            GraphEvent.ENTITY_BULK_DELETE,
            GraphEvent.RELATION_BULK_WRITE,
            GraphEvent.RELATION_BULK_DELETE,
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

        return metadata_kwargs

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission.

        This can be useful during bulk operations or when
        temporary suppression of events is needed.
        """
        self._enabled = False
```

Example usage with metadata tracking:

```python
async def log_entity_changes(event_context: EventContext) -> None:
    """Example event handler that logs entity changes with metadata."""
    metadata = event_context.metadata
    print(f"Operation: {metadata.operation_id}")
    print(f"Timestamp: {metadata.timestamp}")
    print(f"Entity Type: {metadata.entity_type}")

    if metadata.is_bulk:
        print(f"Bulk operation affecting {metadata.affected_count} entities")

    if metadata.affected_types:
        print(f"Affected types: {', '.join(metadata.affected_types)}")

# Subscribe to entity write events
await context.event_system.subscribe(GraphEvent.ENTITY_WRITE, log_entity_changes)

# The handler will be called with detailed metadata for each entity write operation
```

### Store Configuration and Factory

The GraphStore implementation is loaded through a factory that handles configuration internally:

```python
from typing import Dict, Type

class GraphStoreFactory:
    """Factory for creating GraphStore instances from configuration."""

    _stores: Dict[str, Type[GraphStore]] = {}

    @classmethod
    def register(cls, store_type: str, store_class: Type[GraphStore]) -> None:
        """Register a store implementation."""
        cls._stores[store_type] = store_class

    @classmethod
    def create(cls) -> GraphStore:
        """Create a GraphStore instance based on internal configuration."""
        config = cls._load_config()  # Load from env vars, config files, etc.
        if config.type not in cls._stores:
            raise ValueError(f"Unknown store type: {config.type}")
        return cls._stores[config.type](config.config)

    @classmethod
    def _load_config(cls) -> 'StoreConfig':
        """Load store configuration from environment/config files."""
        # Configuration can be loaded from:
        # - Environment variables
        # - Configuration files
        # - System settings
        # - etc.
        pass

class BaseGraphContext(GraphContext):
    """Base implementation of GraphContext interface."""

    def __init__(self):
        self._store = GraphStoreFactory.create()  # Factory handles configuration
        self._events = EventSystem()
        self._entity_types = {}
        self._relation_types = {}
        self._in_transaction = False

    async def create_entity(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """Create a new entity in the graph."""
        self._check_transaction()
        validated_props = self.validate_entity(entity_type, properties)

        entity_id = await self._store.create_entity(entity_type, validated_props)

        await self._events.emit(
            GraphEvent.ENTITY_WRITE,
            entity_id=entity_id,
            entity_type=entity_type
        )

        return entity_id

    # Other GraphContext methods follow similar pattern:
    # 1. Validate state/input
    # 2. Delegate to store
    # 3. Emit appropriate events
    # 4. Return results

## Implementation Guidelines

### 1. Type System Integration

- Implement strict type checking for all operations
- Validate property types against schema definitions
- Handle type coercion where appropriate
- Maintain referential integrity

### 2. Error Handling

```python
class GraphContextError(Exception):
    """Base exception for all graph context errors."""
    pass

class EntityNotFoundError(GraphContextError):
    """Raised when an entity cannot be found."""
    pass

class RelationNotFoundError(GraphContextError):
    """Raised when a relation cannot be found."""
    pass

class ValidationError(GraphContextError):
    """Raised when entity or relation validation fails."""
    pass

class SchemaError(GraphContextError):
    """Raised when there are schema-related issues."""
    pass
```

### 3. Backend Implementation Requirements

Each backend implementation must:

1. Implement all abstract methods from the GraphContext interface
2. Handle transactions appropriately
3. Implement proper error handling and conversion
4. Maintain type safety and validation
5. Support async operations
6. Implement efficient querying and traversal
7. Handle proper resource cleanup

### 4. Testing Requirements

- Minimum 95% test coverage
- Unit tests for all interface methods
- Integration tests with at least one backend
- Property-based testing for type system
- Performance benchmarks for critical operations

## Dependencies

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.5.2",
    "typing-extensions>=4.8.0",
    "asyncio>=3.4.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",
    "pytest-asyncio>=0.21.1",
    "pytest-cov>=4.1.0",
    "hypothesis>=6.87.1",
    "ruff>=0.1.6",
]
```
