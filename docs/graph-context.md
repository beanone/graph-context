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
    │   └── test_event_system.py
    └── types/
        ├── __init__.py
        └── test_type_base.py
```

### Core Interfaces

#### GraphContext Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic
from .types.type_base import Entity, Relation, QuerySpec, TraversalSpec

T = TypeVar('T')

class GraphContext(ABC, Generic[T]):
    """
    Abstract base class defining the core graph operations interface.
    The generic type T represents the native node/edge types of the backend.
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
from enum import Enum
from typing import Any, Callable, Awaitable, Dict
from pydantic import BaseModel
from datetime import datetime

class GraphEvent(str, Enum):
    """Core graph operation events."""
    ENTITY_READ = "entity:read"
    ENTITY_WRITE = "entity:write"
    ENTITY_DELETE = "entity:delete"
    ENTITY_BULK_WRITE = "entity:bulk_write"
    ENTITY_BULK_DELETE = "entity:bulk_delete"
    RELATION_READ = "relation:read"
    RELATION_WRITE = "relation:write"
    RELATION_DELETE = "relation:delete"
    RELATION_BULK_WRITE = "relation:bulk_write"
    RELATION_BULK_DELETE = "relation:bulk_delete"
    QUERY_EXECUTED = "query:executed"
    TRAVERSAL_EXECUTED = "traversal:executed"
    SCHEMA_MODIFIED = "schema:modified"
    TYPE_MODIFIED = "type:modified"
    TRANSACTION_BEGIN = "transaction:begin"
    TRANSACTION_COMMIT = "transaction:commit"
    TRANSACTION_ROLLBACK = "transaction:rollback"

class EventMetadata(BaseModel):
    """Metadata for graph events."""
    entity_type: Optional[str] = None
    relation_type: Optional[str] = None
    operation_id: str
    timestamp: datetime
    query_spec: Optional[Dict[str, Any]] = None
    traversal_spec: Optional[Dict[str, Any]] = None
    is_bulk: bool = False
    affected_count: Optional[int] = None

class EventContext(BaseModel):
    """Context for a graph event."""
    event: GraphEvent
    metadata: EventMetadata
    data: Dict[str, Any]

EventHandler = Callable[[EventContext], Awaitable[None]]

class EventSystem:
    """Simple pub/sub system for graph operations."""

    def __init__(self) -> None:
        """Initialize the event system."""
        self._handlers: dict[GraphEvent, list[EventHandler]] = defaultdict(list)
        self._enabled = True

    async def subscribe(self, event: GraphEvent, handler: EventHandler) -> None:
        """Subscribe to a specific graph event."""
        self._handlers[event].append(handler)

    async def unsubscribe(self, event: GraphEvent, handler: EventHandler) -> None:
        """Unsubscribe from a specific graph event."""
        try:
            self._handlers[event].remove(handler)
        except ValueError:
            pass

    async def emit(self, event: GraphEvent, metadata: Optional[EventMetadata] = None, **data: Any) -> None:
        """Emit a graph event to all subscribers."""
        if not self._enabled:
            return

        if metadata is None:
            metadata = EventMetadata(
                operation_id=str(uuid4()),
                timestamp=datetime.utcnow()
            )

        context = EventContext(event=event, metadata=metadata, data=data)

        for handler in self._handlers[event]:
            try:
                await handler(context)
            except Exception:
                continue
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

## Usage Examples

### Basic Entity Operations

```python
# Create an entity
entity_id = await graph_context.create_entity(
    entity_type="Person",
    properties={
        "name": "Ada Lovelace",
        "birth_year": 1815,
        "fields": ["mathematics", "computing"]
    }
)

# Retrieve an entity
entity = await graph_context.get_entity(entity_id)

# Update an entity
success = await graph_context.update_entity(
    entity_id,
    properties={"death_year": 1852}
)

# Delete an entity
success = await graph_context.delete_entity(entity_id)
```

### Relation Operations

```python
# Create a relation
relation_id = await graph_context.create_relation(
    relation_type="authored",
    from_entity=person_id,
    to_entity=document_id,
    properties={"year": 1843}
)

# Query related entities
results = await graph_context.query({
    "start": person_id,
    "relation": "authored",
    "direction": "outbound"
})
```

### Graph Traversal

```python
# Traverse the graph
results = await graph_context.traverse(
    start_entity=person_id,
    traversal_spec={
        "max_depth": 2,
        "relation_types": ["authored", "cites"],
        "direction": "any"
    }
)
```

## Performance Considerations

1. **Caching Strategy**
   - Implement caching for frequently accessed entities
   - Cache validation results for common types
   - Use LRU cache for query results

2. **Batch Operations**
   - Support bulk entity/relation operations
   - Implement efficient batch querying
   - Optimize traversal for large graphs

3. **Memory Management**
   - Implement proper resource cleanup
   - Use connection pooling for database backends
   - Handle large result sets efficiently

## Security Considerations

1. **Input Validation**
   - Sanitize all input parameters
   - Validate property values against schema
   - Prevent injection attacks in queries

2. **Access Control**
   - Support for tenant isolation
   - Entity-level access control
   - Audit logging for critical operations

## Future Extensions

1. **Advanced Query Features**
   - Full-text search integration
   - Semantic similarity search
   - Pattern matching queries

2. **Schema Evolution**
   - Schema versioning support
   - Migration tooling
   - Backward compatibility

3. **Performance Optimizations**
   - Query plan optimization
   - Parallel query execution
   - Distributed graph support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## License

MIT License - See LICENSE file for details