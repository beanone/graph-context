# Schema-Aware Event-Based Caching Design

## Overview

The caching system for the Graph Context component is designed to be schema-aware and event-driven, providing efficient caching while maintaining data consistency with schema changes. This design focuses on simplicity and extensibility, laying the groundwork for future versioning support.

## Architecture Diagrams

### Component Architecture

```mermaid
classDiagram
    class GraphContext {
        +cache_manager: SchemaAwareCacheManager
        +get_entity()
        +update_entity()
        +query()
    }

    class SchemaAwareCacheManager {
        +cache: TypeAwareCache
        +enabled: bool
        -handlers: Dict
        +handle_event()
        -handle_entity_read()
        -handle_entity_write()
        -handle_type_modified()
    }

    class TypeAwareCache {
        -cache: Dict
        -type_dependencies: Dict
        -query_dependencies: Dict
        +get()
        +set()
        +invalidate_type()
    }

    class GraphEvent {
        <<enumeration>>
        +ENTITY_READ
        +ENTITY_UPDATED
        +SCHEMA_TYPE_MODIFIED
    }

    GraphContext --> SchemaAwareCacheManager : uses
    SchemaAwareCacheManager --> TypeAwareCache : manages
    SchemaAwareCacheManager --> GraphEvent : handles
```

### Component Lifecycle Interactions

```mermaid
stateDiagram-v2
    [*] --> KGInitialization

    state KGInitialization {
        [*] --> SchemaLoading
        SchemaLoading --> CacheInitialization
        CacheInitialization --> Ready
    }

    state KGOperations {
        state EntityOperations {
            Create --> Read
            Read --> Update
            Update --> Delete
        }

        state CacheEvents {
            CacheWrite --> CacheRead
            CacheRead --> CacheInvalidate
            CacheInvalidate --> CacheWrite
        }

        state SchemaOperations {
            TypeRegistration --> TypeModification
            TypeModification --> TypeDeletion
        }

        EntityOperations --> CacheEvents
        SchemaOperations --> CacheEvents
    }

    state CacheLifecycle {
        state fork_state <<fork>>

        [*] --> fork_state

        fork_state --> LocalCache
        fork_state --> DistributedCache

        state LocalCache {
            [*] --> Warm
            Warm --> Hot
            Hot --> Invalidated
            Invalidated --> Warm
        }

        state DistributedCache {
            [*] --> Syncing
            Syncing --> Synchronized
            Synchronized --> PartiallyInvalidated
            PartiallyInvalidated --> Syncing
        }
    }

    KGInitialization --> KGOperations
    KGOperations --> CacheLifecycle
    CacheLifecycle --> KGOperations
```

### Component Interaction Details

```mermaid
flowchart TB
    subgraph KG_Lifecycle
        direction TB
        A1[Schema Definition] --> A2[Entity Creation]
        A2 --> A3[Relation Creation]
        A3 --> A4[Query Execution]
        A4 --> A5[Schema Evolution]
    end

    subgraph Cache_Events
        direction TB
        B1[Cache Initialization] --> B2[Cache Population]
        B2 --> B3[Cache Hit/Miss]
        B3 --> B4[Cache Invalidation]
        B4 --> B5[Cache Revalidation]
    end

    subgraph Cache_States
        direction TB
        C1[Empty] --> C2[Warming]
        C2 --> C3[Hot]
        C3 --> C4[Degraded]
        C4 --> C2
    end

    A1 --> B1
    A2 --> B2
    A3 --> B2
    A4 --> B3
    A5 --> B4

    B1 --> C1
    B2 --> C2
    B3 --> C3
    B4 --> C4

    classDef kgState fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef cacheEvent fill:#fff3e0,stroke:#ff6f00,stroke-width:2px;
    classDef cacheState fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    class A1,A2,A3,A4,A5 kgState;
    class B1,B2,B3,B4,B5 cacheEvent;
    class C1,C2,C3,C4 cacheState;
```

### Event Propagation Matrix

```mermaid
journey
    title Cache Events in KG Lifecycle
    section Schema Changes
        Type Registration: 5: Cache Init
        Type Modification: 3: Cache Invalidate
        Type Deletion: 1: Cache Clear
    section Entity Operations
        Create: 5: Cache Write
        Read: 3: Cache Read
        Update: 2: Cache Update
        Delete: 1: Cache Remove
    section Query Operations
        Simple Query: 5: Cache Hit
        Complex Query: 3: Partial Cache
        Schema-Dependent: 1: Cache Miss
```

### Event Flow

```mermaid
sequenceDiagram
    participant Client
    participant GraphContext
    participant CacheManager
    participant Cache
    participant Backend

    Client->>GraphContext: get_entity(id)
    GraphContext->>CacheManager: get(entity_id)
    CacheManager->>Cache: lookup

    alt Cache Hit
        Cache-->>CacheManager: cached entity
        CacheManager-->>GraphContext: cached entity
        GraphContext-->>Client: entity
    else Cache Miss
        Cache-->>CacheManager: none
        CacheManager-->>GraphContext: none
        GraphContext->>Backend: fetch entity
        Backend-->>GraphContext: entity
        GraphContext->>CacheManager: handle_event(ENTITY_READ)
        CacheManager->>Cache: set(entity)
        GraphContext-->>Client: entity
    end
```

### Cache Invalidation Flow

```mermaid
sequenceDiagram
    participant Client
    participant GraphContext
    participant CacheManager
    participant TypeCache
    participant QueryCache

    Client->>GraphContext: modify_schema(type)
    GraphContext->>CacheManager: handle_event(SCHEMA_MODIFIED)

    par Type Cache Invalidation
        CacheManager->>TypeCache: invalidate_type(type)
        TypeCache-->>CacheManager: cleared
    and Query Cache Invalidation
        CacheManager->>QueryCache: invalidate_queries(type)
        QueryCache-->>CacheManager: cleared
    end

    CacheManager-->>GraphContext: done
    GraphContext-->>Client: success
```

### Dependency Tracking

Note: The following diagram uses example entity types (Person, Address) and cache keys for illustration purposes only. In practice, the actual types and cache keys will depend on your specific graph schema and use cases.

```mermaid
graph TD
    A[Person Type] --> B[Entity Cache Keys]
    A --> C[Query Cache Keys]

    B --> D[person:123]
    B --> E[person:456]

    C --> F[query:age>25]
    C --> G[query:name=*]

    H[Address Type] --> I[Entity Cache Keys]
    H --> J[Query Cache Keys]

    I --> K[address:789]
    J --> F
```

### Cache Key Structure

```mermaid
graph LR
    A[Cache Key] --> B[Operation]
    A --> C[Type Info]
    A --> D[Parameters]

    B --> E[get_entity]
    B --> F[get_relation]
    B --> G[query]

    C --> H[type_name]
    C --> I[type_hash]

    D --> J[entity_id]
    D --> K[query_hash]
```

## Core Components

### 1. Graph Events

The system uses an event-based architecture to handle both operations and schema changes:

```python
class GraphEvent(Enum):
    # Operation Events
    ENTITY_READ = "entity:read"
    ENTITY_CREATED = "entity:created"
    ENTITY_UPDATED = "entity:updated"
    ENTITY_DELETED = "entity:deleted"

    RELATION_READ = "relation:read"
    RELATION_CREATED = "relation:created"
    RELATION_UPDATED = "relation:updated"
    RELATION_DELETED = "relation:deleted"

    QUERY_EXECUTED = "query:executed"

    # Schema Events
    SCHEMA_ENTITY_TYPE_MODIFIED = "schema:entity_type:modified"
    SCHEMA_RELATION_TYPE_MODIFIED = "schema:relation_type:modified"
    SCHEMA_TYPE_DELETED = "schema:type:deleted"
```

### 2. Type-Aware Cache

The cache implementation tracks dependencies between types and cached data:

```python
class TypeAwareCache:
    _cache: Dict[str, Any]  # Main cache storage
    _type_dependencies: Dict[str, Set[str]]  # Type -> Cache Keys mapping
    _query_dependencies: Dict[str, Set[str]]  # Type -> Query Cache Keys mapping
```

Key Features:
- Type-based dependency tracking
- Separate tracking for query results
- Efficient invalidation patterns
- Support for complex type relationships

### 3. Cache Manager

The SchemaAwareCacheManager orchestrates caching operations and event handling:

```python
class SchemaAwareCacheManager:
    cache: TypeAwareCache
    enabled: bool
    _handlers: Dict[GraphEvent, Callable]
```

Responsibilities:
- Event handling and routing
- Cache operation coordination
- Type invalidation management
- Query result caching

## Caching Strategies

### 1. Entity Caching

Entities are cached with their type information:
```python
cache_key = f"get_entity:entity_id={entity_id}"
type_dependencies[entity.type].add(cache_key)
```

Cache Invalidation Triggers:
- Entity updates/deletes
- Schema modifications to entity type
- Related type modifications

### 2. Relation Caching

Relations are cached with both relation type and connected entity types:
```python
cache_key = f"get_relation:relation_id={relation_id}"
type_dependencies[relation.type].add(cache_key)
```

Cache Invalidation Triggers:
- Relation updates/deletes
- Schema modifications to relation type
- Connected entity type modifications

### 3. Query Result Caching

Query results are cached with involved type tracking:
```python
cache_key = f"query:query_hash={query_hash}"
for involved_type in involved_types:
    query_dependencies[involved_type].add(cache_key)
```

Cache Invalidation Triggers:
- Any modification to involved types
- Schema changes to involved types
- Query specification changes

## Event Handling

### 1. Operation Events

#### Read Operations
```python
async def _handle_entity_read(context: EventContext):
    if context.result:
        await cache.set(
            "get_entity",
            context.result,
            type_name=context.result.type,
            entity_id=context.result.id
        )
```

#### Write Operations
```python
async def _handle_entity_write(context: EventContext):
    entity_type = context.metadata.get("entity_type")
    if entity_type:
        await cache.invalidate_type(entity_type)
```

### 2. Schema Events

#### Type Modifications
```python
async def _handle_type_modified(context: EventContext):
    type_name = context.metadata.get("type_name")
    if type_name:
        await cache.invalidate_type(type_name)
```

## Cache Invalidation

### 1. Type-Based Invalidation

When a type is modified:
1. Invalidate all direct cache entries for the type
2. Invalidate affected query results
3. Clear type dependencies

```python
async def invalidate_type(self, type_name: str):
    # Direct dependencies
    keys_to_remove = self._type_dependencies.get(type_name, set())
    for key in keys_to_remove:
        self._cache.pop(key, None)

    # Query dependencies
    query_keys = self._query_dependencies.get(type_name, set())
    for key in query_keys:
        self._cache.pop(key, None)
```

### 2. Query Invalidation

Queries are invalidated when:
- Any involved type is modified
- Schema changes affect query conditions
- Related types are modified

## Integration with GraphContext

### 1. Base Implementation

```python
class BaseGraphContext(GraphContext):
    def __init__(self):
        self.cache_manager = SchemaAwareCacheManager()
        self._entity_types: Dict[str, EntityType] = {}
        self._relation_types: Dict[str, RelationType] = {}
```

### 2. Operation Integration

```python
async def get_entity(self, entity_id: str) -> Optional[Entity]:
    # Try cache
    cached = await self.cache_manager.cache.get(
        "get_entity",
        entity_id=entity_id
    )
    if cached:
        return cached

    # Get from backend and cache
    entity = await self._get_entity_from_backend(entity_id)
    if entity:
        await self.cache_manager.handle_event(
            GraphEvent.ENTITY_READ,
            EventContext(operation="get_entity", result=entity)
        )
    return entity
```

## Deployment Scenarios

The caching system can be deployed in various configurations depending on scale, performance requirements, and infrastructure constraints. Below are some common deployment patterns:

### Single-Node Deployment

```mermaid
flowchart TD
    subgraph "Application Server"
        A[Graph Context] --> B[Cache Manager]
        B --> C[In-Memory Cache]
    end

    subgraph "Storage Layer"
        D[(Graph Database)]
    end

    A --> D
    style C fill:#f9f,stroke:#333,stroke-width:2px
```

### Distributed Cache Deployment

```mermaid
flowchart TD
    subgraph "Application Cluster"
        subgraph "App Server 1"
            A1[Graph Context] --> B1[Cache Manager]
            B1 --> C1[Local Cache]
        end

        subgraph "App Server 2"
            A2[Graph Context] --> B2[Cache Manager]
            B2 --> C2[Local Cache]
        end
    end

    subgraph "Cache Layer"
        D[Redis Cluster]
    end

    subgraph "Storage Layer"
        E[(Graph Database)]
    end

    B1 --> D
    B2 --> D
    A1 --> E
    A2 --> E

    style C1 fill:#f9f,stroke:#333,stroke-width:2px
    style C2 fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#9cf,stroke:#333,stroke-width:2px
```

### High-Availability Configuration

```mermaid
flowchart TD
    subgraph "Region A"
        subgraph "App Cluster A"
            A1[Graph Context] --> B1[Cache Manager]
            B1 --> C1[Local Cache]
        end

        subgraph "Cache Cluster A"
            D1[Redis Primary]
            D2[Redis Replica]
            D1 --> D2
        end
    end

    subgraph "Region B"
        subgraph "App Cluster B"
            A2[Graph Context] --> B2[Cache Manager]
            B2 --> C2[Local Cache]
        end

        subgraph "Cache Cluster B"
            E1[Redis Primary]
            E2[Redis Replica]
            E1 --> E2
        end
    end

    subgraph "Database Cluster"
        F1[(Primary DB)]
        F2[(Secondary DB)]
        F1 --> F2
    end

    B1 --> D1
    B2 --> E1
    D1 <--> E1

    A1 --> F1
    A2 --> F1

    style C1 fill:#f9f,stroke:#333,stroke-width:2px
    style C2 fill:#f9f,stroke:#333,stroke-width:2px
    style D1 fill:#9cf,stroke:#333,stroke-width:2px
    style D2 fill:#9cf,stroke:#333,stroke-width:2px
    style E1 fill:#9cf,stroke:#333,stroke-width:2px
    style E2 fill:#9cf,stroke:#333,stroke-width:2px
```

Note: These deployment diagrams are illustrative examples showing possible configurations. The actual deployment architecture should be designed based on specific requirements such as:
- Scale and performance needs
- High availability requirements
- Data consistency requirements
- Geographic distribution
- Infrastructure constraints
- Cost considerations

### Deployment Considerations

1. **Single-Node Deployment**
   - Suitable for development and small-scale deployments
   - Simple to maintain and debug
   - Limited by single node capacity
   - No high availability

2. **Distributed Cache Deployment**
   - Scales horizontally with application servers
   - Shared cache layer for consistency
   - Better resource utilization
   - Requires cache synchronization strategy

3. **High-Availability Configuration**
   - Multi-region support
   - Disaster recovery capability
   - Complex cache synchronization
   - Higher operational overhead

### Cache Synchronization Strategies

1. **Write-Through**
   ```mermaid
   sequenceDiagram
       participant App
       participant Local
       participant Redis
       participant DB

       App->>Local: Write Data
       App->>Redis: Write Data
       App->>DB: Write Data
       DB-->>App: Confirm
   ```

2. **Write-Behind**
   ```mermaid
   sequenceDiagram
       participant App
       participant Local
       participant Redis
       participant DB

       App->>Local: Write Data
       App->>Redis: Write Data
       Redis-->>App: Confirm
       Redis->>DB: Async Write
       DB-->>Redis: Confirm
   ```

## Performance Considerations

### 1. Cache Key Design
- Efficient key generation
- Minimal string operations
- Predictable key patterns

### 2. Memory Management
- Cache size limits
- Dependency tracking overhead
- Query result size considerations

### 3. Invalidation Efficiency
- Fast type-based lookup
- Efficient dependency tracking
- Minimal lock contention

## Future Extensions

### 1. Schema Versioning
- Version hash per type
- Version-aware cache keys
- Migration support

### 2. Advanced Caching Features
- TTL support
- Priority-based eviction
- Partial cache invalidation

### 3. Monitoring and Debugging
- Cache hit/miss metrics
- Invalidation tracking
- Performance monitoring

## Usage Examples

### 1. Basic Entity Caching

```python
# Create and cache entity
person = await graph_context.create_entity(
    "Person",
    {"name": "John Doe", "age": 30}
)

# Cached read
person = await graph_context.get_entity(person.id)

# Modify schema and invalidate cache
person_type = graph_context._entity_types["Person"]
person_type.properties["email"] = PropertyDefinition(
    type=PropertyType.STRING,
    required=True
)

await graph_context.cache_manager.handle_event(
    GraphEvent.SCHEMA_ENTITY_TYPE_MODIFIED,
    EventContext(
        operation="modify_entity_type",
        result=person_type,
        metadata={"type_name": "Person"}
    )
)
```

### 2. Query Caching

```python
# Execute and cache query
results = await graph_context.query({
    "entity_type": "Person",
    "conditions": [
        {"field": "age", "operator": "gt", "value": 25}
    ]
})

# Modify person type and invalidate query cache
await graph_context.cache_manager.handle_event(
    GraphEvent.SCHEMA_ENTITY_TYPE_MODIFIED,
    EventContext(
        operation="modify_entity_type",
        metadata={"type_name": "Person"}
    )
)
```

## Best Practices

1. **Event Handling**
   - Always emit events for schema changes
   - Include relevant metadata in events
   - Handle event failures gracefully

2. **Cache Management**
   - Monitor cache size and hit rates
   - Implement cache warming for critical data
   - Regular cache maintenance

3. **Type Dependencies**
   - Track all relevant type dependencies
   - Consider indirect dependencies
   - Maintain clean dependency graphs

4. **Error Handling**
   - Graceful degradation on cache failures
   - Clear error messages
   - Automatic recovery mechanisms

## Limitations

1. **Current Limitations**
   - No schema versioning support
   - Simple invalidation strategy
   - Basic query dependency tracking

2. **Known Trade-offs**
   - Memory usage vs cache effectiveness
   - Invalidation granularity vs complexity
   - Event handling overhead

## Next Steps

1. **Short Term**
   - Implement basic monitoring
   - Add cache size limits
   - Improve query dependency tracking

2. **Medium Term**
   - Add schema versioning support
   - Implement TTL support
   - Add partial cache invalidation

3. **Long Term**
   - Distributed cache support
   - Advanced query caching
   - Real-time cache updates