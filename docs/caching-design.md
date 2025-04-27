# Schema-Aware Event-Based Caching Design

## Overview

The caching system for the Graph Context component is designed to be schema-aware and event-driven, providing efficient caching while maintaining data consistency with schema changes. The implementation uses a decorator pattern to add caching capabilities to any graph context implementation, with a focus on transaction support, type awareness, and efficient invalidation strategies.

## Architecture Diagrams

### Component Architecture

```mermaid
classDiagram
    class GraphContext {
        +event_system: EventSystem
        +get_entity()
        +update_entity()
        +query()
        +traverse()
    }

    class CachedGraphContext {
        -base_context: GraphContext
        -cache_manager: CacheManager
        +_initialize()
    }

    class CacheManager {
        +config: CacheConfig
        +store_manager: CacheStoreManager
        +metrics: CacheMetrics
        +handle_event()
        +enable()
        +disable()
        +clear()
        -_subscribe_to_events()
        -_hash_query()
        -_handle_entity_read()
        -_handle_entity_write()
        -_handle_entity_delete()
        -_handle_relation_read()
        -_handle_relation_write()
        -_handle_relation_delete()
        -_handle_query_executed()
        -_handle_traversal_executed()
        -_handle_schema_modified()
    }

    class CacheStoreManager {
        +entity_store: CacheStore
        +relation_store: CacheStore
        +query_store: CacheStore
        +traversal_store: CacheStore
        +get_entity_store()
        +get_relation_store()
        +get_query_store()
        +get_traversal_store()
        +clear_all()
    }

    class CacheStore {
        -_cache: TTLCache | Dict
        +get()
        +set()
        +delete()
        +clear()
        +scan()
    }

    class DisabledCacheStore {
        +get()
        +set()
        +delete()
        +clear()
        +scan()
    }

    class CacheEntry~T~ {
        +value: T
        +created_at: datetime
        +entity_type: Optional[str]
        +relation_type: Optional[str]
        +operation_id: str
        +query_hash: Optional[str]
        +dependencies: Set[str]
    }

    GraphContext <|-- CachedGraphContext
    CachedGraphContext --> CacheManager : uses
    CacheManager --> CacheStoreManager : manages
    CacheStoreManager --> CacheStore : creates
    CacheStore <|-- DisabledCacheStore
    CacheStore --> CacheEntry : stores
    CacheManager --> EventSystem : subscribes to
    EventSystem --> CacheManager : notifies
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

Note: The following diagram uses example entity types (Person, Address) and cache keys for illustration purposes only. The actual types and cache keys depend on your specific graph schema and use cases. The `CacheStore` implementation uses several internal dictionaries to track dependencies:

- `_type_dependencies`: Maps type names (e.g., "Person") to a set of cache keys that depend on this type.
- `_query_dependencies`: Maps query hashes to a set of cache keys representing the results of that query.
- `_reverse_dependencies`: Maps a cache key to a set of other cache keys that depend on it (e.g., a query result key might depend on several entity keys).
- `_entity_relations`: Maps an entity ID to a set of relation cache keys involving that entity.
- `_relation_entities`: Maps a relation cache key to a set of entity IDs involved in that relation.

```mermaid
graph TD
    subgraph "Dependency Maps in CacheStore"
        A["_type_dependencies"] -->|Type -> Keys| B(Set of Cache Keys)
        C["_query_dependencies"] -->|Query Hash -> Keys| B
        D["_reverse_dependencies"] -->|Key -> Dependent Keys| B
        E["_entity_relations"] -->|Entity ID -> Relation Keys| F(Set of Relation Keys)
        G["_relation_entities"] -->|Relation Key -> Entity IDs| H(Set of Entity IDs)
    end

    subgraph "Example: Invalidation Flow"
        direction LR
        I[Update Entity 'person:123'] --> J{Invalidate 'person:123' Cache Entry}
        J --> K[Lookup 'person:123' in _reverse_dependencies]
        K --> L(Set of Dependent Keys e.g., 'query:abc', 'relation:xyz')
        J --> M[Lookup 'person:123' in _entity_relations]
        M --> N(Set of Related Relation Keys e.g., 'relation:789')
        L --> O{Delete Dependent Keys}
        N --> O
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
```

### Cache Key Structure

Cache keys are structured strings to identify the operation and parameters:

- **Entities:** `entity:<entity_id>` (e.g., `entity:person-123`)
- **Relations:** `relation:<relation_id>` (e.g., `relation:knows-456`)
- **Queries:** `query:<query_hash>` (e.g., `query:a1b2c3d4...`)
- **Traversals:** `traversal:<traversal_hash>` (e.g., `traversal:e5f6g7h8...`)

Query and traversal hashes are generated using `hashlib.sha256` on the JSON representation of the respective specification.

```mermaid
graph LR
    A[Cache Key Structure] --> B[Operation Prefix]
    A --> C[Identifier]

    B --> D[entity:]
    B --> E[relation:]
    B --> F[query:]
    B --> G[traversal:]

    C --> H[Entity/Relation ID]
    C --> I[Query/Traversal Hash]
```

## Implementation Details

### Core Components

1. **CachedGraphContext**
   - Implements decorator pattern to wrap any graph context
   - Provides caching for all graph operations
   - Maintains transaction awareness
   - Delegates to underlying context when needed

2. **CacheManager**
   - Central component for cache operations
   - Handles all graph events
   - Manages transaction state
   - Collects cache metrics
   - Coordinates between components

3. **CacheStore**
   - Provides TTL-based caching
   - Tracks type dependencies
   - Manages query result caching
   - Handles entity-relation dependencies
   - Supports efficient bulk operations

4. **CacheEntry**
   - Generic container for cached values
   - Stores metadata (creation time, types)
   - Tracks dependencies
   - Supports query result caching

### Cache Entry Model

The `CacheEntry` class serves as the core data structure for storing cached values with associated metadata:

```python
class CacheEntry[T]:
    value: T                      # The cached value (any JSON-serializable value)
    created_at: datetime          # Entry creation timestamp (UTC)
    entity_type: Optional[str]    # Type name for entity entries
    relation_type: Optional[str]  # Type name for relation entries
    operation_id: str             # Unique identifier for creating operation (UUID)
    query_hash: Optional[str]     # Hash of query that produced this result
    dependencies: Set[str]        # Entity/relation IDs this entry depends on
```

### Cache Store Implementation

The caching system provides two main cache store implementations:

1. **CacheStore**: The primary implementation supporting TTL-based caching
2. **DisabledCacheStore**: A no-op implementation for when caching is disabled

#### Features
- **TTL Support**: Configurable time-to-live for cache entries using `cachetools.TTLCache`
- **Size Limits**: Configurable maximum cache size (default: 10,000 entries)
- **Default TTL**: 5 minutes default TTL for entries
- **Async Interface**: All operations are async-compatible
- **Key Operations**:
  - `get(key)`: Retrieve entry by key
  - `set(key, entry)`: Store new entry
  - `delete(key)`: Remove entry
  - `clear()`: Remove all entries
  - `scan()`: Async iterator over all entries

### Key Features

1. **Transaction Support**
   - Separate transaction cache
   - Proper handling of transaction boundaries
   - Rollback support
   - Consistency during concurrent operations

2. **Type Awareness**
   - Schema-based invalidation
   - Type dependency tracking
   - Efficient type-based cache clearing
   - Support for schema evolution

3. **Query Caching**
   - Result caching with dependency tracking
   - Query hash-based lookup
   - Automatic invalidation on dependencies
   - Support for complex queries

4. **Event System Integration**
   - Subscription to all graph events
   - Event-driven cache updates
   - Automatic invalidation
   - Metrics collection

5. **Performance Optimizations**
   - TTL-based expiration
   - Bulk operation support
   - Efficient dependency tracking
   - Memory-efficient storage

### Cache Invalidation Strategies

1. **Type-Based Invalidation**
   - Triggered by schema changes
   - Clears all entries of affected type
   - Handles cascading dependencies
   - Maintains consistency

2. **Query-Based Invalidation**
   - Triggered by data changes
   - Clears affected query results
   - Tracks query dependencies
   - Supports partial invalidation

3. **Dependency-Based Invalidation**
   - Tracks entity-relation dependencies
   - Handles cascading updates
   - Maintains referential integrity
   - Supports complex graphs

4. **Transaction-Aware Invalidation**
   - Respects transaction boundaries
   - Supports rollback scenarios
   - Maintains ACID properties
   - Handles concurrent access

### Configuration

The caching system is highly configurable through the `CacheConfig` class:

1. **Cache Settings**
   - TTL duration
   - Maximum cache size
   - Metrics collection
   - Debug logging

2. **Store Configuration**
   - Store implementation selection
   - Store-specific settings
   - Connection parameters
   - Persistence options

3. **Event System Integration**
   - Event subscription configuration
   - Handler registration
   - Event filtering
   - Custom event support

### Metrics and Monitoring

1. **Cache Performance**
   - Hit/miss ratios
   - Operation latencies
   - Cache size tracking
   - Memory usage

2. **Health Monitoring**
   - Store connectivity
   - Cache consistency
   - Error tracking
   - Performance alerts

### Best Practices

1. **Cache Usage**
   - Configure appropriate TTLs
   - Monitor cache size
   - Use transactions appropriately
   - Handle errors gracefully

2. **Performance Optimization**
   - Enable metrics collection
   - Monitor hit ratios
   - Tune cache settings
   - Optimize queries

3. **Maintenance**
   - Regular monitoring
   - Performance tuning
   - Error handling
   - Capacity planning

## Caching Strategies

## Usage Examples

### Basic Entity Caching

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

### Query Caching

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

## Limitations and Trade-offs

1. **Current Limitations**
   - No schema versioning support
   - Simple invalidation strategy
   - Basic query dependency tracking

2. **Known Trade-offs**
   - Memory usage vs cache effectiveness
   - Invalidation granularity vs complexity
   - Event handling overhead

## Future Enhancements

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
