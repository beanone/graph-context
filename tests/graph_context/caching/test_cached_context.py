"""Integration tests for the cached graph context implementation."""

import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import uuid4

from graph_context.caching import CachedGraphContext
from graph_context.context_base import BaseGraphContext
from graph_context.types.type_base import Entity, Relation, QuerySpec, TraversalSpec
from graph_context.event_system import EventContext, GraphEvent, EventMetadata

class MockBaseContext(BaseGraphContext):
    """Mock implementation of BaseGraphContext for testing."""

    def __init__(self):
        super().__init__()
        self._entities: Dict[str, Dict[str, Entity]] = {}
        self._relations: Dict[str, Dict[str, Relation]] = {}
        self._call_count: Dict[str, int] = {
            'get_entity': 0,
            'get_relation': 0,
            'query': 0,
            'traverse': 0
        }
        self._in_transaction = False

    async def initialize(self) -> None:
        """Initialize the context."""
        pass

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    async def create_entity(self, entity_type: str, properties: dict[str, Any]) -> Entity:
        """Create an entity in storage."""
        if entity_type not in self._entities:
            self._entities[entity_type] = {}
        entity_id = str(uuid4())
        entity = Entity(id=entity_id, type=entity_type, properties=properties)
        self._entities[entity_type][entity_id] = entity
        await self.emit(
            GraphEvent.ENTITY_WRITE,
            metadata=EventMetadata(
                entity_type=entity_type,
                operation_id=str(uuid4()),
                timestamp=datetime.utcnow()
            ),
            data={'entity': entity, 'entity_id': entity_id}
        )
        return entity

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity from storage."""
        self._call_count['get_entity'] += 1
        for entities in self._entities.values():
            if entity_id in entities:
                return entities[entity_id]
        return None

    async def update_entity(self, entity_id: str, properties: dict[str, Any]) -> Optional[Entity]:
        """Update an entity in storage."""
        entity = await self.get_entity(entity_id)
        if not entity:
            return None

        updated_entity = Entity(
            id=entity_id,
            type=entity.type,
            properties={**entity.properties, **properties}
        )
        self._entities[entity.type][entity_id] = updated_entity
        await self.emit(
            GraphEvent.ENTITY_WRITE,
            metadata=EventMetadata(
                entity_type=entity.type,
                operation_id=str(uuid4()),
                timestamp=datetime.utcnow()
            ),
            data={'entity': updated_entity, 'entity_id': entity_id}
        )
        return updated_entity

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity from storage."""
        for entity_type, entities in self._entities.items():
            if entity_id in entities:
                del entities[entity_id]
                await self.emit(
                    GraphEvent.ENTITY_DELETE,
                    metadata=EventMetadata(
                        entity_type=entity_type,
                        operation_id=str(uuid4()),
                        timestamp=datetime.utcnow()
                    ),
                    data={'entity_id': entity_id}
                )
                return True
        return False

    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: dict[str, Any] | None = None
    ) -> Relation:
        """Create a relation in storage."""
        if relation_type not in self._relations:
            self._relations[relation_type] = {}
        relation_id = str(uuid4())
        relation = Relation(
            id=relation_id,
            type=relation_type,
            source_id=from_entity,
            target_id=to_entity,
            properties=properties or {}
        )
        self._relations[relation_type][relation_id] = relation
        await self.emit(
            GraphEvent.RELATION_WRITE,
            metadata=EventMetadata(
                relation_type=relation_type,
                operation_id=str(uuid4()),
                timestamp=datetime.utcnow()
            ),
            data={'relation': relation, 'relation_id': relation_id}
        )
        return relation

    async def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get a relation from storage."""
        self._call_count['get_relation'] += 1
        for relations in self._relations.values():
            if relation_id in relations:
                return relations[relation_id]
        return None

    async def update_relation(
        self,
        relation_id: str,
        properties: dict[str, Any]
    ) -> Optional[Relation]:
        """Update a relation in storage."""
        relation = await self.get_relation(relation_id)
        if not relation:
            return None

        updated_relation = Relation(
            id=relation_id,
            type=relation.type,
            source_id=relation.source_id,
            target_id=relation.target_id,
            properties={**relation.properties, **properties}
        )
        self._relations[relation.type][relation_id] = updated_relation
        await self.emit(
            GraphEvent.RELATION_WRITE,
            metadata=EventMetadata(
                relation_type=relation.type,
                operation_id=str(uuid4()),
                timestamp=datetime.utcnow()
            ),
            data={'relation': updated_relation, 'relation_id': relation_id}
        )
        return updated_relation

    async def query(self, query_spec: QuerySpec) -> list[Entity]:
        """Execute a query."""
        self._call_count['query'] += 1
        entity_type = query_spec.type
        if not entity_type:
            return []

        results = []
        for entity_id, entity in self._entities.get(entity_type, {}).items():
            if self._matches_filter(entity, query_spec.filter or {}):
                results.append(entity)

        await self.emit(
            GraphEvent.QUERY_EXECUTED,
            metadata=EventMetadata(
                query_spec=query_spec.dict(),
                operation_id=str(uuid4()),
                timestamp=datetime.utcnow()
            ),
            data={'results': results}
        )
        return results

    async def traverse(
        self,
        start_entity: str,
        traversal_spec: TraversalSpec
    ) -> list[Entity]:
        """Execute a traversal."""
        self._call_count['traverse'] += 1
        relation_type = traversal_spec.relation_type
        if not relation_type:
            return []

        results = []
        for relation in self._relations.get(relation_type, {}).values():
            if relation.source_id == start_entity:
                target_entity = await self.get_entity(relation.target_id)
                if target_entity:
                    results.append(target_entity)

        await self.emit(
            GraphEvent.TRAVERSAL_EXECUTED,
            metadata=EventMetadata(
                traversal_spec=traversal_spec.dict(),
                operation_id=str(uuid4()),
                timestamp=datetime.utcnow()
            ),
            data={'results': results}
        )
        return results

    def _matches_filter(self, entity: Entity, filter_spec: Dict[str, Any]) -> bool:
        """Check if an entity matches a filter specification."""
        for key, value in filter_spec.items():
            if key not in entity.properties or entity.properties[key] != value:
                return False
        return True

    async def begin_transaction(self) -> None:
        """Begin a new transaction."""
        if self._in_transaction:
            raise RuntimeError("Transaction already in progress")
        self._in_transaction = True

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        if not self._in_transaction:
            raise RuntimeError("No transaction in progress")
        self._in_transaction = False

    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        if not self._in_transaction:
            raise RuntimeError("No transaction in progress")
        self._in_transaction = False

@pytest.fixture
def base_context():
    """Create a mock base context."""
    return MockBaseContext()

@pytest.fixture
def cached_context(base_context):
    """Create a cached context with the mock base."""
    return CachedGraphContext(base_context=base_context)

@pytest.fixture
def sample_entity():
    """Create a sample entity."""
    return Entity(id="test123", type="person", properties={"name": "Test Person"})

@pytest.fixture
def sample_relation():
    """Create a sample relation."""
    return Relation(
        id="rel123",
        type="knows",
        source_id="person1",
        target_id="person2",
        properties={}
    )

@pytest.mark.asyncio
async def test_entity_caching(cached_context, base_context, sample_entity):
    """Test entity caching behavior."""
    # Create an entity
    entity_id = await cached_context.create_entity("person", sample_entity)

    # First read should hit the base context
    initial_count = base_context._call_count['get_entity']
    entity1 = await cached_context.get_entity("person", entity_id)
    assert entity1 is not None
    assert base_context._call_count['get_entity'] > initial_count

    # Second read should hit the cache
    initial_count = base_context._call_count['get_entity']
    entity2 = await cached_context.get_entity("person", entity_id)
    assert entity2 is not None
    assert entity2 == entity1
    assert base_context._call_count['get_entity'] == initial_count

@pytest.mark.asyncio
async def test_relation_caching(cached_context, base_context, sample_relation):
    """Test relation caching behavior."""
    # Create a relation
    relation_id = await cached_context.create_relation("knows", sample_relation)

    # First read should hit the base context
    initial_count = base_context._call_count['get_relation']
    relation1 = await cached_context.get_relation("knows", relation_id)
    assert relation1 is not None
    assert base_context._call_count['get_relation'] > initial_count

    # Second read should hit the cache
    initial_count = base_context._call_count['get_relation']
    relation2 = await cached_context.get_relation("knows", relation_id)
    assert relation2 is not None
    assert relation2 == relation1
    assert base_context._call_count['get_relation'] == initial_count

@pytest.mark.asyncio
async def test_query_caching(cached_context, base_context, sample_entity):
    """Test query result caching."""
    # Create some test data
    await cached_context.create_entity("person", sample_entity)

    # Execute query first time
    query_spec = {"type": "person", "filter": {"name": "Test Person"}}
    initial_count = base_context._call_count['query']
    results1 = await cached_context.query(query_spec)
    assert len(results1) > 0
    assert base_context._call_count['query'] > initial_count

    # Execute same query again
    initial_count = base_context._call_count['query']
    results2 = await cached_context.query(query_spec)
    assert results2 == results1
    assert base_context._call_count['query'] == initial_count

@pytest.mark.asyncio
async def test_traversal_caching(cached_context, base_context, sample_relation):
    """Test traversal result caching."""
    # Create test data
    relation_id = await cached_context.create_relation("knows", sample_relation)

    # Execute traversal first time
    traversal_spec = {
        "start": sample_relation.source_id,
        "relation": "knows"
    }
    initial_count = base_context._call_count['traverse']
    path1 = await cached_context.traverse(traversal_spec)
    assert len(path1) > 0
    assert base_context._call_count['traverse'] > initial_count

    # Execute same traversal again
    initial_count = base_context._call_count['traverse']
    path2 = await cached_context.traverse(traversal_spec)
    assert path2 == path1
    assert base_context._call_count['traverse'] == initial_count

@pytest.mark.asyncio
async def test_cache_invalidation(cached_context, base_context, sample_entity):
    """Test cache invalidation on updates."""
    # Create and cache an entity
    entity_id = await cached_context.create_entity("person", sample_entity)
    await cached_context.get_entity("person", entity_id)  # Cache it

    # Update the entity
    updated_entity = Entity(
        id=entity_id,
        type="person",
        properties={"name": "Updated Person"}
    )
    await cached_context.update_entity("person", entity_id, updated_entity)

    # Next read should hit base context
    initial_count = base_context._call_count['get_entity']
    entity = await cached_context.get_entity("person", entity_id)
    assert entity is not None
    assert entity.properties["name"] == "Updated Person"
    assert base_context._call_count['get_entity'] > initial_count

@pytest.mark.asyncio
async def test_cache_enable_disable(cached_context, base_context, sample_entity):
    """Test enabling and disabling the cache."""
    # Create test data
    entity_id = await cached_context.create_entity("person", sample_entity)

    # Cache should be enabled by default
    initial_count = base_context._call_count['get_entity']
    await cached_context.get_entity("person", entity_id)
    second_count = base_context._call_count['get_entity']
    await cached_context.get_entity("person", entity_id)
    assert base_context._call_count['get_entity'] == second_count

    # Disable cache
    cached_context.disable_caching()
    initial_count = base_context._call_count['get_entity']
    await cached_context.get_entity("person", entity_id)
    assert base_context._call_count['get_entity'] > initial_count

    # Re-enable cache
    cached_context.enable_caching()
    initial_count = base_context._call_count['get_entity']
    await cached_context.get_entity("person", entity_id)
    second_count = base_context._call_count['get_entity']
    await cached_context.get_entity("person", entity_id)
    assert base_context._call_count['get_entity'] == second_count

@pytest.mark.asyncio
async def test_bulk_operations(cached_context, base_context):
    """Test caching with bulk operations."""
    # Create multiple entities
    entities = [
        Entity(id=f"test{i}", type="person", properties={"name": f"Person {i}"})
        for i in range(3)
    ]

    # Bulk create
    entity_ids = await cached_context.bulk_create_entities("person", entities)

    # First reads should hit base context
    initial_count = base_context._call_count['get_entity']
    for entity_id in entity_ids:
        await cached_context.get_entity("person", entity_id)
    assert base_context._call_count['get_entity'] > initial_count

    # Second reads should hit cache
    initial_count = base_context._call_count['get_entity']
    for entity_id in entity_ids:
        await cached_context.get_entity("person", entity_id)
    assert base_context._call_count['get_entity'] == initial_count

    # Bulk update should invalidate cache
    updates = [
        {"id": entity_id, "name": f"Updated Person {i}"}
        for i, entity_id in enumerate(entity_ids)
    ]
    await cached_context.bulk_update_entities("person", updates)

    # Next reads should hit base context again
    initial_count = base_context._call_count['get_entity']
    for entity_id in entity_ids:
        await cached_context.get_entity("person", entity_id)
    assert base_context._call_count['get_entity'] > initial_count