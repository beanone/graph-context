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
        self._entities: Dict[str, Dict[str, dict[str, Any]]] = {}
        self._relations: Dict[str, Dict[str, dict[str, Any]]] = {}
        self._call_count: Dict[str, int] = {
            'get_entity': 0,
            'get_relation': 0,
            'query': 0,
            'traverse': 0
        }
        self._in_transaction = False

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._entities.clear()
        self._relations.clear()
        self._call_count.clear()
        self._in_transaction = False
        await super().cleanup()

    async def _create_entity_impl(self, entity_type: str, properties: dict[str, Any]) -> str:
        """Implementation method to create an entity."""
        if entity_type not in self._entities:
            self._entities[entity_type] = {}
        entity_id = str(uuid4())
        entity = {
            'id': entity_id,
            'type': entity_type,
            'properties': properties
        }
        self._entities[entity_type][entity_id] = entity
        await self.emit(
            EventContext(
                event=GraphEvent.ENTITY_WRITE,
                metadata=EventMetadata(
                    entity_type=entity_type,
                    operation_id=str(uuid4()),
                    timestamp=datetime.utcnow()
                ),
                data={'entity': entity, 'entity_id': entity_id}
            )
        )
        return entity_id

    async def _get_entity_impl(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Implementation method to get an entity."""
        self._call_count['get_entity'] += 1
        for entities in self._entities.values():
            if entity_id in entities:
                return entities[entity_id]
        return None

    async def _update_entity_impl(self, entity_id: str, properties: dict[str, Any]) -> bool:
        """Implementation method to update an entity."""
        entity = await self.get_entity(entity_id)
        if not entity:
            return False

        updated_entity = {
            'id': entity_id,
            'type': entity['type'],
            'properties': {**entity['properties'], **properties}
        }
        self._entities[entity['type']][entity_id] = updated_entity
        await self.emit(
            EventContext(
                event=GraphEvent.ENTITY_WRITE,
                metadata=EventMetadata(
                    entity_type=entity['type'],
                    operation_id=str(uuid4()),
                    timestamp=datetime.utcnow()
                ),
                data={'entity': updated_entity, 'entity_id': entity_id}
            )
        )
        return True

    async def _delete_entity_impl(self, entity_id: str) -> bool:
        """Implementation method to delete an entity."""
        for entity_type, entities in self._entities.items():
            if entity_id in entities:
                del entities[entity_id]
                await self.emit(
                    EventContext(
                        event=GraphEvent.ENTITY_DELETE,
                        metadata=EventMetadata(
                            entity_type=entity_type,
                            operation_id=str(uuid4()),
                            timestamp=datetime.utcnow()
                        ),
                        data={'entity_id': entity_id}
                    )
                )
                return True
        return False

    async def _create_relation_impl(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: dict[str, Any] | None = None
    ) -> str:
        """Implementation method to create a relation."""
        if relation_type not in self._relations:
            self._relations[relation_type] = {}
        relation_id = str(uuid4())
        relation = {
            'id': relation_id,
            'type': relation_type,
            'source_id': from_entity,
            'target_id': to_entity,
            'properties': properties or {}
        }
        self._relations[relation_type][relation_id] = relation
        await self.emit(
            EventContext(
                event=GraphEvent.RELATION_WRITE,
                metadata=EventMetadata(
                    relation_type=relation_type,
                    operation_id=str(uuid4()),
                    timestamp=datetime.utcnow()
                ),
                data={'relation': relation, 'relation_id': relation_id}
            )
        )
        return relation_id

    async def _get_relation_impl(self, relation_id: str) -> Optional[dict[str, Any]]:
        """Implementation method to get a relation."""
        self._call_count['get_relation'] += 1
        for relations in self._relations.values():
            if relation_id in relations:
                return relations[relation_id]
        return None

    async def _update_relation_impl(
        self,
        relation_id: str,
        properties: dict[str, Any]
    ) -> bool:
        """Implementation method to update a relation."""
        relation = await self.get_relation(relation_id)
        if not relation:
            return False

        updated_relation = {
            'id': relation_id,
            'type': relation['type'],
            'source_id': relation['source_id'],
            'target_id': relation['target_id'],
            'properties': {**relation['properties'], **properties}
        }
        self._relations[relation['type']][relation_id] = updated_relation
        await self.emit(
            EventContext(
                event=GraphEvent.RELATION_WRITE,
                metadata=EventMetadata(
                    relation_type=relation['type'],
                    operation_id=str(uuid4()),
                    timestamp=datetime.utcnow()
                ),
                data={'relation': updated_relation, 'relation_id': relation_id}
            )
        )
        return True

    async def _delete_relation_impl(self, relation_id: str) -> bool:
        """Implementation method to delete a relation."""
        for relation_type, relations in self._relations.items():
            if relation_id in relations:
                del relations[relation_id]
                return True
        return False

    async def _query_impl(self, query_spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Implementation method to execute a query."""
        self._call_count['query'] += 1
        entity_type = query_spec.get('type')
        if not entity_type:
            return []

        results = []
        for entity_id, entity in self._entities.get(entity_type, {}).items():
            if self._matches_filter(entity, query_spec.get('filter', {})):
                results.append(entity)

        await self.emit(
            EventContext(
                event=GraphEvent.QUERY_EXECUTED,
                metadata=EventMetadata(
                    query_spec=query_spec,
                    operation_id=str(uuid4()),
                    timestamp=datetime.utcnow()
                ),
                data={'results': results}
            )
        )
        return results

    async def _traverse_impl(
        self,
        start_entity: str,
        traversal_spec: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Implementation method to execute a traversal."""
        self._call_count['traverse'] += 1
        relation_type = traversal_spec.get('relation_type')
        if not relation_type:
            return []

        results = []
        visited = set()
        max_depth = traversal_spec.get('max_depth', 1)
        direction = traversal_spec.get('direction', 'outbound')

        def add_to_results(entity_id: str, depth: int) -> None:
            if depth > max_depth or entity_id in visited:
                return
            visited.add(entity_id)
            for rel_type, relations in self._relations.items():
                if relation_type and rel_type != relation_type:
                    continue
                for relation in relations.values():
                    next_id = None
                    if direction in ('outbound', 'any') and relation['source_id'] == entity_id:
                        next_id = relation['target_id']
                    elif direction in ('inbound', 'any') and relation['target_id'] == entity_id:
                        next_id = relation['source_id']
                    if next_id and next_id not in visited:
                        entity = self.get_entity(next_id)
                        if entity:
                            results.append(entity)
                            add_to_results(next_id, depth + 1)

        add_to_results(start_entity, 1)
        await self.emit(
            EventContext(
                event=GraphEvent.TRAVERSAL_EXECUTED,
                metadata=EventMetadata(
                    traversal_spec=traversal_spec,
                    operation_id=str(uuid4()),
                    timestamp=datetime.utcnow()
                ),
                data={'results': results}
            )
        )
        return results

    def _matches_filter(self, entity: dict[str, Any], filter_spec: Dict[str, Any]) -> bool:
        """Check if an entity matches a filter specification."""
        for key, value in filter_spec.items():
            if key not in entity['properties'] or entity['properties'][key] != value:
                return False
        return True

    async def begin_transaction(self) -> None:
        """Begin a transaction."""
        self._in_transaction = True
        await self.emit(
            EventContext(
                event=GraphEvent.TRANSACTION_BEGIN,
                metadata=EventMetadata(
                    operation_id=str(uuid4()),
                    timestamp=datetime.utcnow()
                ),
                data={}
            )
        )

    async def commit_transaction(self) -> None:
        """Commit a transaction."""
        self._in_transaction = False
        await self.emit(
            EventContext(
                event=GraphEvent.TRANSACTION_COMMIT,
                metadata=EventMetadata(
                    operation_id=str(uuid4()),
                    timestamp=datetime.utcnow()
                ),
                data={}
            )
        )

    async def rollback_transaction(self) -> None:
        """Rollback a transaction."""
        self._in_transaction = False
        await self.emit(
            EventContext(
                event=GraphEvent.TRANSACTION_ROLLBACK,
                metadata=EventMetadata(
                    operation_id=str(uuid4()),
                    timestamp=datetime.utcnow()
                ),
                data={}
            )
        )

class TestCachedGraphContext(CachedGraphContext):
    """Concrete implementation of CachedGraphContext for testing."""

    async def _create_entity_impl(self, entity_type: str, properties: dict[str, Any]) -> str:
        """Implementation method to create an entity."""
        return await self.base_context._create_entity_impl(entity_type, properties)

    async def _get_entity_impl(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Implementation method to get an entity."""
        return await self.base_context._get_entity_impl(entity_id)

    async def _update_entity_impl(self, entity_id: str, properties: dict[str, Any]) -> bool:
        """Implementation method to update an entity."""
        return await self.base_context._update_entity_impl(entity_id, properties)

    async def _delete_entity_impl(self, entity_id: str) -> bool:
        """Implementation method to delete an entity."""
        return await self.base_context._delete_entity_impl(entity_id)

    async def _create_relation_impl(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: dict[str, Any] | None = None
    ) -> str:
        """Implementation method to create a relation."""
        return await self.base_context._create_relation_impl(relation_type, from_entity, to_entity, properties)

    async def _get_relation_impl(self, relation_id: str) -> Optional[dict[str, Any]]:
        """Implementation method to get a relation."""
        return await self.base_context._get_relation_impl(relation_id)

    async def _update_relation_impl(
        self,
        relation_id: str,
        properties: dict[str, Any]
    ) -> bool:
        """Implementation method to update a relation."""
        return await self.base_context._update_relation_impl(relation_id, properties)

    async def _delete_relation_impl(self, relation_id: str) -> bool:
        """Implementation method to delete a relation."""
        return await self.base_context._delete_relation_impl(relation_id)

    async def _query_impl(self, query_spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Implementation method to execute a query."""
        return await self.base_context._query_impl(query_spec)

    async def _traverse_impl(
        self,
        start_entity: str,
        traversal_spec: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Implementation method to execute a traversal."""
        return await self.base_context._traverse_impl(start_entity, traversal_spec)

@pytest.fixture
def base_context():
    """Create a mock base context."""
    return MockBaseContext()

@pytest.fixture
def cached_context(base_context):
    """Create a cached context with the mock base."""
    return TestCachedGraphContext(base_context=base_context)

@pytest.fixture
def sample_properties():
    """Create sample entity properties."""
    return {"name": "Test Person", "age": 30}

@pytest.fixture
def sample_relation_properties():
    """Create sample relation properties."""
    return {"since": "2024-01-01"}

@pytest.mark.asyncio
async def test_entity_caching(cached_context, base_context, sample_properties):
    """Test entity caching behavior."""
    # Create an entity
    entity = await cached_context.create_entity("person", sample_properties)
    entity_id = entity.id

    # First read should hit the base context
    initial_count = base_context._call_count['get_entity']
    entity1 = await cached_context.get_entity(entity_id)
    assert entity1 is not None
    assert base_context._call_count['get_entity'] > initial_count

    # Second read should hit the cache
    initial_count = base_context._call_count['get_entity']
    entity2 = await cached_context.get_entity(entity_id)
    assert entity2 is not None
    assert entity2 == entity1
    assert base_context._call_count['get_entity'] == initial_count

@pytest.mark.asyncio
async def test_relation_caching(cached_context, base_context, sample_relation_properties):
    """Test relation caching behavior."""
    # Create two entities to relate
    entity1 = await cached_context.create_entity("person", {"name": "Person 1"})
    entity2 = await cached_context.create_entity("person", {"name": "Person 2"})

    # Create a relation
    relation = await cached_context.create_relation(
        "knows",
        entity1.id,
        entity2.id,
        sample_relation_properties
    )
    relation_id = relation.id

    # First read should hit the base context
    initial_count = base_context._call_count['get_relation']
    relation1 = await cached_context.get_relation(relation_id)
    assert relation1 is not None
    assert base_context._call_count['get_relation'] > initial_count

    # Second read should hit the cache
    initial_count = base_context._call_count['get_relation']
    relation2 = await cached_context.get_relation(relation_id)
    assert relation2 is not None
    assert relation2 == relation1
    assert base_context._call_count['get_relation'] == initial_count

@pytest.mark.asyncio
async def test_query_caching(cached_context, base_context, sample_properties):
    """Test query result caching."""
    # Create some test data
    await cached_context.create_entity("person", sample_properties)

    # Execute query first time
    query_spec = QuerySpec(type="person", filter={"name": "Test Person"})
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
async def test_traversal_caching(cached_context, base_context, sample_relation_properties):
    """Test traversal result caching."""
    # Create test data
    entity1 = await cached_context.create_entity("person", {"name": "Person 1"})
    entity2 = await cached_context.create_entity("person", {"name": "Person 2"})
    relation = await cached_context.create_relation(
        "knows",
        entity1.id,
        entity2.id,
        sample_relation_properties
    )

    # Execute traversal first time
    traversal_spec = TraversalSpec(
        relation_type="knows",
        direction="outbound"
    )
    initial_count = base_context._call_count['traverse']
    results1 = await cached_context.traverse(entity1.id, traversal_spec)
    assert len(results1) > 0
    assert base_context._call_count['traverse'] > initial_count

    # Execute same traversal again
    initial_count = base_context._call_count['traverse']
    results2 = await cached_context.traverse(entity1.id, traversal_spec)
    assert results2 == results1
    assert base_context._call_count['traverse'] == initial_count

@pytest.mark.asyncio
async def test_cache_invalidation(cached_context, base_context, sample_properties):
    """Test cache invalidation on updates."""
    # Create and cache an entity
    entity = await cached_context.create_entity("person", sample_properties)
    entity_id = entity.id
    await cached_context.get_entity(entity_id)  # Cache it

    # Update the entity
    updated_properties = {**sample_properties, "name": "Updated Person"}
    await cached_context.update_entity(entity_id, updated_properties)

    # Next read should hit base context
    initial_count = base_context._call_count['get_entity']
    entity = await cached_context.get_entity(entity_id)
    assert entity is not None
    assert entity.properties["name"] == "Updated Person"
    assert base_context._call_count['get_entity'] > initial_count

@pytest.mark.asyncio
async def test_cache_enable_disable(cached_context, base_context, sample_properties):
    """Test enabling and disabling the cache."""
    # Create test data
    entity = await cached_context.create_entity("person", sample_properties)
    entity_id = entity.id

    # Cache should be enabled by default
    initial_count = base_context._call_count['get_entity']
    await cached_context.get_entity(entity_id)
    second_count = base_context._call_count['get_entity']
    await cached_context.get_entity(entity_id)
    assert base_context._call_count['get_entity'] == second_count

    # Disable cache
    cached_context.disable_caching()
    initial_count = base_context._call_count['get_entity']
    await cached_context.get_entity(entity_id)
    assert base_context._call_count['get_entity'] > initial_count

    # Re-enable cache
    cached_context.enable_caching()
    initial_count = base_context._call_count['get_entity']
    await cached_context.get_entity(entity_id)
    second_count = base_context._call_count['get_entity']
    await cached_context.get_entity(entity_id)
    assert base_context._call_count['get_entity'] == second_count

@pytest.mark.asyncio
async def test_transaction_behavior(cached_context, base_context, sample_properties):
    """Test caching behavior within transactions."""
    # Start a transaction
    await cached_context.begin_transaction()

    # Create an entity in the transaction
    entity = await cached_context.create_entity("person", sample_properties)
    entity_id = entity.id

    # Should be able to read the entity within the transaction
    entity1 = await cached_context.get_entity(entity_id)
    assert entity1 is not None
    assert entity1.properties == sample_properties

    # Commit the transaction
    await cached_context.commit_transaction()

    # Should still be able to read the entity after commit
    entity2 = await cached_context.get_entity(entity_id)
    assert entity2 is not None
    assert entity2 == entity1

@pytest.mark.asyncio
async def test_delete_invalidation(cached_context, base_context, sample_properties):
    """Test cache invalidation on delete."""
    # Create and cache an entity
    entity = await cached_context.create_entity("person", sample_properties)
    entity_id = entity.id

    # Cache the entity
    await cached_context.get_entity(entity_id)

    # Delete the entity
    success = await cached_context.delete_entity(entity_id)
    assert success is True

    # Next read should hit base context and return None
    initial_count = base_context._call_count['get_entity']
    result = await cached_context.get_entity(entity_id)
    assert result is None
    assert base_context._call_count['get_entity'] > initial_count