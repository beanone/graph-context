import pytest
from typing import Dict, Any, Optional, List
from graph_context.interface import GraphContext
from graph_context.exceptions import ValidationError

class MockGraphContext(GraphContext):
    """Mock implementation of GraphContext for testing."""

    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relations: Dict[str, Dict[str, Any]] = {}
        self.next_id = 1
        self.in_transaction = False
        self.transaction_entities: Dict[str, Dict[str, Any]] = {}
        self.transaction_relations: Dict[str, Dict[str, Any]] = {}

    def _generate_id(self) -> str:
        id_str = str(self.next_id)
        self.next_id += 1
        return id_str

    async def initialize(self) -> None:
        """Initialize the graph context."""
        pass

    async def cleanup(self) -> None:
        """Clean up resources."""
        self.entities.clear()
        self.relations.clear()
        self.transaction_entities.clear()
        self.transaction_relations.clear()
        self.in_transaction = False

    async def begin_transaction(self) -> None:
        """Begin a transaction."""
        if self.in_transaction:
            raise ValidationError("Nested transactions are not supported")
        self.in_transaction = True
        # Store original state
        self.transaction_entities = {}
        self.transaction_relations = {}
        for entity_id, entity in self.entities.items():
            self.transaction_entities[entity_id] = {
                "type": entity["type"],
                "properties": entity["properties"].copy()
            }
        for relation_id, relation in self.relations.items():
            self.transaction_relations[relation_id] = {
                "type": relation["type"],
                "from_entity": relation["from_entity"],
                "to_entity": relation["to_entity"],
                "properties": relation["properties"].copy() if relation["properties"] else {}
            }

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        if not self.in_transaction:
            raise ValidationError("No active transaction")
        self.entities = self.transaction_entities
        self.relations = self.transaction_relations
        self.in_transaction = False

    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        if not self.in_transaction:
            raise ValidationError("No active transaction")
        self.in_transaction = False

    async def create_entity(
        self,
        entity_type: str,
        properties: Dict[str, Any]
    ) -> str:
        entity_id = self._generate_id()
        entity = {
            "type": entity_type,
            "properties": properties
        }
        if self.in_transaction:
            self.transaction_entities[entity_id] = entity
        else:
            self.entities[entity_id] = entity
        return entity_id

    async def get_entity(
        self,
        entity_id: str
    ) -> Optional[Dict[str, Any]]:
        if self.in_transaction:
            return self.transaction_entities.get(entity_id)
        return self.entities.get(entity_id)

    async def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        entities = self.transaction_entities if self.in_transaction else self.entities
        if entity_id in entities:
            entities[entity_id]["properties"].update(properties)
            return True
        return False

    async def delete_entity(
        self,
        entity_id: str
    ) -> bool:
        entities = self.transaction_entities if self.in_transaction else self.entities
        relations = self.transaction_relations if self.in_transaction else self.relations

        if entity_id in entities:
            # Delete all relations involving this entity
            relations_to_delete = []
            for rel_id, rel in relations.items():
                if rel["from_entity"] == entity_id or rel["to_entity"] == entity_id:
                    relations_to_delete.append(rel_id)

            for rel_id in relations_to_delete:
                del relations[rel_id]

            # Delete the entity
            del entities[entity_id]
            return True
        return False

    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        relation_id = self._generate_id()
        relation = {
            "type": relation_type,
            "from_entity": from_entity,
            "to_entity": to_entity,
            "properties": properties or {}
        }
        if self.in_transaction:
            self.transaction_relations[relation_id] = relation
        else:
            self.relations[relation_id] = relation
        return relation_id

    async def get_relation(
        self,
        relation_id: str
    ) -> Optional[Dict[str, Any]]:
        if self.in_transaction:
            return self.transaction_relations.get(relation_id)
        return self.relations.get(relation_id)

    async def update_relation(
        self,
        relation_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        relations = self.transaction_relations if self.in_transaction else self.relations
        if relation_id in relations:
            relations[relation_id]["properties"].update(properties)
            return True
        return False

    async def delete_relation(
        self,
        relation_id: str
    ) -> bool:
        relations = self.transaction_relations if self.in_transaction else self.relations
        if relation_id in relations:
            del relations[relation_id]
            return True
        return False

    async def query(
        self,
        query_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        results = []
        start = query_spec.get("start")
        relation_type = query_spec.get("relation")
        direction = query_spec.get("direction", "outbound")

        relations = self.transaction_relations if self.in_transaction else self.relations
        for rel_id, rel in relations.items():
            if rel["type"] != relation_type:
                continue
            if direction == "outbound" and rel["from_entity"] == start:
                results.append({"id": rel_id, **rel})
            elif direction == "inbound" and rel["to_entity"] == start:
                results.append({"id": rel_id, **rel})
            elif direction == "any" and (rel["from_entity"] == start or rel["to_entity"] == start):
                results.append({"id": rel_id, **rel})
        return results

    async def traverse(
        self,
        start_entity: str,
        traversal_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        results = []
        max_depth = traversal_spec.get("max_depth", 1)
        relation_types = traversal_spec.get("relation_types", [])
        direction = traversal_spec.get("direction", "any")

        visited_entities = set()
        visited_relations = set()
        current_depth = 0
        current_entities = {start_entity}

        relations = self.transaction_relations if self.in_transaction else self.relations
        while current_depth < max_depth and current_entities:
            next_entities = set()
            for entity_id in current_entities:
                if entity_id in visited_entities:
                    continue
                visited_entities.add(entity_id)

                for rel_id, rel in relations.items():
                    if rel_id in visited_relations:
                        continue

                    if relation_types and rel["type"] not in relation_types:
                        continue

                    if direction in ("outbound", "any") and rel["from_entity"] == entity_id:
                        next_entities.add(rel["to_entity"])
                        results.append({"id": rel_id, **rel})
                        visited_relations.add(rel_id)
                    elif direction in ("inbound", "any") and rel["to_entity"] == entity_id:
                        next_entities.add(rel["from_entity"])
                        results.append({"id": rel_id, **rel})
                        visited_relations.add(rel_id)

            current_entities = next_entities - visited_entities
            current_depth += 1

        return results

@pytest.fixture
async def graph_context():
    return MockGraphContext()

@pytest.mark.asyncio
async def test_create_and_get_entity(graph_context):
    # Test creating an entity
    entity_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace", "birth_year": 1815}
    )
    assert entity_id is not None

    # Test getting the created entity
    entity = await graph_context.get_entity(entity_id)
    assert entity is not None
    assert entity["type"] == "Person"
    assert entity["properties"]["name"] == "Ada Lovelace"
    assert entity["properties"]["birth_year"] == 1815

@pytest.mark.asyncio
async def test_update_entity(graph_context):
    # Create an entity first
    entity_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )

    # Test updating the entity
    success = await graph_context.update_entity(
        entity_id,
        properties={"birth_year": 1815}
    )
    assert success is True

    # Verify the update
    entity = await graph_context.get_entity(entity_id)
    assert entity["properties"]["birth_year"] == 1815
    assert entity["properties"]["name"] == "Ada Lovelace"  # Original property should remain

@pytest.mark.asyncio
async def test_delete_entity(graph_context):
    # Create an entity first
    entity_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )

    # Test deleting the entity
    success = await graph_context.delete_entity(entity_id)
    assert success is True

    # Verify the deletion
    entity = await graph_context.get_entity(entity_id)
    assert entity is None

@pytest.mark.asyncio
async def test_create_and_get_relation(graph_context):
    # Create two entities first
    person_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )
    document_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Notes"}
    )

    # Test creating a relation
    relation_id = await graph_context.create_relation(
        relation_type="authored",
        from_entity=person_id,
        to_entity=document_id,
        properties={"year": 1843}
    )
    assert relation_id is not None

    # Test getting the created relation
    relation = await graph_context.get_relation(relation_id)
    assert relation is not None
    assert relation["type"] == "authored"
    assert relation["from_entity"] == person_id
    assert relation["to_entity"] == document_id
    assert relation["properties"]["year"] == 1843

@pytest.mark.asyncio
async def test_update_relation(graph_context):
    # Create entities and relation first
    person_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )
    document_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Notes"}
    )
    relation_id = await graph_context.create_relation(
        relation_type="authored",
        from_entity=person_id,
        to_entity=document_id
    )

    # Test updating the relation
    success = await graph_context.update_relation(
        relation_id,
        properties={"year": 1843}
    )
    assert success is True

    # Verify the update
    relation = await graph_context.get_relation(relation_id)
    assert relation["properties"]["year"] == 1843

@pytest.mark.asyncio
async def test_delete_relation(graph_context):
    # Create entities and relation first
    person_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )
    document_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Notes"}
    )
    relation_id = await graph_context.create_relation(
        relation_type="authored",
        from_entity=person_id,
        to_entity=document_id
    )

    # Test deleting the relation
    success = await graph_context.delete_relation(relation_id)
    assert success is True

    # Verify the deletion
    relation = await graph_context.get_relation(relation_id)
    assert relation is None

@pytest.mark.asyncio
async def test_query(graph_context):
    # Create test data
    person_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )
    doc1_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Notes"}
    )
    doc2_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Letters"}
    )

    # Create relations
    await graph_context.create_relation(
        relation_type="authored",
        from_entity=person_id,
        to_entity=doc1_id
    )
    await graph_context.create_relation(
        relation_type="authored",
        from_entity=person_id,
        to_entity=doc2_id
    )

    # Test outbound query
    results = await graph_context.query({
        "start": person_id,
        "relation": "authored",
        "direction": "outbound"
    })
    assert len(results) == 2

    # Test inbound query
    results = await graph_context.query({
        "start": doc1_id,
        "relation": "authored",
        "direction": "inbound"
    })
    assert len(results) == 1

@pytest.mark.asyncio
async def test_traverse(graph_context):
    # Create test data
    person1_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )
    person2_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Charles Babbage"}
    )
    doc1_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Notes"}
    )
    doc2_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Letters"}
    )

    # Create relations
    await graph_context.create_relation(
        relation_type="authored",
        from_entity=person1_id,
        to_entity=doc1_id
    )
    await graph_context.create_relation(
        relation_type="collaborated_with",
        from_entity=person1_id,
        to_entity=person2_id
    )
    await graph_context.create_relation(
        relation_type="authored",
        from_entity=person2_id,
        to_entity=doc2_id
    )

    # Test traversal
    results = await graph_context.traverse(
        start_entity=person1_id,
        traversal_spec={
            "max_depth": 2,
            "relation_types": ["authored", "collaborated_with"],
            "direction": "any"
        }
    )
    assert len(results) == 3  # Should find all relations

@pytest.mark.asyncio
async def test_nonexistent_entity_operations(graph_context):
    # Test getting non-existent entity
    entity = await graph_context.get_entity("nonexistent")
    assert entity is None

    # Test updating non-existent entity
    success = await graph_context.update_entity(
        "nonexistent",
        properties={"test": "value"}
    )
    assert success is False

    # Test deleting non-existent entity
    success = await graph_context.delete_entity("nonexistent")
    assert success is False

@pytest.mark.asyncio
async def test_nonexistent_relation_operations(graph_context):
    # Test getting non-existent relation
    relation = await graph_context.get_relation("nonexistent")
    assert relation is None

    # Test updating non-existent relation
    success = await graph_context.update_relation(
        "nonexistent",
        properties={"test": "value"}
    )
    assert success is False

    # Test deleting non-existent relation
    success = await graph_context.delete_relation("nonexistent")
    assert success is False