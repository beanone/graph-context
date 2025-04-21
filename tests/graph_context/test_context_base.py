import pytest
from typing import Dict, Any, Optional, List
from graph_context.context_base import BaseGraphContext
from graph_context.exceptions import (
    EntityNotFoundError,
    RelationNotFoundError,
    ValidationError,
    SchemaError
)
from graph_context.types.type_base import (
    EntityType,
    PropertyDefinition,
    RelationType
)

class TestGraphContext(BaseGraphContext):
    """Test implementation of BaseGraphContext."""

    def __init__(self):
        super().__init__()
        self._in_transaction = False
        self._transaction_entities = {}
        self._transaction_relations = {}
        self.entities = {}
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relations: Dict[str, Dict[str, Any]] = {}
        self.next_id = 1
        self.in_transaction = False
        self.transaction_entities: Dict[str, Dict[str, Any]] = {}
        self.transaction_relations: Dict[str, Dict[str, Any]] = {}

        # Register test entity types
        self.register_entity_type(EntityType(
            name="Person",
            properties={
                "name": PropertyDefinition(type="string", required=True),
                "birth_year": PropertyDefinition(type="integer", required=False)
            }
        ))
        self.register_entity_type(EntityType(
            name="Document",
            properties={
                "title": PropertyDefinition(type="string", required=True)
            }
        ))

        # Register test relation types
        self.register_relation_type(RelationType(
            name="authored",
            from_types=["Person"],
            to_types=["Document"],
            properties={
                "year": PropertyDefinition(type="integer", required=False)
            }
        ))
        self.register_relation_type(RelationType(
            name="collaborated_with",
            from_types=["Person"],
            to_types=["Person"]
        ))

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
        """Create a new entity."""
        if not entity_type:
            raise ValidationError("Entity type cannot be empty")
        if properties is None:
            raise ValidationError("Properties cannot be None")
        return await self._create_entity_internal(entity_type, properties)

    async def get_entity(
        self,
        entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get an entity by ID."""
        if not entity_id:
            raise ValidationError("Entity ID cannot be empty")
        entity = await self._get_entity_internal(entity_id)
        if entity is None:
            raise EntityNotFoundError(f"Entity {entity_id} not found")
        return entity

    async def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Update an entity."""
        if not entity_id:
            raise ValidationError("Entity ID cannot be empty")
        if properties is None:
            raise ValidationError("Properties cannot be None")
        if not await self._get_entity_internal(entity_id):
            raise EntityNotFoundError(f"Entity {entity_id} not found")
        return await self._update_entity_internal(entity_id, properties)

    async def delete_entity(
        self,
        entity_id: str
    ) -> bool:
        """Delete an entity."""
        if not entity_id:
            raise ValidationError("Entity ID cannot be empty")
        if not await self._get_entity_internal(entity_id):
            raise EntityNotFoundError(f"Entity {entity_id} not found")
        return await self._delete_entity_internal(entity_id)

    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new relation."""
        if not relation_type:
            raise ValidationError("Relation type cannot be empty")
        if not from_entity:
            raise ValidationError("From entity ID cannot be empty")
        if not to_entity:
            raise ValidationError("To entity ID cannot be empty")
        if not await self._get_entity_internal(from_entity):
            raise EntityNotFoundError(f"From entity {from_entity} not found")
        if not await self._get_entity_internal(to_entity):
            raise EntityNotFoundError(f"To entity {to_entity} not found")
        return await self._create_relation_internal(relation_type, from_entity, to_entity, properties)

    async def get_relation(
        self,
        relation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a relation by ID."""
        if not relation_id:
            raise ValidationError("Relation ID cannot be empty")
        relation = await self._get_relation_internal(relation_id)
        if relation is None:
            raise RelationNotFoundError(f"Relation {relation_id} not found")
        return relation

    async def update_relation(
        self,
        relation_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Update a relation."""
        if not relation_id:
            raise ValidationError("Relation ID cannot be empty")
        if properties is None:
            raise ValidationError("Properties cannot be None")
        if not await self._get_relation_internal(relation_id):
            raise RelationNotFoundError(f"Relation {relation_id} not found")
        return await self._update_relation_internal(relation_id, properties)

    async def delete_relation(
        self,
        relation_id: str
    ) -> bool:
        """Delete a relation."""
        if not relation_id:
            raise ValidationError("Relation ID cannot be empty")
        if not await self._get_relation_internal(relation_id):
            raise RelationNotFoundError(f"Relation {relation_id} not found")
        return await self._delete_relation_internal(relation_id)

    async def query(
        self,
        query_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute a query."""
        if query_spec is None:
            raise ValidationError("Query spec cannot be None")
        if not query_spec:
            raise ValidationError("Query spec cannot be empty")
        if "direction" in query_spec and query_spec["direction"] not in ["inbound", "outbound", "any"]:
            raise ValidationError("Invalid direction in query spec")
        return await self._query_internal(query_spec)

    async def traverse(
        self,
        start_entity: str,
        traversal_spec: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Traverse the graph."""
        if not start_entity:
            raise ValidationError("Start entity ID cannot be empty")
        if traversal_spec is None:
            raise ValidationError("Traversal spec cannot be None")
        if "max_depth" in traversal_spec and traversal_spec["max_depth"] < 0:
            raise ValidationError("Max depth cannot be negative")
        if "direction" in traversal_spec and traversal_spec["direction"] not in ["inbound", "outbound", "any"]:
            raise ValidationError("Invalid direction in traversal spec")
        return await self._traverse_internal(start_entity, traversal_spec)

    async def _create_entity_internal(
        self,
        entity_type: str,
        properties: Dict[str, Any]
    ) -> str:
        # Validate entity using parent class
        validated_props = self.validate_entity(entity_type, properties)

        entity_id = self._generate_id()
        entity = {
            "type": entity_type,
            "properties": validated_props
        }

        if self.in_transaction:
            self.transaction_entities[entity_id] = entity
        else:
            self.entities[entity_id] = entity
        return entity_id

    async def _get_entity_internal(
        self,
        entity_id: str
    ) -> Optional[Dict[str, Any]]:
        if self.in_transaction:
            return self.transaction_entities.get(entity_id)
        return self.entities.get(entity_id)

    async def _update_entity_internal(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        entities = self.transaction_entities if self.in_transaction else self.entities
        if entity_id in entities:
            entity = entities[entity_id]
            # Validate updated properties using parent class
            validated_props = self.validate_entity(entity["type"], {**entity["properties"], **properties})
            entity["properties"] = validated_props
            return True
        return False

    async def _delete_entity_internal(
        self,
        entity_id: str
    ) -> bool:
        """Delete an entity and all its relations."""
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

    async def _create_relation_internal(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        # Get entity types for validation
        from_entity_data = await self._get_entity_internal(from_entity)
        to_entity_data = await self._get_entity_internal(to_entity)
        if not from_entity_data or not to_entity_data:
            raise EntityNotFoundError("Source or target entity not found")

        # Validate relation using parent class
        validated_props = self.validate_relation(
            relation_type,
            from_entity_data["type"],
            to_entity_data["type"],
            properties or {}
        )

        relation_id = self._generate_id()
        relation = {
            "type": relation_type,
            "from_entity": from_entity,
            "to_entity": to_entity,
            "properties": validated_props
        }

        if self.in_transaction:
            self.transaction_relations[relation_id] = relation
        else:
            self.relations[relation_id] = relation
        return relation_id

    async def _get_relation_internal(
        self,
        relation_id: str
    ) -> Optional[Dict[str, Any]]:
        if self.in_transaction:
            return self.transaction_relations.get(relation_id)
        return self.relations.get(relation_id)

    async def _update_relation_internal(
        self,
        relation_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        relations = self.transaction_relations if self.in_transaction else self.relations
        if relation_id in relations:
            relation = relations[relation_id]
            # Get entity types for validation
            from_entity_data = await self._get_entity_internal(relation["from_entity"])
            to_entity_data = await self._get_entity_internal(relation["to_entity"])
            if not from_entity_data or not to_entity_data:
                raise EntityNotFoundError("Source or target entity not found")

            # Validate updated properties using parent class
            validated_props = self.validate_relation(
                relation["type"],
                from_entity_data["type"],
                to_entity_data["type"],
                {**relation["properties"], **properties}
            )
            relation["properties"] = validated_props
            return True
        return False

    async def _delete_relation_internal(
        self,
        relation_id: str
    ) -> bool:
        relations = self.transaction_relations if self.in_transaction else self.relations
        if relation_id in relations:
            del relations[relation_id]
            return True
        return False

    async def _query_internal(
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

    async def _traverse_internal(
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
    return TestGraphContext()

@pytest.mark.asyncio
async def test_create_entity_validation(graph_context):
    # Test with valid entity
    entity_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace", "birth_year": 1815}
    )
    assert entity_id is not None

    # Test with invalid entity type
    with pytest.raises(ValidationError):
        await graph_context.create_entity(
            entity_type="",  # Empty type
            properties={"name": "Invalid"}
        )

    # Test with invalid properties
    with pytest.raises(ValidationError):
        await graph_context.create_entity(
            entity_type="Person",
            properties=None  # None properties
        )

@pytest.mark.asyncio
async def test_get_entity_validation(graph_context):
    # Test with invalid entity ID
    with pytest.raises(ValidationError):
        await graph_context.get_entity("")  # Empty ID

    # Test with non-existent entity
    with pytest.raises(EntityNotFoundError):
        await graph_context.get_entity("nonexistent")

@pytest.mark.asyncio
async def test_update_entity_validation(graph_context):
    # Create test entity
    entity_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )

    # Test with invalid entity ID
    with pytest.raises(ValidationError):
        await graph_context.update_entity(
            "",  # Empty ID
            properties={"test": "value"}
        )

    # Test with invalid properties
    with pytest.raises(ValidationError):
        await graph_context.update_entity(
            entity_id,
            properties=None  # None properties
        )

    # Test with non-existent entity
    with pytest.raises(EntityNotFoundError):
        await graph_context.update_entity(
            "nonexistent",
            properties={"test": "value"}
        )

@pytest.mark.asyncio
async def test_delete_entity_validation(graph_context):
    # Test with invalid entity ID
    with pytest.raises(ValidationError):
        await graph_context.delete_entity("")  # Empty ID

    # Test with non-existent entity
    with pytest.raises(EntityNotFoundError):
        await graph_context.delete_entity("nonexistent")

@pytest.mark.asyncio
async def test_create_relation_validation(graph_context):
    # Create test entities
    person_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )
    document_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Notes"}
    )

    # Test with valid relation
    relation_id = await graph_context.create_relation(
        relation_type="authored",
        from_entity=person_id,
        to_entity=document_id,
        properties={"year": 1843}
    )
    assert relation_id is not None

    # Test with invalid relation type
    with pytest.raises(ValidationError):
        await graph_context.create_relation(
            relation_type="",  # Empty type
            from_entity=person_id,
            to_entity=document_id
        )

    # Test with invalid from_entity
    with pytest.raises(ValidationError):
        await graph_context.create_relation(
            relation_type="authored",
            from_entity="",  # Empty ID
            to_entity=document_id
        )

    # Test with invalid to_entity
    with pytest.raises(ValidationError):
        await graph_context.create_relation(
            relation_type="authored",
            from_entity=person_id,
            to_entity=""  # Empty ID
        )

    # Test with non-existent from_entity
    with pytest.raises(EntityNotFoundError):
        await graph_context.create_relation(
            relation_type="authored",
            from_entity="nonexistent",
            to_entity=document_id
        )

    # Test with non-existent to_entity
    with pytest.raises(EntityNotFoundError):
        await graph_context.create_relation(
            relation_type="authored",
            from_entity=person_id,
            to_entity="nonexistent"
        )

@pytest.mark.asyncio
async def test_get_relation_validation(graph_context):
    # Test with invalid relation ID
    with pytest.raises(ValidationError):
        await graph_context.get_relation("")  # Empty ID

    # Test with non-existent relation
    with pytest.raises(RelationNotFoundError):
        await graph_context.get_relation("nonexistent")

@pytest.mark.asyncio
async def test_update_relation_validation(graph_context):
    # Create test entities and relation
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

    # Test with invalid relation ID
    with pytest.raises(ValidationError):
        await graph_context.update_relation(
            "",  # Empty ID
            properties={"test": "value"}
        )

    # Test with invalid properties
    with pytest.raises(ValidationError):
        await graph_context.update_relation(
            relation_id,
            properties=None  # None properties
        )

    # Test with non-existent relation
    with pytest.raises(RelationNotFoundError):
        await graph_context.update_relation(
            "nonexistent",
            properties={"test": "value"}
        )

@pytest.mark.asyncio
async def test_delete_relation_validation(graph_context):
    # Test with invalid relation ID
    with pytest.raises(ValidationError):
        await graph_context.delete_relation("")  # Empty ID

    # Test with non-existent relation
    with pytest.raises(RelationNotFoundError):
        await graph_context.delete_relation("nonexistent")

@pytest.mark.asyncio
async def test_query_validation(graph_context):
    # Test with invalid query spec
    with pytest.raises(ValidationError):
        await graph_context.query(None)  # None query spec

    # Test with missing required fields
    with pytest.raises(ValidationError):
        await graph_context.query({})  # Empty query spec

    # Test with invalid direction
    with pytest.raises(ValidationError):
        await graph_context.query({
            "start": "1",
            "relation": "authored",
            "direction": "invalid"  # Invalid direction
        })

@pytest.mark.asyncio
async def test_traverse_validation(graph_context):
    # Test with invalid start entity
    with pytest.raises(ValidationError):
        await graph_context.traverse(
            "",  # Empty ID
            traversal_spec={"max_depth": 1}
        )

    # Test with invalid traversal spec
    with pytest.raises(ValidationError):
        await graph_context.traverse(
            "1",
            traversal_spec=None  # None traversal spec
        )

    # Test with invalid max_depth
    with pytest.raises(ValidationError):
        await graph_context.traverse(
            "1",
            traversal_spec={"max_depth": -1}  # Invalid depth
        )

    # Test with invalid direction
    with pytest.raises(ValidationError):
        await graph_context.traverse(
            "1",
            traversal_spec={
                "max_depth": 1,
                "direction": "invalid"  # Invalid direction
            }
        )

@pytest.mark.asyncio
async def test_complex_operations(graph_context):
    # Create test entities
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
    rel1_id = await graph_context.create_relation(
        relation_type="authored",
        from_entity=person1_id,
        to_entity=doc1_id
    )
    rel2_id = await graph_context.create_relation(
        relation_type="collaborated_with",
        from_entity=person1_id,
        to_entity=person2_id
    )
    rel3_id = await graph_context.create_relation(
        relation_type="authored",
        from_entity=person2_id,
        to_entity=doc2_id
    )

    # Test complex query
    results = await graph_context.query({
        "start": person1_id,
        "relation": "authored",
        "direction": "outbound"
    })
    assert len(results) == 1
    assert results[0]["from_entity"] == person1_id
    assert results[0]["to_entity"] == doc1_id

    # Test complex traversal
    results = await graph_context.traverse(
        start_entity=person1_id,
        traversal_spec={
            "max_depth": 2,
            "relation_types": ["authored", "collaborated_with"],
            "direction": "any"
        }
    )
    assert len(results) == 3  # Should find all relations

    # Test cascading deletes
    await graph_context.delete_entity(person1_id)

    # Verify relations are deleted
    with pytest.raises(RelationNotFoundError):
        await graph_context.get_relation(rel1_id)
    with pytest.raises(RelationNotFoundError):
        await graph_context.get_relation(rel2_id)

    # Verify other relations still exist
    relation = await graph_context.get_relation(rel3_id)
    assert relation is not None

@pytest.mark.asyncio
async def test_transaction_operations(graph_context):
    """Test comprehensive transaction operations including commit, rollback, and error cases."""
    # Initial state check
    person_name = "Ada Lovelace"
    doc_title = "Notes on the Analytical Engine"

    # 1. Test successful transaction with multiple operations
    await graph_context.begin_transaction()

    # Create entities and relations in transaction
    person_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": person_name}
    )
    doc_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": doc_title}
    )
    relation_id = await graph_context.create_relation(
        relation_type="authored",
        from_entity=person_id,
        to_entity=doc_id,
        properties={"year": 1843}
    )

    # Verify data is accessible within transaction
    person = await graph_context.get_entity(person_id)
    assert person["properties"]["name"] == person_name

    # Commit transaction and verify persistence
    await graph_context.commit_transaction()
    person = await graph_context.get_entity(person_id)
    doc = await graph_context.get_entity(doc_id)
    relation = await graph_context.get_relation(relation_id)
    assert person["properties"]["name"] == person_name
    assert doc["properties"]["title"] == doc_title
    assert relation["type"] == "authored"

    # 2. Test transaction rollback
    await graph_context.begin_transaction()

    # Modify existing entity
    await graph_context.update_entity(
        person_id,
        properties={"name": "Modified Name", "birth_year": 1815}
    )

    # Create new entity in transaction
    new_doc_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Draft Notes"}
    )

    # Verify changes are visible within transaction
    modified_person = await graph_context.get_entity(person_id)
    assert modified_person["properties"]["name"] == "Modified Name"
    new_doc = await graph_context.get_entity(new_doc_id)
    assert new_doc["properties"]["title"] == "Draft Notes"

    # Rollback transaction
    await graph_context.rollback_transaction()

    # Verify original state is preserved
    original_person = await graph_context.get_entity(person_id)
    assert original_person["properties"]["name"] == person_name
    assert "birth_year" not in original_person["properties"]

    # Verify new entity was not persisted
    with pytest.raises(EntityNotFoundError):
        await graph_context.get_entity(new_doc_id)

    # 3. Test transaction error cases
    # Test nested transactions
    await graph_context.begin_transaction()
    with pytest.raises(ValidationError, match="Nested transactions are not supported"):
        await graph_context.begin_transaction()
    await graph_context.rollback_transaction()

    # Test operations without transaction
    with pytest.raises(ValidationError, match="No active transaction"):
        await graph_context.commit_transaction()

    with pytest.raises(ValidationError, match="No active transaction"):
        await graph_context.rollback_transaction()

    # 4. Test transaction isolation
    # Create initial entity
    initial_doc_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Original Document"}
    )

    # Start transaction and modify entity
    await graph_context.begin_transaction()
    await graph_context.update_entity(
        initial_doc_id,
        properties={"title": "Modified Document"}
    )

    # Verify modified state in transaction
    modified_doc = await graph_context.get_entity(initial_doc_id)
    assert modified_doc["properties"]["title"] == "Modified Document"

    # Rollback and verify isolation
    await graph_context.rollback_transaction()
    final_doc = await graph_context.get_entity(initial_doc_id)
    assert final_doc["properties"]["title"] == "Original Document"

@pytest.mark.asyncio
async def test_schema_validation_errors(graph_context):
    """Test schema validation error cases."""
    # Test registering duplicate entity type
    with pytest.raises(SchemaError) as exc_info:
        graph_context.register_entity_type(EntityType(
            name="Person",  # Already registered
            properties={"name": PropertyDefinition(type="string", required=True)}
        ))
    assert "Entity type already exists" in str(exc_info.value)

    # Test registering duplicate relation type
    with pytest.raises(SchemaError) as exc_info:
        graph_context.register_relation_type(RelationType(
            name="authored",  # Already registered
            from_types=["Person"],
            to_types=["Document"]
        ))
    assert "Relation type already exists" in str(exc_info.value)

    # Test registering relation type with unknown entity type
    with pytest.raises(SchemaError) as exc_info:
        graph_context.register_relation_type(RelationType(
            name="new_relation",
            from_types=["UnknownType"],  # Unknown entity type
            to_types=["Document"]
        ))
    assert "Unknown entity type in from_types" in str(exc_info.value)

    with pytest.raises(SchemaError) as exc_info:
        graph_context.register_relation_type(RelationType(
            name="new_relation",
            from_types=["Person"],
            to_types=["UnknownType"]  # Unknown entity type
        ))
    assert "Unknown entity type in to_types" in str(exc_info.value)

@pytest.mark.asyncio
async def test_property_validation_errors(graph_context):
    """Test property validation error cases."""
    # Test creating entity with unknown type
    with pytest.raises(SchemaError) as exc_info:
        await graph_context.create_entity(
            entity_type="UnknownType",
            properties={"name": "Test"}
        )
    assert "Unknown entity type" in str(exc_info.value)

    # Test creating entity with missing required property
    with pytest.raises(ValidationError) as exc_info:
        await graph_context.create_entity(
            entity_type="Person",
            properties={"birth_year": 1815}  # Missing required "name" property
        )
    assert "Required property missing" in str(exc_info.value)

    # Test creating entity with unknown property
    with pytest.raises(ValidationError) as exc_info:
        await graph_context.create_entity(
            entity_type="Person",
            properties={
                "name": "Ada Lovelace",
                "unknown_field": "value"  # Unknown property
            }
        )
    assert "Unknown properties" in str(exc_info.value)

@pytest.mark.asyncio
async def test_relation_validation_errors(graph_context):
    """Test relation validation error cases."""
    # Create test entities
    person_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )
    doc_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Notes"}
    )

    # Test creating relation with unknown type
    with pytest.raises(SchemaError) as exc_info:
        await graph_context.create_relation(
            relation_type="unknown_relation",
            from_entity=person_id,
            to_entity=doc_id
        )
    assert "Unknown relation type" in str(exc_info.value)

    # Test creating relation with invalid source entity type
    with pytest.raises(SchemaError) as exc_info:
        await graph_context.create_relation(
            relation_type="authored",
            from_entity=doc_id,  # Document can't author
            to_entity=doc_id
        )
    assert "Invalid source entity type" in str(exc_info.value)

    # Test creating relation with invalid target entity type
    with pytest.raises(SchemaError) as exc_info:
        await graph_context.create_relation(
            relation_type="authored",
            from_entity=person_id,
            to_entity=person_id  # Person can't be authored
        )
    assert "Invalid target entity type" in str(exc_info.value)

@pytest.mark.asyncio
async def test_transaction_error_cases(graph_context):
    """Test transaction error cases."""
    # Test nested transactions
    await graph_context.begin_transaction()
    with pytest.raises(ValidationError) as exc_info:
        await graph_context.begin_transaction()
    assert "Nested transactions are not supported" in str(exc_info.value)
    await graph_context.rollback_transaction()

    # Test committing without active transaction
    with pytest.raises(ValidationError) as exc_info:
        await graph_context.commit_transaction()
    assert "No active transaction" in str(exc_info.value)

    # Test rolling back without active transaction
    with pytest.raises(ValidationError) as exc_info:
        await graph_context.rollback_transaction()
    assert "No active transaction" in str(exc_info.value)

@pytest.mark.asyncio
async def test_property_validation_with_defaults(graph_context):
    """Test property validation with default values."""
    # Register entity type with default value
    graph_context.register_entity_type(EntityType(
        name="Task",
        properties={
            "title": PropertyDefinition(type="string", required=True),
            "status": PropertyDefinition(type="string", required=False, default="pending")
        }
    ))

    # Create entity without optional property
    entity_id = await graph_context.create_entity(
        entity_type="Task",
        properties={"title": "Test Task"}
    )

    # Verify default value was set
    entity = await graph_context.get_entity(entity_id)
    assert entity["properties"]["status"] == "pending"

@pytest.mark.asyncio
async def test_relation_validation_with_properties(graph_context):
    """Test relation validation with properties."""
    # Register relation type with properties
    graph_context.register_relation_type(RelationType(
        name="reviewed",
        from_types=["Person"],
        to_types=["Document"],
        properties={
            "rating": PropertyDefinition(type="integer", required=True),
            "comment": PropertyDefinition(type="string", required=False)
        }
    ))

    # Create test entities
    person_id = await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )
    doc_id = await graph_context.create_entity(
        entity_type="Document",
        properties={"title": "Notes"}
    )

    # Test creating relation with missing required property
    with pytest.raises(ValidationError) as exc_info:
        await graph_context.create_relation(
            relation_type="reviewed",
            from_entity=person_id,
            to_entity=doc_id,
            properties={"comment": "Great work!"}  # Missing required "rating"
        )
    assert "Required property missing" in str(exc_info.value)

    # Test creating relation with unknown property
    with pytest.raises(ValidationError) as exc_info:
        await graph_context.create_relation(
            relation_type="reviewed",
            from_entity=person_id,
            to_entity=doc_id,
            properties={
                "rating": 5,
                "unknown_field": "value"  # Unknown property
            }
        )
    assert "Unknown properties" in str(exc_info.value)

    # Test creating relation with valid properties
    relation_id = await graph_context.create_relation(
        relation_type="reviewed",
        from_entity=person_id,
        to_entity=doc_id,
        properties={
            "rating": 5,
            "comment": "Excellent work!"
        }
    )
    assert relation_id is not None