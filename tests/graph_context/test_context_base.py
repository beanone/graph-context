from typing import Any, Optional, Dict

import pytest

from graph_context.context_base import BaseGraphContext
from graph_context.exceptions import (
    EntityNotFoundError,
    RelationNotFoundError,
    SchemaError,
    TransactionError,
    ValidationError,
)
from graph_context.types.type_base import (
    EntityType,
    PropertyDefinition,
    PropertyType,
    RelationType,
)


class TestGraphContext(BaseGraphContext):
    """Test implementation of BaseGraphContext."""

    def __init__(self):
        super().__init__()
        self._in_transaction = False
        self._entities = {}
        self._relations = {}
        self.next_id = 1
        self._transaction_entities = {}
        self._transaction_relations = {}

    async def setup_default_types(self):
        """Set up default entity and relation types for testing."""
        # Register test entity types
        await self.register_entity_type(EntityType(
            name="Person",
            properties={
                "name": PropertyDefinition(type="string", required=True),
                "birth_year": PropertyDefinition(type="integer", required=False)
            }
        ))
        await self.register_entity_type(EntityType(
            name="Document",
            properties={
                "title": PropertyDefinition(type="string", required=True)
            }
        ))

        # Register test relation types
        await self.register_relation_type(RelationType(
            name="authored",
            from_types=["Person"],
            to_types=["Document"],
            properties={
                "year": PropertyDefinition(type="integer", required=False)
            }
        ))
        await self.register_relation_type(RelationType(
            name="collaborated_with",
            from_types=["Person"],
            to_types=["Person"]
        ))

    def _generate_id(self) -> str:
        id_str = str(self.next_id)
        self.next_id += 1
        return id_str

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Clear in-memory stores
        self._entities.clear()
        self._relations.clear()
        self._transaction_entities.clear()
        self._transaction_relations.clear()

        # Call parent cleanup to handle type registries and transaction state
        await super().cleanup()

    async def begin_transaction(self) -> None:
        """Begin a transaction."""
        if self._in_transaction:
            raise TransactionError("Transaction already in progress", state="active")
        self._in_transaction = True
        # Store original state
        self._transaction_entities = {}
        self._transaction_relations = {}
        for entity_id, entity in self._entities.items():
            self._transaction_entities[entity_id] = {
                "type": entity["type"],
                "properties": entity["properties"].copy()
            }
        for relation_id, relation in self._relations.items():
            self._transaction_relations[relation_id] = {
                "type": relation["type"],
                "from_entity": relation["from_entity"],
                "to_entity": relation["to_entity"],
                "properties": relation["properties"].copy() if relation["properties"] else {}
            }

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        if not self._in_transaction:
            raise TransactionError("No transaction in progress", state="none")
        self._entities = self._transaction_entities
        self._relations = self._transaction_relations
        self._in_transaction = False

    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        if not self._in_transaction:
            raise TransactionError("No transaction in progress", state="none")
        self._in_transaction = False

    async def create_entity(
        self,
        entity_type: str,
        properties: dict[str, Any]
    ) -> str:
        """Create a new entity."""
        if not entity_type:
            raise ValidationError("Entity type cannot be empty")
        if properties is None:
            raise ValidationError("Properties cannot be None")
        return await self._create_entity_impl(entity_type, properties)

    async def get_entity(
        self,
        entity_id: str
    ) -> Optional[dict[str, Any]]:
        """Get an entity by ID."""
        if not entity_id:
            raise ValidationError("Entity ID cannot be empty")
        entity = await self._get_entity_impl(entity_id)
        if entity is None:
            raise EntityNotFoundError(f"Entity {entity_id} not found")
        return entity

    async def update_entity(
        self,
        entity_id: str,
        properties: dict[str, Any]
    ) -> bool:
        """Update an entity."""
        if not entity_id:
            raise ValidationError("Entity ID cannot be empty")
        if properties is None:
            raise ValidationError("Properties cannot be None")
        if not await self._get_entity_impl(entity_id):
            raise EntityNotFoundError(f"Entity {entity_id} not found")
        return await self._update_entity_impl(entity_id, properties)

    async def delete_entity(
        self,
        entity_id: str
    ) -> bool:
        """Delete an entity."""
        if not entity_id:
            raise ValidationError("Entity ID cannot be empty")
        if not await self._get_entity_impl(entity_id):
            raise EntityNotFoundError(f"Entity {entity_id} not found")
        return await self._delete_entity_impl(entity_id)

    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[dict[str, Any]] = None
    ) -> str:
        """Create a new relation.

        Args:
            relation_type: The type of relation to create.
            from_entity: The ID of the source entity.
            to_entity: The ID of the target entity.
            properties: Optional properties for the relation.

        Returns:
            str: The ID of the created relation.

        Raises:
            ValidationError: If any of the input parameters are invalid.
            EntityNotFoundError: If either the source or target entity does not exist.
        """
        if not relation_type:
            raise ValidationError("Relation type cannot be empty")
        if not from_entity:
            raise ValidationError("From entity ID cannot be empty")
        if not to_entity:
            raise ValidationError("To entity ID cannot be empty")
        if not await self._get_entity_impl(from_entity):
            raise EntityNotFoundError(f"From entity {from_entity} not found")
        if not await self._get_entity_impl(to_entity):
            raise EntityNotFoundError(f"To entity {to_entity} not found")
        return await self._create_relation_impl(
            relation_type,
            from_entity,
            to_entity,
            properties
        )

    async def get_relation(
        self,
        relation_id: str
    ) -> Optional[dict[str, Any]]:
        """Get a relation by ID."""
        if not relation_id:
            raise ValidationError("Relation ID cannot be empty")
        relation = await self._get_relation_impl(relation_id)
        if relation is None:
            raise RelationNotFoundError(f"Relation {relation_id} not found")
        return relation

    async def update_relation(
        self,
        relation_id: str,
        properties: dict[str, Any]
    ) -> bool:
        """Update a relation."""
        if not relation_id:
            raise ValidationError("Relation ID cannot be empty")
        if properties is None:
            raise ValidationError("Properties cannot be None")
        if not await self._get_relation_impl(relation_id):
            raise RelationNotFoundError(f"Relation {relation_id} not found")
        return await self._update_relation_impl(relation_id, properties)

    async def delete_relation(
        self,
        relation_id: str
    ) -> bool:
        """Delete a relation."""
        if not relation_id:
            raise ValidationError("Relation ID cannot be empty")
        if not await self._get_relation_impl(relation_id):
            raise RelationNotFoundError(f"Relation {relation_id} not found")
        return await self._delete_relation_impl(relation_id)

    async def query(
        self,
        query_spec: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Execute a query."""
        if query_spec is None:
            raise ValidationError("Query spec cannot be None")
        if not query_spec:
            raise ValidationError("Query spec cannot be empty")
        if "direction" in query_spec and query_spec["direction"] not in ["inbound", "outbound", "any"]:
            raise ValidationError("Invalid direction in query spec")
        return await self._query_impl(query_spec)

    async def traverse(
        self,
        start_entity: str,
        traversal_spec: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Traverse the graph.

        Args:
            start_entity: The ID of the starting entity.
            traversal_spec: The specification for traversal.

        Returns:
            list[dict[str, Any]]: The traversal results.

        Raises:
            ValidationError: If the input parameters are invalid.
        """
        if not start_entity:
            raise ValidationError("Start entity ID cannot be empty")
        if traversal_spec is None:
            raise ValidationError("Traversal spec cannot be None")
        if "max_depth" in traversal_spec and traversal_spec["max_depth"] < 0:
            raise ValidationError("Max depth cannot be negative")
        if "direction" in traversal_spec and traversal_spec["direction"] not in [
            "inbound",
            "outbound",
            "any"
        ]:
            raise ValidationError("Invalid direction in traversal spec")
        return await self._traverse_impl(start_entity, traversal_spec)

    async def _get_entity_impl(self, entity_id: str) -> dict[str, Any] | None:
        """Implementation method to get an entity."""
        if self._in_transaction:
            return self._transaction_entities.get(entity_id)
        return self._entities.get(entity_id)

    async def _create_entity_impl(self, entity_type: str, properties: dict[str, Any]) -> str:
        """Implementation method to create an entity."""
        # Validate entity type and properties
        if entity_type not in self._entity_types:
            raise SchemaError(f"Unknown entity type: {entity_type}")

        # Validate properties against schema
        await self._validate_entity_impl(entity_type, properties)

        entity_id = self._generate_id()
        entity = {
            "type": entity_type,
            "properties": properties
        }
        if self._in_transaction:
            self._transaction_entities[entity_id] = entity
        else:
            self._entities[entity_id] = entity
        return entity_id

    async def _update_entity_impl(self, entity_id: str, properties: dict[str, Any]) -> bool:
        """Implementation method to update an entity."""
        entities = self._transaction_entities if self._in_transaction else self._entities
        if entity_id in entities:
            entities[entity_id]["properties"].update(properties)
            return True
        return False

    async def _delete_entity_impl(self, entity_id: str) -> bool:
        """Implementation method to delete an entity."""
        entities = self._transaction_entities if self._in_transaction else self._entities
        relations = self._transaction_relations if self._in_transaction else self._relations

        if entity_id in entities:
            # Delete the entity
            del entities[entity_id]

            # Delete all relations involving this entity
            relations_to_delete = []
            for rel_id, rel in relations.items():
                if rel["from_entity"] == entity_id or rel["to_entity"] == entity_id:
                    relations_to_delete.append(rel_id)

            for rel_id in relations_to_delete:
                del relations[rel_id]

            return True
        return False

    async def _get_relation_impl(self, relation_id: str) -> dict[str, Any] | None:
        """Implementation method to get a relation."""
        if self._in_transaction:
            return self._transaction_relations.get(relation_id)
        return self._relations.get(relation_id)

    async def _create_relation_impl(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: dict[str, Any]
    ) -> str:
        """Implementation method to create a relation."""
        # Validate relation type
        if relation_type not in self._relation_types:
            raise SchemaError(f"Unknown relation type: {relation_type}")

        # Get entity types
        from_entity_data = self._transaction_entities.get(from_entity) if self._in_transaction else self._entities.get(from_entity)
        to_entity_data = self._transaction_entities.get(to_entity) if self._in_transaction else self._entities.get(to_entity)

        if not from_entity_data or not to_entity_data:
            raise EntityNotFoundError("One or both entities not found")

        # Validate entity types against relation schema
        relation_schema = self._relation_types[relation_type]
        if from_entity_data["type"] not in relation_schema.from_types:
            raise SchemaError(f"Invalid source entity type: {from_entity_data['type']}")
        if to_entity_data["type"] not in relation_schema.to_types:
            raise SchemaError(f"Invalid target entity type: {to_entity_data['type']}")

        # Validate properties
        await self._validate_relation_impl(relation_type, properties or {})

        relation_id = self._generate_id()
        relation = {
            "type": relation_type,
            "from_entity": from_entity,
            "to_entity": to_entity,
            "properties": properties or {}
        }
        if self._in_transaction:
            self._transaction_relations[relation_id] = relation
        else:
            self._relations[relation_id] = relation
        return relation_id

    async def _update_relation_impl(self, relation_id: str, properties: dict[str, Any]) -> bool:
        """Implementation method to update a relation."""
        relations = self._transaction_relations if self._in_transaction else self._relations
        if relation_id in relations:
            relations[relation_id]["properties"].update(properties)
            return True
        return False

    async def _delete_relation_impl(self, relation_id: str) -> bool:
        """Implementation method to delete a relation."""
        relations = self._transaction_relations if self._in_transaction else self._relations
        if relation_id in relations:
            del relations[relation_id]
            return True
        return False

    async def _query_impl(self, query_spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Implementation method to execute a query."""
        results = []
        start = query_spec.get("start")
        relation_type = query_spec.get("relation")
        direction = query_spec.get("direction", "outbound")

        relations = self._transaction_relations if self._in_transaction else self._relations
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

    async def _traverse_impl(self, start_entity: str, traversal_spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Implementation method to execute a traversal."""
        results = []
        max_depth = traversal_spec.get("max_depth", 1)
        relation_types = traversal_spec.get("relation_types", [])
        direction = traversal_spec.get("direction", "any")

        visited_entities = set()
        visited_relations = set()
        current_depth = 0
        current_entities = {start_entity}

        relations = self._transaction_relations if self._in_transaction else self._relations
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

    async def validate_properties(self, properties: dict, property_definitions: dict) -> dict:
        """Validate properties against their definitions."""
        validated = {}
        for name, definition in property_definitions.items():
            if name in properties:
                validated[name] = properties[name]
            elif "default" in definition:
                validated[name] = definition["default"]
            elif definition.get("required", False):
                raise ValidationError(f"Required property missing: {name}")
        return validated

    async def validate_property(self, name: str, value: Any, constraints: dict) -> Any:
        """Validate a single property against its constraints."""
        if "pattern" in constraints:
            if not constraints["pattern"].match(str(value)):
                raise ValidationError(f"Value {value} does not match pattern {constraints['pattern'].pattern}")
        return value

    async def _validate_entity_impl(self, entity_type: str, properties: Dict[str, Any]) -> None:
        """Implementation method to validate an entity against its schema."""
        if entity_type not in self._entity_types:
            raise SchemaError(f"Entity type {entity_type} is not registered")

        schema = self._entity_types[entity_type]
        required_props = {name: prop for name, prop in schema.properties.items() if prop.required}

        # Check for missing required properties
        for prop_name, prop_def in required_props.items():
            if prop_name not in properties:
                raise ValidationError(f"Missing required property: {prop_name}")

        # Validate property types
        for prop_name, value in properties.items():
            if prop_name not in schema.properties:
                raise ValidationError(f"Unknown property: {prop_name}")

            prop_def = schema.properties[prop_name]
            prop_type = prop_def.type

            if prop_type == PropertyType.STRING and not isinstance(value, str):
                raise ValidationError(f"Property {prop_name} must be a string")
            elif prop_type == PropertyType.INTEGER and not isinstance(value, int):
                raise ValidationError(f"Property {prop_name} must be an integer")
            elif prop_type == PropertyType.FLOAT and not isinstance(value, (int, float)):
                raise ValidationError(f"Property {prop_name} must be a number")
            elif prop_type == PropertyType.BOOLEAN and not isinstance(value, bool):
                raise ValidationError(f"Property {prop_name} must be a boolean")

    async def _validate_relation_impl(self, relation_type: str, properties: Dict[str, Any]) -> None:
        """Implementation method to validate a relation against its schema."""
        if relation_type not in self._relation_types:
            raise SchemaError(f"Relation type {relation_type} is not registered")

        schema = self._relation_types[relation_type]
        required_props = {name: prop for name, prop in schema.properties.items() if prop.required}

        # Check for missing required properties
        for prop_name, prop_def in required_props.items():
            if prop_name not in properties:
                raise ValidationError(f"Missing required property: {prop_name}")

        # Validate property types
        for prop_name, value in properties.items():
            if prop_name not in schema.properties:
                raise ValidationError(f"Unknown property: {prop_name}")

            prop_def = schema.properties[prop_name]
            prop_type = prop_def.type

            if prop_type == PropertyType.STRING and not isinstance(value, str):
                raise ValidationError(f"Property {prop_name} must be a string")
            elif prop_type == PropertyType.INTEGER and not isinstance(value, int):
                raise ValidationError(f"Property {prop_name} must be an integer")
            elif prop_type == PropertyType.FLOAT and not isinstance(value, (int, float)):
                raise ValidationError(f"Property {prop_name} must be a number")
            elif prop_type == PropertyType.BOOLEAN and not isinstance(value, bool):
                raise ValidationError(f"Property {prop_name} must be a boolean")

# Start with fixtures
@pytest.fixture
async def empty_graph_context():
    """Create an empty graph context for testing."""
    context = TestGraphContext()
    yield context
    await context.cleanup()

@pytest.fixture
async def graph_context(empty_graph_context):
    """Create a graph context with default types for testing."""
    await empty_graph_context.setup_default_types()
    yield empty_graph_context
    await empty_graph_context.cleanup()

# 1. Test the most basic function first - _check_transaction
@pytest.mark.asyncio
async def test_check_transaction(empty_graph_context):
    """Test _check_transaction method directly for all code paths."""
    # Initial state - no transaction
    assert empty_graph_context._in_transaction is False

    # Case 1: No transaction, required=True (should raise)
    with pytest.raises(TransactionError) as exc_info:
        empty_graph_context._check_transaction(required=True)
    assert "No transaction in progress" in str(exc_info.value)
    assert exc_info.value.details.get("state") == "none"

    # Case 2: No transaction, required=False (should NOT raise)
    empty_graph_context._check_transaction(required=False)  # Should not raise

    # Set transaction state directly since we're testing the basic function
    empty_graph_context._in_transaction = True

    # Case 3: Transaction exists, required=True (should NOT raise)
    empty_graph_context._check_transaction(required=True)  # Should not raise

    # Case 4: Transaction exists, required=False (should raise)
    with pytest.raises(TransactionError) as exc_info:
        empty_graph_context._check_transaction(required=False)
    assert "Transaction already in progress" in str(exc_info.value)
    assert exc_info.value.details.get("state") == "active"

# 2. Test type registration (next most basic functions)
@pytest.mark.asyncio
async def test_register_entity_type(empty_graph_context):
    """Test registering entity types."""
    # Test registering a valid entity type
    entity_type = EntityType(
        name="test",
        properties={
            "name": PropertyDefinition(type="string", required=True)
        }
    )
    await empty_graph_context.register_entity_type(entity_type)
    assert "test" in empty_graph_context._entity_types

    # Test registering a duplicate entity type
    with pytest.raises(SchemaError):
        await empty_graph_context.register_entity_type(entity_type)

@pytest.mark.asyncio
async def test_register_relation_type(empty_graph_context):
    """Test registering relation types."""
    # Register required entity types first
    person_type = EntityType(
        name="Person",
        properties={
            "name": PropertyDefinition(type="string", required=True)
        }
    )
    await empty_graph_context.register_entity_type(person_type)

    # Test registering a valid relation type
    relation_type = RelationType(
        name="knows",
        from_types=["Person"],
        to_types=["Person"]
    )
    await empty_graph_context.register_relation_type(relation_type)
    assert "knows" in empty_graph_context._relation_types

    # Test registering a duplicate relation type
    with pytest.raises(SchemaError):
        await empty_graph_context.register_relation_type(relation_type)

    # Test registering with unknown entity type
    invalid_type = RelationType(
        name="invalid",
        from_types=["Unknown"],
        to_types=["Person"]
    )
    with pytest.raises(SchemaError):
        await empty_graph_context.register_relation_type(invalid_type)

# 3. Test validation functions (depend only on registered types)
@pytest.mark.asyncio
async def test_validate_entity(empty_graph_context):
    """Test entity validation."""
    # Register test type
    entity_type = EntityType(
        name="test",
        properties={
            "required": PropertyDefinition(
                type=PropertyType.STRING,
                required=True
            ),
            "optional": PropertyDefinition(
                type=PropertyType.INTEGER,
                required=False,
                default=42
            )
        }
    )
    await empty_graph_context.register_entity_type(entity_type)

    # Test valid entity
    validated = empty_graph_context.validate_entity(
        "test",
        {"required": "value"}
    )
    assert validated["required"] == "value"
    assert validated["optional"] == 42  # Default value

    # Test missing required property
    with pytest.raises(ValidationError) as exc_info:
        empty_graph_context.validate_entity("test", {})
    assert "Required property missing" in str(exc_info.value)

    # Test unknown property
    with pytest.raises(ValidationError) as exc_info:
        empty_graph_context.validate_entity(
            "test",
            {"required": "value", "unknown": "value"}
        )
    assert "Unknown properties" in str(exc_info.value)

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
async def test_transaction_methods(empty_graph_context):
    """Test the transaction methods directly."""
    # Test begin_transaction
    await empty_graph_context.begin_transaction()
    assert empty_graph_context._in_transaction is True

    # Test commit_transaction
    await empty_graph_context.commit_transaction()
    assert empty_graph_context._in_transaction is False

    # Test rollback_transaction
    await empty_graph_context.begin_transaction()
    assert empty_graph_context._in_transaction is True
    await empty_graph_context.rollback_transaction()
    assert empty_graph_context._in_transaction is False

@pytest.mark.asyncio
async def test_transaction_operations(empty_graph_context):
    """Test basic transaction operations."""
    # Test starting a transaction
    await empty_graph_context.begin_transaction()
    assert empty_graph_context._in_transaction is True

    # Test starting a transaction when one is already active
    with pytest.raises(TransactionError) as exc_info:
        await empty_graph_context.begin_transaction()
    assert "Transaction already in progress" in str(exc_info.value)

    # Test committing a transaction
    await empty_graph_context.commit_transaction()
    assert empty_graph_context._in_transaction is False

@pytest.mark.asyncio
async def test_transaction_error_cases(empty_graph_context):
    """Test error cases in transaction handling."""
    # Test committing when no transaction is active
    with pytest.raises(TransactionError) as exc_info:
        await empty_graph_context.commit_transaction()
    assert "No transaction in progress" in str(exc_info.value)

    # Test rolling back when no transaction is active
    with pytest.raises(TransactionError) as exc_info:
        await empty_graph_context.rollback_transaction()
    assert "No transaction in progress" in str(exc_info.value)

    # Test starting a transaction when one is already active
    await empty_graph_context.begin_transaction()
    with pytest.raises(TransactionError) as exc_info:
        await empty_graph_context.begin_transaction()
    assert "Transaction already in progress" in str(exc_info.value)

@pytest.mark.asyncio
async def test_schema_validation_errors(graph_context):
    """Test schema validation error cases."""
    # Test registering duplicate entity type
    with pytest.raises(SchemaError) as exc_info:
        await graph_context.register_entity_type(EntityType(
            name="Person",  # Already registered in setup
            properties={
                "name": PropertyDefinition(type=PropertyType.STRING, required=True)
            }
        ))
    assert "Entity type already exists" in str(exc_info.value)

    # Test registering duplicate relation type
    with pytest.raises(SchemaError) as exc_info:
        await graph_context.register_relation_type(RelationType(
            name="authored",  # Already registered in setup
            from_types=["Person"],
            to_types=["Document"]
        ))
    assert "Relation type already exists" in str(exc_info.value)

    # Test registering relation type with unknown entity type
    with pytest.raises(SchemaError) as exc_info:
        await graph_context.register_relation_type(RelationType(
            name="new_relation",
            from_types=["UnknownType"],  # Unknown entity type
            to_types=["Document"]
        ))
    assert "Unknown entity type in from_types" in str(exc_info.value)

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
    assert "Missing required property: name" in str(exc_info.value)

    # Test creating entity with unknown property
    with pytest.raises(ValidationError) as exc_info:
        await graph_context.create_entity(
            entity_type="Person",
            properties={
                "name": "Ada Lovelace",
                "unknown_field": "value"  # Unknown property
            }
        )
    assert "Unknown property" in str(exc_info.value)

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
async def test_relation_property_validation(empty_graph_context):
    """Test relation property validation with required and default values."""
    # Register test types
    person_type = EntityType(
        name="person",
        properties={"name": PropertyDefinition(type="string", required=True)}
    )
    await empty_graph_context.register_entity_type(person_type)

    # Register relation type with required and default properties
    relation_type = RelationType(
        name="knows",
        from_types=["person"],
        to_types=["person"],
        properties={
            "since": PropertyDefinition(type="integer", required=True),
            "strength": PropertyDefinition(type="integer", required=False, default=5),
            "notes": PropertyDefinition(type="string", required=False)
        }
    )
    await empty_graph_context.register_relation_type(relation_type)

    # Test missing required property
    with pytest.raises(ValidationError) as exc_info:
        empty_graph_context.validate_relation(
            "knows",
            "person",
            "person",
            properties={"notes": "test"}  # Missing required 'since' property
        )
    assert "Required property missing: since" in str(exc_info.value)

    # Test unknown property
    with pytest.raises(ValidationError) as exc_info:
        empty_graph_context.validate_relation(
            "knows",
            "person",
            "person",
            properties={"since": 2023, "unknown": "value"}
        )
    assert "Unknown properties" in str(exc_info.value)

    # Test default value
    validated = empty_graph_context.validate_relation(
        "knows",
        "person",
        "person",
        properties={"since": 2023}
    )
    assert validated["since"] == 2023
    assert validated["strength"] == 5  # Default value
    assert "notes" not in validated  # Optional with no default

@pytest.mark.asyncio
async def test_transaction_state_transitions(empty_graph_context):
    """Test transaction state transitions and error conditions."""
    # Test initial state
    assert empty_graph_context._in_transaction is False

    # Test successful begin -> commit sequence
    await empty_graph_context.begin_transaction()
    assert empty_graph_context._in_transaction is True
    await empty_graph_context.commit_transaction()
    assert empty_graph_context._in_transaction is False

    # Test successful begin -> rollback sequence
    await empty_graph_context.begin_transaction()
    assert empty_graph_context._in_transaction is True
    await empty_graph_context.rollback_transaction()
    assert empty_graph_context._in_transaction is False

    # Test double begin (should fail)
    await empty_graph_context.begin_transaction()
    with pytest.raises(TransactionError) as exc_info:
        await empty_graph_context.begin_transaction()
    assert "Transaction already in progress" in str(exc_info.value)
    await empty_graph_context.rollback_transaction()

    # Test commit without begin
    with pytest.raises(TransactionError) as exc_info:
        await empty_graph_context.commit_transaction()
    assert "No transaction in progress" in str(exc_info.value)

    # Test rollback without begin
    with pytest.raises(TransactionError) as exc_info:
        await empty_graph_context.rollback_transaction()
    assert "No transaction in progress" in str(exc_info.value)

@pytest.mark.asyncio
async def test_property_default_values(empty_graph_context):
    """Test property default values."""
    # Define properties with defaults
    properties = {
        "required_with_default": {
            "type": "string",
            "required": True,
            "default": "default_value"
        },
        "optional_with_default": {
            "type": "integer",
            "required": False,
            "default": 100
        }
    }

    # Test validation with missing required property (should use default)
    validated = await empty_graph_context.validate_properties({}, properties)
    assert validated["required_with_default"] == "default_value"
    assert validated["optional_with_default"] == 100

    # Test validation with override values
    validated = await empty_graph_context.validate_properties({
        "required_with_default": "override",
        "optional_with_default": 200
    }, properties)
    assert validated["required_with_default"] == "override"
    assert validated["optional_with_default"] == 200

@pytest.mark.asyncio
async def test_comprehensive_validation(empty_graph_context):
    """Test comprehensive validation scenarios."""
    import re

    # Test pattern validation
    pattern = re.compile(r'^[A-Za-z]+$')
    with pytest.raises(ValidationError):
        await empty_graph_context.validate_property("test_pattern", "123", {"pattern": pattern})

@pytest.mark.asyncio
async def test_cleanup(graph_context):
    """Test cleanup behavior."""
    # Setup: Create some entities and relations
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

    # Start a transaction
    await graph_context.begin_transaction()

    # Verify initial state
    assert graph_context._in_transaction is True
    assert len(graph_context._entities) > 0
    assert len(graph_context._relations) > 0
    assert len(graph_context._entity_types) > 0
    assert len(graph_context._relation_types) > 0

    # Cleanup
    await graph_context.cleanup()

    # Verify cleanup
    assert graph_context._in_transaction is False
    assert len(graph_context._entities) == 0
    assert len(graph_context._relations) == 0
    assert len(graph_context._entity_types) == 0
    assert len(graph_context._relation_types) == 0
    assert len(graph_context._transaction_entities) == 0
    assert len(graph_context._transaction_relations) == 0

@pytest.mark.asyncio
async def test_cleanup_idempotent(graph_context):
    """Test that cleanup can be called multiple times safely."""
    # Setup: Create an entity
    await graph_context.create_entity(
        entity_type="Person",
        properties={"name": "Ada Lovelace"}
    )

    # Multiple cleanups should not raise errors
    await graph_context.cleanup()
    await graph_context.cleanup()
    await graph_context.cleanup()

    # Verify final state
    assert graph_context._in_transaction is False
    assert len(graph_context._entities) == 0
    assert len(graph_context._relations) == 0
    assert len(graph_context._entity_types) == 0
    assert len(graph_context._relation_types) == 0

class SimpleBaseGraphContext(BaseGraphContext):
    """Simple implementation of BaseGraphContext for testing transaction methods."""

    async def _get_entity_impl(self, entity_id: str) -> dict[str, Any] | None:
        """Implementation method to get an entity."""
        return None

    async def _create_entity_impl(self, entity_type: str, properties: dict[str, Any]) -> str:
        """Implementation method to create an entity."""
        return "1"

    async def _update_entity_impl(self, entity_id: str, properties: dict[str, Any]) -> bool:
        """Implementation method to update an entity."""
        return True

    async def _delete_entity_impl(self, entity_id: str) -> bool:
        """Implementation method to delete an entity."""
        return True

    async def _get_relation_impl(self, relation_id: str) -> dict[str, Any] | None:
        """Implementation method to get a relation."""
        return None

    async def _create_relation_impl(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: dict[str, Any]
    ) -> str:
        """Implementation method to create a relation."""
        return "1"

    async def _update_relation_impl(self, relation_id: str, properties: dict[str, Any]) -> bool:
        """Implementation method to update a relation."""
        return True

    async def _delete_relation_impl(self, relation_id: str) -> bool:
        """Implementation method to delete a relation."""
        return True

    async def _query_impl(self, query_spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Implementation method to execute a query."""
        return []

    async def _traverse_impl(self, start_entity: str, traversal_spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Implementation method to execute a traversal."""
        return []

@pytest.mark.asyncio
async def test_base_transaction_methods():
    """Test transaction methods directly on BaseGraphContext."""
    context = SimpleBaseGraphContext()

    # Test initial state
    assert context._in_transaction is False

    # Test begin_transaction
    await context.begin_transaction()
    assert context._in_transaction is True

    # Test commit_transaction
    await context.commit_transaction()
    assert context._in_transaction is False

    # Test rollback_transaction
    await context.begin_transaction()
    await context.rollback_transaction()
    assert context._in_transaction is False

    # Test error cases
    await context.begin_transaction()
    with pytest.raises(TransactionError):
        await context.begin_transaction()  # Already in transaction
    await context.rollback_transaction()

    with pytest.raises(TransactionError):
        await context.commit_transaction()  # No transaction

    with pytest.raises(TransactionError):
        await context.rollback_transaction()  # No transaction

@pytest.mark.asyncio
async def test_entity_schema_validation(graph_context):
    """Test entity schema validation."""
    # Register a schema with required properties
    await graph_context.register_entity_type(EntityType(
        name="person",
        properties={
            "name": PropertyDefinition(type=PropertyType.STRING, required=True),
            "age": PropertyDefinition(type=PropertyType.INTEGER, required=False),
            "is_active": PropertyDefinition(type=PropertyType.BOOLEAN, required=False)
        }
    ))

    # Test missing required property
    with pytest.raises(ValidationError, match="Missing required property: name"):
        await graph_context.create_entity("person", {})

    # Test invalid property type
    with pytest.raises(ValidationError, match="Property name must be a string"):
        await graph_context.create_entity("person", {"name": 123})

    # Test unknown property
    with pytest.raises(ValidationError, match="Unknown property: unknown"):
        await graph_context.create_entity("person", {"name": "John", "unknown": "value"})

    # Test valid entity creation
    entity_id = await graph_context.create_entity("person", {"name": "John", "age": 30, "is_active": True})
    assert entity_id is not None

@pytest.mark.asyncio
async def test_relation_schema_validation(graph_context):
    """Test relation schema validation."""
    # Register entity types
    await graph_context.register_entity_type(EntityType(
        name="person",
        properties={
            "name": PropertyDefinition(type=PropertyType.STRING, required=True)
        }
    ))
    await graph_context.register_entity_type(EntityType(
        name="company",
        properties={
            "name": PropertyDefinition(type=PropertyType.STRING, required=True)
        }
    ))

    # Register relation type with required properties
    await graph_context.register_relation_type(RelationType(
        name="works_at",
        from_types=["person"],
        to_types=["company"],
        properties={
            "role": PropertyDefinition(type=PropertyType.STRING, required=True),
            "salary": PropertyDefinition(type=PropertyType.FLOAT, required=False),
            "is_remote": PropertyDefinition(type=PropertyType.BOOLEAN, required=False)
        }
    ))

    # Create entities for testing
    person_id = await graph_context.create_entity("person", {"name": "John"})
    company_id = await graph_context.create_entity("company", {"name": "Acme Inc"})

    # Test missing required property
    with pytest.raises(ValidationError, match="Missing required property: role"):
        await graph_context.create_relation("works_at", person_id, company_id, {})

    # Test invalid property type
    with pytest.raises(ValidationError, match="Property role must be a string"):
        await graph_context.create_relation("works_at", person_id, company_id, {"role": 123})

    # Test unknown property
    with pytest.raises(ValidationError, match="Unknown property: unknown"):
        await graph_context.create_relation(
            "works_at",
            person_id,
            company_id,
            {"role": "Developer", "unknown": "value"}
        )

    # Test valid relation creation
    relation_id = await graph_context.create_relation(
        "works_at",
        person_id,
        company_id,
        {"role": "Developer", "salary": 100000.0, "is_remote": True}
    )
    assert relation_id is not None