"""
Tests for the integration between BaseGraphContext, GraphStoreFactory, and GraphStore.

These tests ensure that BaseGraphContext correctly uses the GraphStoreFactory to
create store instances and interacts with them through the GraphStore interface.
"""

from datetime import UTC, datetime
from typing import Any, Optional
from unittest import mock

import pytest

from graph_context.context_base import BaseGraphContext
from graph_context.factory import GraphStoreFactory
from graph_context.interfaces.store import GraphStore
from graph_context.types.type_base import (
    Entity,
    EntityType,
    PropertyDefinition,
    Relation,
    RelationType,
)


class MockGraphStore(GraphStore):
    """Mock implementation of GraphStore for testing."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with config."""
        self.config = config
        self.entities = {}
        self.relations = {}
        self.transaction_active = False

        # Track method calls for verification
        self.method_calls = []

    def _get_current_time(self) -> datetime:
        """Get the current time with UTC timezone for timestamps."""
        return datetime.now(UTC)

    async def create_entity(self, entity_type: str, properties: dict[str, Any]) -> str:
        """Mock create_entity method."""
        self.method_calls.append(("create_entity", entity_type, properties))
        entity_id = f"e-{len(self.entities) + 1}"
        self.entities[entity_id] = Entity(
            id=entity_id,
            type=entity_type,
            properties=properties,
            created_at=self._get_current_time(),
            updated_at=self._get_current_time(),
        )
        return entity_id

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Mock get_entity method."""
        self.method_calls.append(("get_entity", entity_id))
        return self.entities.get(entity_id)

    async def update_entity(self, entity_id: str, properties: dict[str, Any]) -> bool:
        """Mock update_entity method."""
        self.method_calls.append(("update_entity", entity_id, properties))
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            updated_props = {**entity.properties, **properties}
            self.entities[entity_id] = Entity(
                id=entity_id,
                type=entity.type,
                properties=updated_props,
                created_at=entity.created_at,
                updated_at=self._get_current_time(),
            )
            return True
        return False

    async def delete_entity(self, entity_id: str) -> bool:
        """Mock delete_entity method."""
        self.method_calls.append(("delete_entity", entity_id))
        if entity_id in self.entities:
            del self.entities[entity_id]
            # Delete any relations involving this entity
            to_delete = []
            for rel_id, rel in self.relations.items():
                if rel.from_entity == entity_id or rel.to_entity == entity_id:
                    to_delete.append(rel_id)
            for rel_id in to_delete:
                del self.relations[rel_id]
            return True
        return False

    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> str:
        """Mock create_relation method."""
        self.method_calls.append(
            ("create_relation", relation_type, from_entity, to_entity, properties)
        )
        relation_id = f"r-{len(self.relations) + 1}"
        self.relations[relation_id] = Relation(
            id=relation_id,
            type=relation_type,
            from_entity=from_entity,
            to_entity=to_entity,
            properties=properties or {},
            created_at=self._get_current_time(),
            updated_at=self._get_current_time(),
        )
        return relation_id

    async def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Mock get_relation method."""
        self.method_calls.append(("get_relation", relation_id))
        return self.relations.get(relation_id)

    async def update_relation(
        self, relation_id: str, properties: dict[str, Any]
    ) -> bool:
        """Mock update_relation method."""
        self.method_calls.append(("update_relation", relation_id, properties))
        if relation_id in self.relations:
            relation = self.relations[relation_id]
            updated_props = {**relation.properties, **properties}
            self.relations[relation_id] = Relation(
                id=relation_id,
                type=relation.type,
                from_entity=relation.from_entity,
                to_entity=relation.to_entity,
                properties=updated_props,
                created_at=relation.created_at,
                updated_at=self._get_current_time(),
            )
            return True
        return False

    async def delete_relation(self, relation_id: str) -> bool:
        """Mock delete_relation method."""
        self.method_calls.append(("delete_relation", relation_id))
        if relation_id in self.relations:
            del self.relations[relation_id]
            return True
        return False

    async def query(self, query_spec: dict[str, Any]) -> list[Entity]:
        """Mock query method."""
        self.method_calls.append(("query", query_spec))
        return list(self.entities.values())

    async def traverse(
        self, start_entity: str, traversal_spec: dict[str, Any]
    ) -> list[Entity]:
        """Mock traverse method."""
        self.method_calls.append(("traverse", start_entity, traversal_spec))
        return list(self.entities.values())

    async def begin_transaction(self) -> None:
        """Mock begin_transaction method."""
        self.method_calls.append(("begin_transaction",))
        self.transaction_active = True

    async def commit_transaction(self) -> None:
        """Mock commit_transaction method."""
        self.method_calls.append(("commit_transaction",))
        self.transaction_active = False

    async def rollback_transaction(self) -> None:
        """Mock rollback_transaction method."""
        self.method_calls.append(("rollback_transaction",))
        self.transaction_active = False


class TestBaseGraphContextStoreIntegration:
    """Test BaseGraphContext integration with GraphStoreFactory and GraphStore."""

    def setup_method(self):
        """Set up test environment."""
        # Save original store_types
        self.original_store_types = GraphStoreFactory._store_types.copy()

        # Register our mock store
        GraphStoreFactory.register_store_type("mock", MockGraphStore)

        # Patch the GraphStoreFactory._load_config method to return our mock config
        self.load_config_patcher = mock.patch(
            "graph_context.factory.GraphStoreFactory._load_config",
            return_value=mock.MagicMock(type="mock", config={"test": "config"}),
        )
        self.load_config_patcher.start()

    def teardown_method(self):
        """Clean up test environment."""
        # Restore original store types
        GraphStoreFactory._store_types = self.original_store_types

        # Stop the patcher
        self.load_config_patcher.stop()

    @pytest.mark.asyncio
    async def test_basegraphcontext_uses_graphstorefactory(self):
        """Test that BaseGraphContext uses GraphStoreFactory.create()."""
        # Create a BaseGraphContext instance
        with mock.patch(
            "graph_context.factory.GraphStoreFactory.create"
        ) as mock_create:
            mock_create.return_value = MockGraphStore({"test": "config"})
            # Instantiate BaseGraphContext to test real integration and
            # delegation to the store and managers
            BaseGraphContext()

            # Verify that GraphStoreFactory.create was called
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_operations_delegate_to_store(self):
        """Test that entity operations delegate to the store."""
        # Create a context with our mock store
        context = BaseGraphContext()
        store = context._store

        # Verify store is a MockGraphStore instance
        assert isinstance(store, MockGraphStore)

        # Register entity type for testing
        await context.register_entity_type(
            EntityType(
                name="Person",
                properties={"name": PropertyDefinition(type="string", required=True)},
            )
        )

        # Begin transaction (required for operations)
        await context.begin_transaction()

        # Create entity and verify delegation
        entity_id = await context.create_entity("Person", {"name": "Alice"})
        assert ("create_entity", "Person", {"name": "Alice"}) in store.method_calls

        # Get entity and verify delegation
        await context.get_entity(entity_id)
        assert ("get_entity", entity_id) in store.method_calls

        # Update entity and verify delegation
        await context.update_entity(entity_id, {"name": "Alicia"})
        assert any(
            call[0] == "update_entity" and call[1] == entity_id
            for call in store.method_calls
        )

        # Delete entity and verify delegation
        await context.delete_entity(entity_id)
        assert ("delete_entity", entity_id) in store.method_calls

        # Commit transaction and verify delegation
        await context.commit_transaction()
        assert ("commit_transaction",) in store.method_calls

    @pytest.mark.asyncio
    async def test_relation_operations_delegate_to_store(self):
        """Test that relation operations delegate to the store."""
        # Create a context with our mock store
        context = BaseGraphContext()
        store = context._store

        # Register types for testing
        await context.register_entity_type(
            EntityType(
                name="Person",
                properties={"name": PropertyDefinition(type="string", required=True)},
            )
        )
        await context.register_entity_type(
            EntityType(
                name="Document",
                properties={"title": PropertyDefinition(type="string", required=True)},
            )
        )
        await context.register_relation_type(
            RelationType(
                name="Authored",
                from_types=["Person"],
                to_types=["Document"],
                properties={"year": PropertyDefinition(type="integer", required=False)},
            )
        )

        # Begin transaction
        await context.begin_transaction()

        # Create entities
        person_id = await context.create_entity("Person", {"name": "Bob"})
        doc_id = await context.create_entity("Document", {"title": "Report"})

        # Clear method calls for cleaner verification
        store.method_calls.clear()

        # Create relation and verify delegation
        relation_id = await context.create_relation("Authored", person_id, doc_id)
        assert any(
            call[0] == "create_relation"
            and call[1] == "Authored"
            and call[2] == person_id
            and call[3] == doc_id
            for call in store.method_calls
        )

        # Get relation and verify delegation
        await context.get_relation(relation_id)
        assert ("get_relation", relation_id) in store.method_calls

        # Update relation and verify delegation
        await context.update_relation(relation_id, {"year": 2023})
        assert any(
            call[0] == "update_relation" and call[1] == relation_id
            for call in store.method_calls
        )

        # Delete relation and verify delegation
        await context.delete_relation(relation_id)
        assert ("delete_relation", relation_id) in store.method_calls

        # Commit transaction
        await context.commit_transaction()

    @pytest.mark.asyncio
    async def test_query_and_traverse_delegate_to_store(self):
        """Test that query and traverse operations delegate to the store."""
        # Create a context with our mock store
        context = BaseGraphContext()
        store = context._store

        # Register entity type for testing
        await context.register_entity_type(
            EntityType(
                name="Person",
                properties={"name": PropertyDefinition(type="string", required=True)},
            )
        )

        # Begin transaction
        await context.begin_transaction()

        # Create entity
        entity_id = await context.create_entity("Person", {"name": "Charlie"})

        # Clear method calls for cleaner verification
        store.method_calls.clear()

        # Query and verify delegation
        query_spec = {"entity_type": "Person"}
        await context.query(query_spec)
        assert any(call[0] == "query" for call in store.method_calls)

        # Traverse and verify delegation
        traversal_spec = {"max_depth": 2, "direction": "outbound"}
        await context.traverse(entity_id, traversal_spec)
        assert any(
            call[0] == "traverse" and call[1] == entity_id
            for call in store.method_calls
        )

        # Commit transaction
        await context.commit_transaction()

    @pytest.mark.asyncio
    async def test_transaction_management_delegates_to_store(self):
        """Test that transaction management delegates to the store."""
        # Create a context with our mock store
        context = BaseGraphContext()
        store = context._store

        # Clear method calls
        store.method_calls.clear()

        # Begin transaction and verify delegation
        await context.begin_transaction()
        assert ("begin_transaction",) in store.method_calls

        # Commit transaction and verify delegation
        await context.commit_transaction()
        assert ("commit_transaction",) in store.method_calls

        # Begin another transaction
        await context.begin_transaction()

        # Rollback transaction and verify delegation
        await context.rollback_transaction()
        assert ("rollback_transaction",) in store.method_calls
