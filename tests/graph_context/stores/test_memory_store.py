"""Tests for the in-memory graph store implementation."""
from datetime import datetime

import pytest

from graph_context.exceptions import EntityNotFoundError, TransactionError
from graph_context.stores.memory_store import InMemoryGraphStore
from graph_context.traversal import TraversalPath


@pytest.fixture
def store() -> InMemoryGraphStore:
    """Create a fresh store instance for each test."""
    return InMemoryGraphStore({})


async def test_create_entity(store: InMemoryGraphStore):
    """Test creating an entity."""
    entity_id = await store.create_entity("person", {"name": "Alice"})
    assert entity_id is not None

    entity = await store.get_entity(entity_id)
    assert entity is not None
    assert entity.type == "person"
    assert entity.properties["name"] == "Alice"
    assert isinstance(entity.created_at, datetime)
    assert isinstance(entity.updated_at, datetime)


async def test_update_entity(store: InMemoryGraphStore):
    """Test updating an entity."""
    entity_id = await store.create_entity("person", {"name": "Alice"})
    original = await store.get_entity(entity_id)

    # Update properties
    success = await store.update_entity(entity_id, {"age": 30})
    assert success is True

    updated = await store.get_entity(entity_id)
    assert updated is not None
    assert updated.properties["name"] == "Alice"  # Original property preserved
    assert updated.properties["age"] == 30  # New property added
    assert updated.created_at == original.created_at
    assert updated.updated_at > original.updated_at


async def test_delete_entity(store: InMemoryGraphStore):
    """Test deleting an entity."""
    entity_id = await store.create_entity("person", {"name": "Alice"})

    # Delete should succeed
    success = await store.delete_entity(entity_id)
    assert success is True

    # Entity should no longer exist
    entity = await store.get_entity(entity_id)
    assert entity is None

    # Delete should fail for non-existent entity
    success = await store.delete_entity("nonexistent")
    assert success is False


async def test_create_relation(store: InMemoryGraphStore):
    """Test creating a relation between entities."""
    person1_id = await store.create_entity("person", {"name": "Alice"})
    person2_id = await store.create_entity("person", {"name": "Bob"})

    relation_id = await store.create_relation(
        "knows",
        person1_id,
        person2_id,
        {"since": "2023"}
    )
    assert relation_id is not None

    relation = await store.get_relation(relation_id)
    assert relation is not None
    assert relation.type == "knows"
    assert relation.from_entity == person1_id
    assert relation.to_entity == person2_id
    assert relation.properties["since"] == "2023"


async def test_create_relation_nonexistent_entity(store: InMemoryGraphStore):
    """Test creating a relation with non-existent entities fails."""
    person_id = await store.create_entity("person", {"name": "Alice"})

    with pytest.raises(EntityNotFoundError):
        await store.create_relation("knows", person_id, "nonexistent")

    with pytest.raises(EntityNotFoundError):
        await store.create_relation("knows", "nonexistent", person_id)


async def test_delete_entity_cascades_relations(store: InMemoryGraphStore):
    """Test that deleting an entity also deletes its relations."""
    person1_id = await store.create_entity("person", {"name": "Alice"})
    person2_id = await store.create_entity("person", {"name": "Bob"})

    relation_id = await store.create_relation("knows", person1_id, person2_id)

    # Delete one of the entities
    await store.delete_entity(person1_id)

    # Relation should be deleted
    relation = await store.get_relation(relation_id)
    assert relation is None


async def test_query_entities(store: InMemoryGraphStore):
    """Test querying entities."""
    # Create test data
    await store.create_entity("person", {"name": "Alice", "age": 30})
    await store.create_entity("person", {"name": "Bob", "age": 25})
    await store.create_entity("company", {"name": "Acme"})

    # Query by type
    results = await store.query({"entity_type": "person"})
    assert len(results) == 2
    assert all(e.type == "person" for e in results)

    # Query with property condition
    results = await store.query({
        "entity_type": "person",
        "conditions": [
            {"property": "age", "operator": "gt", "value": 25}
        ]
    })
    assert len(results) == 1
    assert results[0].properties["name"] == "Alice"


async def test_traverse_graph(store: InMemoryGraphStore):
    """Test graph traversal."""
    # Create a simple social network
    alice_id = await store.create_entity("person", {"name": "Alice"})
    bob_id = await store.create_entity("person", {"name": "Bob"})
    charlie_id = await store.create_entity("person", {"name": "Charlie"})

    await store.create_relation("knows", alice_id, bob_id)
    await store.create_relation("knows", bob_id, charlie_id)

    # Traverse outbound from Alice
    results = await store.traverse(alice_id, {
        "direction": "outbound",
        "max_depth": 2
    })

    assert len(results) == 2
    names = {e.properties["name"] for e in results}
    assert names == {"Bob", "Charlie"}


async def test_transaction_commit(store: InMemoryGraphStore):
    """Test transaction commit."""
    await store.begin_transaction()

    entity_id = await store.create_entity("person", {"name": "Alice"})

    # Entity should be visible within transaction
    entity = await store.get_entity(entity_id)
    assert entity is not None

    # Attempt to get non-existent entity within transaction
    non_existent_entity = await store.get_entity("nonexistent_tx_commit")
    assert non_existent_entity is None

    # Attempt to update non-existent relation within transaction
    update_success = await store.update_relation("nonexistent_rel_tx", {"prop": "val"})
    assert update_success is False

    await store.commit_transaction()

    # Entity should still exist after commit
    entity = await store.get_entity(entity_id)
    assert entity is not None


async def test_transaction_rollback(store: InMemoryGraphStore):
    """Test transaction rollback."""
    await store.begin_transaction()

    entity_id = await store.create_entity("person", {"name": "Alice"})

    # Entity should be visible within transaction
    entity = await store.get_entity(entity_id)
    assert entity is not None

    await store.rollback_transaction()

    # Entity should not exist after rollback
    entity = await store.get_entity(entity_id)
    assert entity is None


async def test_transaction_errors(store: InMemoryGraphStore):
    """Test transaction error conditions."""
    # Cannot begin transaction when one is already active
    await store.begin_transaction()
    with pytest.raises(TransactionError):
        await store.begin_transaction()
    await store.rollback_transaction()

    # Cannot commit when no transaction is active
    with pytest.raises(TransactionError):
        await store.commit_transaction()

    # Cannot rollback when no transaction is active
    with pytest.raises(TransactionError):
        await store.rollback_transaction()


async def test_update_relation_nonexistent(store: InMemoryGraphStore):
    """Test updating a non-existent relation."""
    success = await store.update_relation("nonexistent", {"key": "value"})
    assert success is False


async def test_update_relation(store: InMemoryGraphStore):
    """Test updating an existing relation with new properties."""
    # Setup: Create entities and relation
    entity1_id = await store.create_entity("person", {"name": "Alice"})
    entity2_id = await store.create_entity("person", {"name": "Bob"})

    relation_id = await store.create_relation(
        "knows",
        entity1_id,
        entity2_id,
        {"since": "2020", "type": "friend"}
    )

    # Get the original relation to compare later
    original = await store.get_relation(relation_id)
    assert original is not None
    assert original.properties["since"] == "2020"

    # Update relation with new properties
    success = await store.update_relation(relation_id, {
        "since": "2019",  # Override existing property
        "level": "close"  # Add new property
    })
    assert success is True

    # Verify the update
    updated = await store.get_relation(relation_id)
    assert updated is not None

    # Check that properties were correctly updated
    assert updated.properties["since"] == "2019"  # Updated value
    assert updated.properties["type"] == "friend"  # Original value preserved
    assert updated.properties["level"] == "close"  # New value added

    # Check that timestamps were handled correctly
    assert updated.created_at == original.created_at  # Creation time unchanged
    assert updated.updated_at > original.updated_at  # Update time changed

    # Test within transaction
    await store.begin_transaction()

    # Update relation within transaction
    success = await store.update_relation(relation_id, {"note": "important"})
    assert success is True

    # Verify update within transaction
    tx_updated = await store.get_relation(relation_id)
    assert tx_updated.properties["note"] == "important"
    assert tx_updated.properties["since"] == "2019"  # Value from previous update

    # Rollback and verify original state is restored
    await store.rollback_transaction()

    # After rollback, should not have the transaction changes
    final = await store.get_relation(relation_id)
    assert "note" not in final.properties
    assert final.properties["since"] == "2019"  # But should have the changes before transaction


async def test_delete_relation_nonexistent(store: InMemoryGraphStore):
    """Test deleting a non-existent relation."""
    success = await store.delete_relation("nonexistent")
    assert success is False


async def test_query_with_pagination(store: InMemoryGraphStore):
    """Test querying entities with offset and limit."""
    # Create test data
    for i in range(5):
        await store.create_entity("person", {"name": f"Person{i}", "age": 20 + i})

    # Test offset
    results = await store.query({
        "entity_type": "person",
        "offset": 2
    })
    assert len(results) == 3
    assert results[0].properties["name"] == "Person2"

    # Test limit
    results = await store.query({
        "entity_type": "person",
        "limit": 2
    })
    assert len(results) == 2
    assert results[0].properties["name"] == "Person0"

    # Test both offset and limit
    results = await store.query({
        "entity_type": "person",
        "offset": 1,
        "limit": 2
    })
    assert len(results) == 2
    assert results[0].properties["name"] == "Person1"
    assert results[1].properties["name"] == "Person2"


async def test_query_conditions_operators(store: InMemoryGraphStore):
    """Test query conditions with different operators."""
    await store.create_entity("test", {
        "str_val": "test_string",
        "num_val": 42,
        "list_val": ["a", "b", "c"]
    })

    # Test each operator
    operators = {
        "eq": ("str_val", "test_string", 1),
        "neq": ("str_val", "other", 1),
        "gt": ("num_val", 40, 1),
        "gte": ("num_val", 42, 1),
        "lt": ("num_val", 50, 1),
        "lte": ("num_val", 42, 1),
        "contains": ("list_val", "b", 1),
        "startswith": ("str_val", "test", 1),
        "endswith": ("str_val", "string", 1)
    }

    for op, (prop, val, expected_count) in operators.items():
        results = await store.query({
            "conditions": [
                {"property": prop, "operator": op, "value": val}
            ]
        })
        assert len(results) == expected_count, f"Operator {op} failed"


async def test_traverse_directions(store: InMemoryGraphStore):
    """Test graph traversal with different directions."""
    # Create a small graph
    a_id = await store.create_entity("node", {"name": "A"})
    b_id = await store.create_entity("node", {"name": "B"})
    c_id = await store.create_entity("node", {"name": "C"})

    await store.create_relation("connects", a_id, b_id, {"type": "forward"})
    await store.create_relation("links", b_id, c_id, {"type": "forward"})
    await store.create_relation("refers", c_id, a_id, {"type": "back"})

    # Test outbound traversal
    results = await store.traverse(a_id, {
        "direction": "outbound",
        "max_depth": 2
    })
    assert len(results) == 2
    names = {e.properties["name"] for e in results}
    assert names == {"B", "C"}

    # Test inbound traversal
    results = await store.traverse(a_id, {
        "direction": "inbound",
        "max_depth": 2
    })
    assert len(results) == 2
    names = {e.properties["name"] for e in results}
    assert names == {"B", "C"}

    # Test specific relation types
    results = await store.traverse(a_id, {
        "direction": "outbound",
        "relation_types": ["connects"],
        "max_depth": 2
    })
    assert len(results) == 1
    assert results[0].properties["name"] == "B"


async def test_query_conditions_edge_cases(store: InMemoryGraphStore):
    """Test query conditions with edge cases."""
    await store.create_entity("test", {
        "str_val": "test_string",
        "num_val": 42,
        "list_val": ["a", "b", "c"],
        "none_val": None
    })

    # Test property not in entity
    results = await store.query({
        "conditions": [
            {"property": "nonexistent", "operator": "eq", "value": "any"}
        ]
    })
    assert len(results) == 0

    # Test invalid operator
    results = await store.query({
        "conditions": [
            {"property": "str_val", "operator": "invalid", "value": "any"}
        ]
    })
    assert len(results) == 0


async def test_traverse_edge_cases(store: InMemoryGraphStore):
    """Test graph traversal edge cases."""
    # Test traversal with non-existent start entity
    with pytest.raises(EntityNotFoundError):
        await store.traverse("nonexistent", {
            "direction": "outbound",
            "max_depth": 1
        })

    # Create a cyclic graph
    a_id = await store.create_entity("node", {"name": "A"})
    b_id = await store.create_entity("node", {"name": "B"})
    c_id = await store.create_entity("node", {"name": "C"})

    await store.create_relation("connects", a_id, b_id)
    await store.create_relation("connects", b_id, c_id)
    await store.create_relation("connects", c_id, a_id)

    # Test cycle handling
    results = await store.traverse(a_id, {
        "direction": "outbound",
        "max_depth": 10  # Should not cause infinite loop
    })
    assert len(results) == 2  # Should only include B and C once
    names = {e.properties["name"] for e in results}
    assert names == {"B", "C"}

    # Test with empty relation_types
    results = await store.traverse(a_id, {
        "direction": "outbound",
        "relation_types": [],
        "max_depth": 1
    })
    assert len(results) == 1  # Should still traverse without type filtering


async def test_query_conditions_all_operators(store: InMemoryGraphStore):
    """Test all query condition operators."""
    # Create test data with various property types
    await store.create_entity("test", {
        "str_val": "test_string",
        "num_val": 42,
        "list_val": ["a", "b", "c"],
        "none_val": None,
        "bool_val": True
    })

    # Test each operator with appropriate values
    test_cases = [
        # Basic comparison operators
        ({"property": "num_val", "operator": "eq", "value": 42}, 1),
        ({"property": "num_val", "operator": "neq", "value": 41}, 1),
        ({"property": "num_val", "operator": "gt", "value": 41}, 1),
        ({"property": "num_val", "operator": "gt", "value": 42}, 0),
        ({"property": "num_val", "operator": "gte", "value": 42}, 1),
        ({"property": "num_val", "operator": "lt", "value": 43}, 1),
        ({"property": "num_val", "operator": "lt", "value": 42}, 0),
        ({"property": "num_val", "operator": "lte", "value": 42}, 1),

        # String operators
        ({"property": "str_val", "operator": "contains", "value": "string"}, 1),
        ({"property": "str_val", "operator": "startswith", "value": "test"}, 1),
        ({"property": "str_val", "operator": "endswith", "value": "string"}, 1),

        # Edge cases
        ({"property": "bool_val", "operator": "eq", "value": True}, 1),
        ({"property": "none_val", "operator": "eq", "value": None}, 1),
        ({"property": "list_val", "operator": "contains", "value": "b"}, 1),

        # Negative cases
        ({"property": "str_val", "operator": "contains", "value": "xyz"}, 0),
        ({"property": "str_val", "operator": "startswith", "value": "xyz"}, 0),
        ({"property": "str_val", "operator": "endswith", "value": "xyz"}, 0),
        ({"property": "num_val", "operator": "invalid", "value": 42}, 0),
    ]

    for condition, expected_count in test_cases:
        results = await store.query({
            "conditions": [condition]
        })
        assert len(results) == expected_count, f"Failed for operator {condition['operator']}"


async def test_traverse_all_directions(store: InMemoryGraphStore):
    """Test graph traversal with all possible directions."""
    # Create a diamond-shaped graph with cycles
    #     A
    #    / \
    #   B   C
    #    \ /
    #     D
    #     |
    #     A (cycle back to A)

    a_id = await store.create_entity("node", {"name": "A"})
    b_id = await store.create_entity("node", {"name": "B"})
    c_id = await store.create_entity("node", {"name": "C"})
    d_id = await store.create_entity("node", {"name": "D"})

    # Create relations with different types
    await store.create_relation("parent", a_id, b_id)
    await store.create_relation("friend", a_id, c_id)
    await store.create_relation("child", b_id, d_id)
    await store.create_relation("knows", c_id, d_id)
    await store.create_relation("cycle", d_id, a_id)

    # Test outbound traversal with specific relation type
    results = await store.traverse(a_id, {
        "direction": "outbound",
        "relation_types": ["parent"],
        "max_depth": 2
    })
    assert len(results) == 1 # Should only find B following the 'parent' relation
    names = {e.properties["name"] for e in results}
    assert names == {"B"}

    # Test inbound traversal with no relation type filter
    results = await store.traverse(d_id, {
        "direction": "inbound",
        "max_depth": 1
    })
    assert len(results) == 2  # Should find B and C
    names = {e.properties["name"] for e in results}
    assert names == {"B", "C"}

    # Test any direction with cycle detection
    results = await store.traverse(a_id, {
        "direction": "any",
        "max_depth": 10
    })
    # Should find all other nodes exactly once despite the cycle
    assert len(results) == 3
    names = {e.properties["name"] for e in results}
    assert names == {"B", "C", "D"}

    # Test with empty relation types list (should still traverse)
    results = await store.traverse(a_id, {
        "direction": "outbound",
        "relation_types": [],
        "max_depth": 1
    })
    assert len(results) == 2  # Should find B and C
    names = {e.properties["name"] for e in results}
    assert names == {"B", "C"}


async def test_traverse_with_path_tracking(store: InMemoryGraphStore):
    """Test graph traversal with path tracking enabled."""
    # Create a small graph with multiple paths to the same node
    #     A
    #    / \
    #   B   C
    #    \ / \
    #     D   E

    a_id = await store.create_entity("node", {"name": "A"})
    b_id = await store.create_entity("node", {"name": "B"})
    c_id = await store.create_entity("node", {"name": "C"})
    d_id = await store.create_entity("node", {"name": "D"})
    e_id = await store.create_entity("node", {"name": "E"})

    rel_ab = await store.create_relation("edge", a_id, b_id)
    rel_ac = await store.create_relation("edge", a_id, c_id)
    rel_bd = await store.create_relation("edge", b_id, d_id)
    rel_cd = await store.create_relation("edge", c_id, d_id)
    rel_ce = await store.create_relation("edge", c_id, e_id)

    # Traverse with return_paths=True
    results = await store.traverse(a_id, {
        "direction": "outbound",
        "return_paths": True,
        "max_depth": 2
    })

    assert len(results) == 5  # Should find B, C, D (via B), D (via C), E

    # Check result format (list of TraversalPath objects)
    assert all(isinstance(r, TraversalPath) for r in results)

    # Check paths to D
    paths_to_d = [r for r in results if r.entity.properties["name"] == "D"]
    assert len(paths_to_d) == 2
    assert paths_to_d[0].depth == 2
    assert paths_to_d[1].depth == 2

    # Verify path structure (example: one path to D)
    path_bd = paths_to_d[0] # Path A -> B -> D
    assert len(path_bd.path) == 2
    assert path_bd.path[0][0].id == rel_ab # Relation A->B
    assert path_bd.path[0][1].id == b_id   # Entity B
    assert path_bd.path[1][0].id == rel_bd # Relation B->D
    assert path_bd.path[1][1].id == d_id   # Entity D

    # Verify the other path to D
    path_cd = paths_to_d[1] # Path A -> C -> D
    assert len(path_cd.path) == 2
    assert path_cd.path[0][0].id == rel_ac # Relation A->C
    assert path_cd.path[0][1].id == c_id   # Entity C
    assert path_cd.path[1][0].id == rel_cd # Relation C->D
    assert path_cd.path[1][1].id == d_id   # Entity D


async def test_traverse_empty_relations(store: InMemoryGraphStore):
    """Test traversal behavior with no relations."""
    entity_id = await store.create_entity("test", {"name": "lonely"})

    results = await store.traverse(entity_id, {
        "direction": "any",
        "return_paths": True,
        "include_start": True  # Explicitly set to True
    })

    # Should only return start node if include_start is True
    assert len(results) == 1
    assert results[0].entity.id == entity_id
    assert len(results[0].path) == 0

    # Should return empty list if include_start is False
    results = await store.traverse(entity_id, {
        "direction": "any",
        "include_start": False
    })
    assert len(results) == 0


async def test_nested_transactions(store: InMemoryGraphStore):
    """Test that nested transactions are not allowed."""
    await store.begin_transaction()

    # Create an entity in the first transaction
    entity_id = await store.create_entity("test", {"name": "test"})

    # Attempt to start a nested transaction
    with pytest.raises(TransactionError, match="Transaction already in progress"):
        await store.begin_transaction()

    # Rollback the first transaction
    await store.rollback_transaction()

    # Verify entity was not created
    assert await store.get_entity(entity_id) is None


async def test_complex_query_combinations(store: InMemoryGraphStore):
    """Test complex combinations of query conditions."""
    # Create test entities
    await store.create_entity("person", {
        "name": "Alice",
        "age": 30,
        "tags": ["developer", "python"],
        "active": True
    })
    await store.create_entity("person", {
        "name": "Bob",
        "age": 25,
        "tags": ["developer", "java"],
        "active": True
    })
    await store.create_entity("person", {
        "name": "Charlie",
        "age": 35,
        "tags": ["manager"],
        "active": False
    })

    # Test multiple conditions with different operators
    results = await store.query({
        "entity_type": "person",
        "conditions": [
            {"property": "age", "operator": "gte", "value": 25},
            {"property": "tags", "operator": "contains", "value": "developer"},
            {"property": "active", "operator": "eq", "value": True}
        ]
    })

    assert len(results) == 2
    names = {e.properties["name"] for e in results}
    assert names == {"Alice", "Bob"}

    # Test with invalid operator
    results = await store.query({
        "conditions": [
            {"property": "name", "operator": "invalid_op", "value": "Alice"}
        ]
    })
    assert len(results) == 0


async def test_relation_error_cases(store: InMemoryGraphStore):
    """Test error cases for relation operations."""
    entity_id = await store.create_entity("test", {"name": "test"})

    # Test creating relation with non-existent 'to' entity
    with pytest.raises(EntityNotFoundError):
        await store.create_relation(
            "test",
            entity_id,
            "nonexistent",
            {"prop": "value"}
        )

    # Test creating relation with non-existent 'from' entity
    with pytest.raises(EntityNotFoundError):
        await store.create_relation(
            "test",
            "nonexistent",
            entity_id,
            {"prop": "value"}
        )

    # Create a valid relation
    relation_id = await store.create_relation(
        "test",
        entity_id,
        entity_id,  # Self-relation for testing
        {"prop": "value"}
    )

    # Test updating non-existent relation
    success = await store.update_relation("nonexistent", {"prop": "new_value"})
    assert not success

    # Delete the source entity and verify the relation is also deleted
    await store.delete_entity(entity_id)
    assert await store.get_relation(relation_id) is None


async def test_transaction_isolation(store: InMemoryGraphStore):
    """Test transaction isolation and visibility of changes."""
    # Create initial entity outside transaction
    entity_id = await store.create_entity("test", {"name": "original"})

    # Start transaction and modify entity
    await store.begin_transaction()
    await store.update_entity(entity_id, {"name": "modified"})

    # Verify changes are visible within transaction
    entity = await store.get_entity(entity_id)
    assert entity.properties["name"] == "modified"

    # Rollback transaction
    await store.rollback_transaction()

    # Verify original state is restored
    entity = await store.get_entity(entity_id)
    assert entity.properties["name"] == "original"

    # Test transaction commit
    await store.begin_transaction()
    await store.update_entity(entity_id, {"name": "committed"})
    await store.commit_transaction()

    # Verify changes persist after commit
    entity = await store.get_entity(entity_id)
    assert entity.properties["name"] == "committed"


async def test_get_relation(store: InMemoryGraphStore):
    """Test retrieving relations by ID."""
    p1 = await store.create_entity("node", {})
    p2 = await store.create_entity("node", {})
    rel_id = await store.create_relation("links", p1, p2)

    # Get existing relation
    relation = await store.get_relation(rel_id)
    assert relation is not None
    assert relation.id == rel_id
    assert relation.type == "links"

    # Get non-existent relation
    relation = await store.get_relation("nonexistent")
    assert relation is None

    # Test within transaction
    await store.begin_transaction()
    rel_id_tx = await store.create_relation("connects", p1, p2)

    # Get existing relation within transaction
    relation_tx = await store.get_relation(rel_id_tx)
    assert relation_tx is not None
    assert relation_tx.id == rel_id_tx

    # Get relation created outside transaction
    relation_orig_tx = await store.get_relation(rel_id)
    assert relation_orig_tx is not None
    assert relation_orig_tx.id == rel_id

    # Get non-existent relation within transaction
    relation_none_tx = await store.get_relation("nonexistent_tx")
    assert relation_none_tx is None

    await store.rollback_transaction()


async def test_coverage_edge_cases(store: InMemoryGraphStore):
    """Test edge cases to maximize coverage."""
    # Attempt to get non-existent entity (to hit line 94)
    non_existent_id = "non_existent_id"
    entity = await store.get_entity(non_existent_id)
    assert entity is None

    # Create and delete an entity to ensure it doesn't exist
    entity_id = await store.create_entity("test", {"name": "temp"})
    assert await store.delete_entity(entity_id) is True

    # Verify entity is gone
    assert await store.get_entity(entity_id) is None

    # Create entities and relation for update test
    e1 = await store.create_entity("test", {})
    e2 = await store.create_entity("test", {})
    rel_id = await store.create_relation("test_rel", e1, e2)

    # Attempt to update a non-existent relation (to hit lines 199-200)
    deleted_rel_id = "deleted_relation_id"
    update_result = await store.update_relation(deleted_rel_id, {"key": "value"})
    assert update_result is False

    # Within transaction for additional coverage
    await store.begin_transaction()
    tx_result = await store.update_relation(deleted_rel_id, {"key": "value"})
    assert tx_result is False
    await store.rollback_transaction()


async def test_delete_relation_nonexistent(store: InMemoryGraphStore):
    """Test deleting a non-existent relation."""
    success = await store.delete_relation("nonexistent")
    assert success is False