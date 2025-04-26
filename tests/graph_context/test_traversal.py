"""Tests for the graph traversal module."""

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import pytest

from graph_context.traversal import (
    BreadthFirstTraversal,
    GraphLike,
    TraversalSpec,
    create_traversal_spec,
    traverse,
)
from graph_context.types.type_base import Entity, Relation


class MockGraph(GraphLike):
    """Mock graph implementation for testing."""

    def __init__(self):
        """Initialize the mock graph."""
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_relations(self) -> Dict[str, Relation]:
        """Get all relations in the graph."""
        return self.relations

    def add_entity(self, id: str, type: str, properties: Dict[str, Any] | None = None) -> None:
        """Helper to add an entity to the mock graph."""
        self.entities[id] = Entity(
            id=id,
            type=type,
            properties=properties or {},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    def add_relation(
        self,
        id: str,
        type: str,
        from_entity: str,
        to_entity: str,
        properties: Dict[str, Any] | None = None,
    ) -> None:
        """Helper to add a relation to the mock graph."""
        self.relations[id] = Relation(
            id=id,
            type=type,
            from_entity=from_entity,
            to_entity=to_entity,
            properties=properties or {},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )


class DiamondGraph(GraphLike):
    r"""A diamond-shaped graph for testing traversal order.

       A
      / \
     B   C
      \ /
       D
    """

    def __init__(self, entities: List[Entity], relations: List[Relation]):
        """Initialize the diamond graph."""
        self.entities = {e.id: e for e in entities}
        self.relations = {r.id: r for r in relations}

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_relations(self) -> Dict[str, Relation]:
        """Get all relations in the graph."""
        return self.relations


@pytest.fixture
def simple_graph():
    """Create a simple graph for testing:
    A --(parent)--> B --(friend)--> C
    |
    +-(parent)--> D --(friend)--> E
    """
    graph = MockGraph()

    # Add entities
    for id, type in [
        ("A", "person"),
        ("B", "person"),
        ("C", "person"),
        ("D", "person"),
        ("E", "person"),
    ]:
        graph.add_entity(id, type)

    # Add relations
    graph.add_relation("r1", "parent", "A", "B")
    graph.add_relation("r2", "friend", "B", "C")
    graph.add_relation("r3", "parent", "A", "D")
    graph.add_relation("r4", "friend", "D", "E")

    return graph


@pytest.fixture
def cyclic_graph():
    """Create a cyclic graph for testing:
    A --(friend)--> B --(friend)--> C
    ^                               |
    |                              |
    +-------(friend)---------------+
    """
    graph = MockGraph()

    # Add entities
    for id in ["A", "B", "C"]:
        graph.add_entity(id, "person")

    # Add relations forming a cycle
    graph.add_relation("r1", "friend", "A", "B")
    graph.add_relation("r2", "friend", "B", "C")
    graph.add_relation("r3", "friend", "C", "A")

    return graph


@pytest.fixture
def empty_graph():
    """Create an empty graph for testing."""

    class EmptyGraph(GraphLike):
        async def get_entity(self, entity_id: str) -> Optional[Entity]:
            return None

        def get_relations(self) -> Dict[str, Relation]:
            return {}

    return EmptyGraph()


def test_create_traversal_spec():
    """Test creation of traversal specification."""
    # Test default values
    spec = create_traversal_spec()
    assert spec.direction == "any"
    assert spec.relation_types is None
    assert spec.max_depth == float("inf")
    assert not spec.include_start
    assert not spec.return_paths

    # Test custom values
    spec = create_traversal_spec(
        direction="outbound",
        relation_types=["friend"],
        max_depth=2,
        include_start=True,
        return_paths=True,
    )
    assert spec.direction == "outbound"
    assert spec.relation_types == ["friend"]
    assert spec.max_depth == 2
    assert spec.include_start
    assert spec.return_paths


@pytest.mark.asyncio
async def test_simple_traversal(simple_graph):
    """Test basic traversal functionality."""
    # Test outbound traversal
    results = await traverse(simple_graph, "A", {"direction": "outbound"})
    assert len(results) == 4
    assert all(isinstance(r, Entity) for r in results)
    assert {r.id for r in results} == {"B", "C", "D", "E"}

    # Test inbound traversal (should find nothing from A)
    results = await traverse(simple_graph, "A", {"direction": "inbound"})
    assert len(results) == 0

    # Test with specific relation type
    results = await traverse(simple_graph, "A", {"direction": "outbound", "relation_types": ["parent"]})
    assert len(results) == 2
    assert {r.id for r in results} == {"B", "D"}


@pytest.mark.asyncio
async def test_path_tracking(simple_graph):
    """Test that path information is correctly recorded."""
    # Create a simple chain: A -> B -> C
    spec = {
        "direction": "outbound",  # Add direction to get specific paths
        "return_paths": True,
        "include_start": True,
    }
    paths = await traverse(simple_graph, "A", spec)

    # Verify we get all valid paths
    assert len([p for p in paths if p.entity.id == "A"]) == 1  # Start node
    assert len([p for p in paths if p.entity.id == "B"]) >= 1  # At least one path to B
    assert len([p for p in paths if p.entity.id == "C"]) >= 1  # At least one path to C

    # Check path depths are correct
    for path in paths:
        assert path.depth == len(path.path)  # Depth should match path length


@pytest.mark.asyncio
async def test_max_depth(simple_graph):
    """Test depth limiting in traversal."""
    # Depth 1 should only find direct connections
    results = await traverse(simple_graph, "A", {"direction": "outbound", "max_depth": 1})
    assert len(results) == 2
    assert {r.id for r in results} == {"B", "D"}

    # Depth 0 should find nothing (since include_start is False by default)
    results = await traverse(simple_graph, "A", {"max_depth": 0})
    assert len(results) == 0

    # Depth 0 with include_start should only find start node
    results = await traverse(simple_graph, "A", {"max_depth": 0, "include_start": True})
    assert len(results) == 1
    assert results[0].id == "A"


@pytest.mark.asyncio
async def test_cycle_handling(cyclic_graph):
    """Test handling of cycles in the graph."""
    # Without path tracking, each node should appear once
    results = await traverse(cyclic_graph, "A", {"direction": "outbound"})
    assert len(results) == 2  # B and C (A is excluded by default)
    assert {r.id for r in results} == {"B", "C"}

    # With path tracking, we should respect cycle prevention
    results = await traverse(
        cyclic_graph,
        "A",
        {
            "direction": "outbound",
            "return_paths": True,
            "max_depth": 3,  # Limit depth to prevent infinite paths
        },
    )

    # Each path should be unique and not contain cycles
    for path in results:
        node_ids = {step[1].id for step in path.path}
        assert len(node_ids) == len(path.path)  # No repeated nodes in path


@pytest.mark.asyncio
async def test_invalid_start_entity(simple_graph):
    """Test traversal with invalid start entity."""
    with pytest.raises(ValueError):
        await traverse(simple_graph, "nonexistent", {})


@pytest.mark.asyncio
async def test_direction_filtering(simple_graph):
    """Test filtering by direction."""
    # From B, outbound should find immediate neighbors only
    results = await traverse(
        simple_graph,
        "B",
        {
            "direction": "outbound",
            "max_depth": 1,  # Limit to immediate neighbors
        },
    )
    assert len(results) == 1
    assert results[0].id == "C"

    # From B, inbound should find immediate neighbors only
    results = await traverse(
        simple_graph,
        "B",
        {
            "direction": "inbound",
            "max_depth": 1,  # Limit to immediate neighbors
        },
    )
    assert len(results) == 1
    assert results[0].id == "A"

    # From B, any direction should find immediate neighbors only
    results = await traverse(
        simple_graph,
        "B",
        {
            "direction": "any",
            "max_depth": 1,  # Limit to immediate neighbors
        },
    )
    assert len(results) == 2
    assert {r.id for r in results} == {"A", "C"}


@pytest.mark.asyncio
async def test_relation_type_filtering(simple_graph):
    """Test filtering by relation type."""
    # From A, following only friend relations
    # Should find no direct friends
    results = await traverse(simple_graph, "A", {"direction": "outbound", "relation_types": ["friend"]})
    assert len(results) == 0

    # From A, following only parent relations
    # Should find direct children B and D
    results = await traverse(simple_graph, "A", {"direction": "outbound", "relation_types": ["parent"]})
    assert len(results) == 2
    assert {r.id for r in results} == {"B", "D"}

    # Test with return_paths to ensure only friend relations are followed
    results = await traverse(
        simple_graph,
        "B",
        {"direction": "outbound", "relation_types": ["friend"], "return_paths": True},
    )
    assert len(results) == 1
    assert results[0].entity.id == "C"
    assert results[0].path[0][0].type == "friend"


@pytest.mark.asyncio
async def test_traversal_strategies(simple_graph):
    """Test different traversal strategies (BFS vs DFS)."""
    # Test BFS strategy
    bfs_results = await traverse(
        simple_graph,
        "A",
        {"direction": "outbound", "return_paths": True},
        strategy="bfs",
    )

    # Test DFS strategy
    dfs_results = await traverse(
        simple_graph,
        "A",
        {"direction": "outbound", "return_paths": True},
        strategy="dfs",
    )

    # Both strategies should find the same nodes
    bfs_nodes = {r.entity.id for r in bfs_results}
    dfs_nodes = {r.entity.id for r in dfs_results}
    assert bfs_nodes == dfs_nodes

    # But they might find them in different orders
    bfs_order = [r.entity.id for r in bfs_results]
    dfs_order = [r.entity.id for r in dfs_results]
    assert len(bfs_order) == len(dfs_order)

    # Test invalid strategy
    with pytest.raises(ValueError):
        await traverse(simple_graph, "A", {}, strategy="invalid")


@pytest.mark.asyncio
async def test_max_paths_per_node(cyclic_graph):
    """Test the max_paths_per_node limit in traversal."""
    # Set a small max_paths_per_node limit
    results = await traverse(
        cyclic_graph,
        "A",
        {"direction": "outbound", "return_paths": True, "max_paths_per_node": 2},
    )

    # Count paths per node
    path_counts = {}
    for result in results:
        path_counts[result.entity.id] = path_counts.get(result.entity.id, 0) + 1

    # Verify no node has more paths than the limit
    assert all(count <= 2 for count in path_counts.values())


@pytest.mark.asyncio
async def test_mixed_directional_relations():
    """Test behavior with mixed bidirectional and unidirectional relations."""
    graph = MockGraph()
    # Create a diamond pattern with mixed directions
    for i in range(1, 5):
        graph.add_entity(str(i), "test")

    # Bidirectional relation between 1-2
    graph.add_relation("r1", "both", "1", "2")
    graph.add_relation("r2", "both", "2", "1")

    # Unidirectional relations completing the diamond
    graph.add_relation("r3", "one_way", "2", "3")
    graph.add_relation("r4", "one_way", "2", "4")
    graph.add_relation("r5", "one_way", "3", "4")

    # Test outbound traversal
    spec = create_traversal_spec(direction="outbound", return_paths=True)
    result = await traverse(graph, "1", asdict(spec))
    assert len(result) > 0
    # Verify we can reach node 4 through both paths
    paths_to_4 = [p for p in result if p.entity.id == "4"]
    assert len(paths_to_4) == 2

    # Test inbound traversal
    spec = create_traversal_spec(direction="inbound", return_paths=True)
    result = await traverse(graph, "4", asdict(spec))
    assert len(result) > 0
    # Verify we can reach node 1 through the bidirectional relation
    paths_to_1 = [p for p in result if p.entity.id == "1"]
    assert len(paths_to_1) > 0


@pytest.mark.asyncio
async def test_many_paths_per_node():
    """Test behavior with many paths to the same node."""
    graph = MockGraph()
    # Create a complete graph with 5 nodes
    for i in range(1, 6):
        graph.add_entity(str(i), "test")
        for j in range(1, 6):
            if i != j:
                graph.add_relation(f"r{i}{j}", "test", str(i), str(j))

    # Test with default max_paths_per_node
    spec = create_traversal_spec(return_paths=True)
    result = await traverse(graph, "1", asdict(spec))
    # Should be limited by max_paths_per_node
    node_path_counts = {}
    for path in result:
        node_id = path.entity.id
        node_path_counts[node_id] = node_path_counts.get(node_id, 0) + 1
        assert node_path_counts[node_id] <= 100  # Default max_paths_per_node

    # Test with custom max_paths_per_node
    spec = create_traversal_spec(return_paths=True, max_paths_per_node=5)
    result = await traverse(graph, "1", asdict(spec))
    node_path_counts = {}
    for path in result:
        node_id = path.entity.id
        node_path_counts[node_id] = node_path_counts.get(node_id, 0) + 1
        assert node_path_counts[node_id] <= 5


@pytest.mark.asyncio
async def test_empty_graph(empty_graph):
    """Test traversal behavior with an empty graph."""
    spec = {
        "direction": "any",
        "relation_types": None,
        "max_depth": float("inf"),
        "include_start": False,
        "return_paths": False,
        "max_paths_per_node": 100,
    }
    with pytest.raises(ValueError, match="Start entity not found"):
        await traverse(empty_graph, "non_existent", spec)


@pytest.mark.asyncio
async def test_self_referential_relation(cyclic_graph):
    """Test traversal correctly handles self-referential relations (cycle detection)."""
    # Add a self-referential relation
    cyclic_graph.add_relation("r_self", "friend", "C", "C")

    results = await traverse(
        cyclic_graph,
        "A",
        {
            "direction": "outbound",
            "return_paths": True,
            "max_depth": 5,  # Allow deeper traversal
        },
    )

    # Verify traversal completes and finds paths
    assert len(results) > 0

    # Verify that no returned path includes the self-referential step C->C
    found_self_loop = False
    for path_result in results:
        for relation, _ in path_result.path:
            if relation.id == "r_self" or (relation.from_entity == "C" and relation.to_entity == "C"):
                found_self_loop = True
                break
        if found_self_loop:
            break
    assert (
        not found_self_loop
    ), f"Self-referential relation C->C incorrectly included in path: {path_result.path if found_self_loop else ''}"

    # Verify expected paths (excluding cycles) are present
    paths_found = {"->".join(["A"] + [step[1].id for step in p.path]) for p in results}
    assert "A->B" in paths_found
    assert "A->B->C" in paths_found
    # Path A->B->C->A should be prevented by cycle detection
    assert "A->B->C->A" not in paths_found


def test_should_skip_node():
    """Test the _should_skip_node method of TraversalStrategy."""
    strategy = BreadthFirstTraversal()
    visited = set()
    path_counts = {}
    current_path = []

    # Create test entities and relations

    entity2 = Entity(
        id="2",
        type="test",
        properties={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    relation = Relation(
        id="r1",
        type="test",
        from_entity="1",
        to_entity="2",
        properties={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    # Test 1: Non-path traversal with visited node
    spec = TraversalSpec(return_paths=False)
    visited.add("2")
    assert strategy._should_skip_node("2", visited, path_counts, current_path, spec)
    visited.remove("2")
    assert not strategy._should_skip_node("2", visited, path_counts, current_path, spec)

    # Test 2: Path traversal with node in current path (cycle detection)
    spec = TraversalSpec(return_paths=True)
    current_path = [(relation, entity2)]  # Node "2" is in current path
    assert strategy._should_skip_node("2", visited, path_counts, current_path, spec)
    current_path = []  # Empty path
    assert not strategy._should_skip_node("2", visited, path_counts, current_path, spec)

    # Test 3: Max paths per node exceeded
    spec = TraversalSpec(return_paths=True, max_paths_per_node=2)
    path_counts["2"] = 2  # Node has reached max paths
    assert strategy._should_skip_node("2", visited, path_counts, current_path, spec)
    path_counts["2"] = 1  # Node has not reached max paths
    assert not strategy._should_skip_node("2", visited, path_counts, current_path, spec)

    # Test 4: Multiple conditions together
    spec = TraversalSpec(return_paths=True, max_paths_per_node=2)
    visited.add("2")
    path_counts["2"] = 2
    current_path = [(relation, entity2)]
    assert strategy._should_skip_node("2", visited, path_counts, current_path, spec)


@pytest.mark.asyncio
async def test_dfs_multiple_skip_conditions():
    """Test DFS traversal with multiple skip conditions."""
    # Create a graph with a structure that would trigger multiple skip conditions
    graph = MockGraph()

    # Create a graph with multiple paths to the same nodes
    # A -> B -> C
    # |    ^    ^
    # |    |    |
    # + -> D -> E

    # Add entities
    for id in ["A", "B", "C", "D", "E"]:
        graph.add_entity(id, "test")

    # Add relations to create multiple paths
    graph.add_relation("r1", "test", "A", "B")  # A->B
    graph.add_relation("r2", "test", "B", "C")  # B->C
    graph.add_relation("r3", "test", "A", "D")  # A->D
    graph.add_relation("r4", "test", "D", "B")  # D->B
    graph.add_relation("r5", "test", "D", "E")  # D->E
    graph.add_relation("r6", "test", "E", "C")  # E->C

    # Test DFS traversal with conditions that would trigger multiple skip checks:
    # - return_paths=True (for cycle detection)
    # - max_paths_per_node=1 (to limit paths)
    spec = {
        "direction": "outbound",
        "return_paths": True,
        "max_paths_per_node": 1,  # This will force path skipping
    }

    results = await traverse(graph, "A", spec, strategy="dfs")

    # Verify results
    # 1. Each node should appear at most once (due to max_paths_per_node=1)
    node_counts = {}
    for result in results:
        node_id = result.entity.id
        node_counts[node_id] = node_counts.get(node_id, 0) + 1
        assert node_counts[node_id] == 1, f"Node {node_id} appears more than once"

    # 2. No cycles should be present in any path
    for result in results:
        path_nodes = {step[1].id for step in result.path}
        assert len(path_nodes) == len(result.path), f"Cycle detected in path to {result.entity.id}"

    # 3. All paths should be valid
    for result in results:
        if result.path:  # Skip start node if included
            # Each step in path should be connected to previous
            for i in range(len(result.path)):
                if i > 0:
                    prev_node = result.path[i - 1][1].id
                    curr_relation = result.path[i][0]
                    assert curr_relation.from_entity == prev_node, f"Invalid path: {result.path}"
