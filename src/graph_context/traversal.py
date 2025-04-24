"""
Graph traversal module providing different traversal strategies and utilities.

This module encapsulates the logic for traversing graphs, supporting different
traversal strategies, path tracking, and cycle detection.
"""
from typing import Dict, List, Any, Set, Tuple, Optional, TypeVar, Generic, Protocol, Iterator
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .types.type_base import Entity, Relation


class GraphLike(Protocol):
    """Protocol defining the minimal interface needed for graph traversal."""

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        ...

    def get_relations(self) -> Dict[str, Relation]:
        """Get all relations in the graph."""
        ...


@dataclass
class TraversalPath:
    """Represents a path through the graph during traversal."""
    entity: Entity
    path: List[Tuple[Relation, Entity]]
    depth: int


@dataclass
class TraversalSpec:
    """Specification for how to traverse the graph."""
    direction: str = "any"  # "outbound", "inbound", or "any"
    relation_types: Optional[List[str]] = None  # List of relation types to follow (None for any)
    max_depth: float = float("inf")  # Maximum traversal depth
    include_start: bool = False  # Whether to include start entity in results
    return_paths: bool = False  # Whether to return full paths instead of just entities
    max_paths_per_node: int = 100  # Maximum number of paths to store per node
    start_entity: Optional[str] = None  # ID of the start entity

    def __post_init__(self):
        """Validate the traversal specification."""
        if self.direction not in ["outbound", "inbound", "any"]:
            raise ValueError(f"Invalid direction: {self.direction}")
        if self.max_depth < 0:
            raise ValueError(f"Invalid max_depth: {self.max_depth}")
        if self.max_paths_per_node < 0:
            raise ValueError(f"Invalid max_paths_per_node: {self.max_paths_per_node}")


class TraversalStrategy(ABC):
    """Abstract base class for different traversal strategies."""

    def _find_connected_entities(
        self,
        graph: GraphLike,
        current_id: str,
        spec: TraversalSpec,
        current_path: List[Tuple[Relation, Entity]]
    ) -> Iterator[Tuple[str, Relation]]:
        """Find all connected entities based on traversal specification.

        Args:
            graph: The graph to traverse
            current_id: ID of the current entity
            spec: Traversal specification
            current_path: Current path being traversed

        Returns:
            Iterator of tuples containing (next_entity_id, relation)
        """
        for relation in graph.get_relations().values():
            # Check relation type first if specified
            if spec.relation_types and relation.type not in spec.relation_types:
                continue

            # Check direction and get next entity
            next_id = None
            if spec.direction in ("outbound", "any") and relation.from_entity == current_id:
                next_id = relation.to_entity
            elif spec.direction in ("inbound", "any") and relation.to_entity == current_id:
                next_id = relation.from_entity

            if next_id is not None:
                yield next_id, relation

    def _should_skip_node(
        self,
        next_id: str,
        visited: Set[str],
        path_counts: Dict[str, int],
        current_path: List[Tuple[Relation, Entity]],
        spec: TraversalSpec
    ) -> bool:
        """Determine if a node should be skipped during traversal.

        Args:
            next_id: ID of the next entity to visit
            visited: Set of visited entities (for non-path traversal)
            path_counts: Dictionary tracking number of paths per node
            current_path: Current path being traversed
            spec: Traversal specification

        Returns:
            True if the node should be skipped, False otherwise
        """
        # Skip if we've seen this entity before (only for non-path traversal)
        if not spec.return_paths and next_id in visited:
            return True

        # For path traversal, check if node is in current path (cycle detection)
        if spec.return_paths:
            in_current_path = any(step[1].id == next_id for step in current_path)
            if in_current_path:
                return True

        # Skip if we've exceeded max paths for this node
        if next_id in path_counts and path_counts[next_id] >= spec.max_paths_per_node:
            return True

        return False

    async def _add_to_results(
        self,
        results: List[Entity | TraversalPath],
        current_id: str,
        current_path: List[Tuple[Relation, Entity]],
        depth: int,
        visited: Set[str],
        path_counts: Dict[str, int],
        graph: GraphLike,
        spec: TraversalSpec
    ) -> None:
        """Add current entity to results if appropriate.

        Handles both path and non-path traversal logic based on spec.return_paths.

        This method needs to be async to await graph.get_entity().

        Args:
            results: List to store results
            current_id: ID of current entity
            current_path: Current path being traversed
            depth: Current depth in traversal
            visited: Set of visited entities
            path_counts: Dictionary tracking number of paths per node
            graph: The graph being traversed
            spec: Traversal specification
        """
        if (current_id != spec.start_entity or spec.include_start) and depth <= spec.max_depth:
            entity = await graph.get_entity(current_id)
            if spec.return_paths:
                # For path traversal, allow multiple paths to the same node
                path_counts[current_id] = path_counts.get(current_id, 0) + 1
                if path_counts[current_id] <= spec.max_paths_per_node:
                    results.append(TraversalPath(
                        entity=entity,
                        path=current_path,
                        depth=depth
                    ))
            elif current_id not in visited:
                # For non-path traversal, visit each node only once
                results.append(entity)
                visited.add(current_id)

    @abstractmethod
    async def traverse(
        self,
        graph: GraphLike,
        start_entity: str,
        spec: TraversalSpec
    ) -> List[Entity | TraversalPath]:
        """
        Traverse the graph according to the strategy.

        Args:
            graph: The graph to traverse
            start_entity: ID of the entity to start from
            spec: Traversal specification

        Returns:
            List of entities or paths found during traversal
        """
        pass


class BreadthFirstTraversal(TraversalStrategy):
    """Breadth-first traversal strategy."""

    async def traverse(
        self,
        graph: GraphLike,
        start_entity: str,
        spec: TraversalSpec
    ) -> List[Entity | TraversalPath]:
        """Traverse the graph using breadth-first search."""
        if not await graph.get_entity(start_entity):
            raise ValueError(f"Start entity not found: {start_entity}")

        # Store start_entity in spec for _add_to_results
        spec.start_entity = start_entity

        # Initialize traversal state
        visited = set()  # Track visited entities for non-path traversal
        path_counts = {}  # Track number of paths per node
        results = []

        # Queue format: (entity_id, path, depth)
        queue = [(start_entity, [], 0)]

        while queue:
            current_id, current_path, depth = queue.pop(0)

            # Add current entity to results if appropriate
            await self._add_to_results(
                results, current_id, current_path, depth,
                visited, path_counts, graph, spec
            )

            if depth < spec.max_depth:
                # Find connected entities
                for next_id, relation in self._find_connected_entities(
                    graph, current_id, spec, current_path
                ):
                    if self._should_skip_node(
                        next_id, visited, path_counts, current_path, spec
                    ):
                        continue

                    # Create new path by appending current step
                    next_entity = await graph.get_entity(next_id)
                    new_path = current_path + [(relation, next_entity)]
                    queue.append((next_id, new_path, depth + 1))

        return results


class DepthFirstTraversal(TraversalStrategy):
    """Depth-first traversal strategy."""

    async def traverse(
        self,
        graph: GraphLike,
        start_entity: str,
        spec: TraversalSpec
    ) -> List[Entity | TraversalPath]:
        """Traverse the graph using depth-first search."""
        if not await graph.get_entity(start_entity):
            raise ValueError(f"Start entity not found: {start_entity}")

        # Store start_entity in spec for _add_to_results
        spec.start_entity = start_entity

        # Initialize traversal state
        visited = set()  # Track visited entities for non-path traversal
        path_counts = {}  # Track number of paths per node
        results = []

        async def dfs(current_id: str, current_path: List[Tuple[Relation, Entity]], depth: int) -> None:
            # Add current entity to results if appropriate
            await self._add_to_results(
                results, current_id, current_path, depth,
                visited, path_counts, graph, spec
            )

            if depth < spec.max_depth:
                # Find connected entities
                for next_id, relation in self._find_connected_entities(
                    graph, current_id, spec, current_path
                ):
                    if self._should_skip_node(
                        next_id, visited, path_counts, current_path, spec
                    ):
                        continue

                    # Create new path by appending current step
                    next_entity = await graph.get_entity(next_id)
                    new_path = current_path + [(relation, next_entity)]
                    await dfs(next_id, new_path, depth + 1)

        # Start traversal from the start entity
        await dfs(start_entity, [], 0)
        return results


def create_traversal_spec(**kwargs) -> TraversalSpec:
    """Create a traversal specification with default values."""
    return TraversalSpec(**kwargs)


async def traverse(
    graph: GraphLike,
    start_entity: str,
    traversal_spec: Dict[str, Any],
    strategy: str = "bfs"
) -> List[Entity | TraversalPath]:
    """
    Traverse the graph according to the specified strategy.

    Args:
        graph: The graph to traverse
        start_entity: ID of the entity to start from
        traversal_spec: Dictionary of traversal parameters
        strategy: Traversal strategy to use ("bfs" or "dfs")

    Returns:
        List of entities or paths found during traversal

    Raises:
        ValueError: If start entity not found or invalid strategy
    """
    # Create traversal specification
    spec = create_traversal_spec(**traversal_spec)
    spec.start_entity = start_entity  # Set start entity in spec

    # Create traversal strategy
    if strategy == "bfs":
        traverser = BreadthFirstTraversal()
    elif strategy == "dfs":
        traverser = DepthFirstTraversal()
    else:
        raise ValueError(f"Unknown traversal strategy: {strategy}")

    # Perform traversal
    return await traverser.traverse(graph, start_entity, spec)