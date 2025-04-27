"""
Query management for the graph-context module.

This module provides the QueryManager class for managing query and traversal
operations in the graph context.
"""

from ..event_system import EventSystem, GraphEvent
from ..interfaces.store import GraphStore
from ..types.type_base import Entity, QuerySpec, TraversalSpec


class QueryManager:
    """
    Manages query and traversal operations.

    This class encapsulates query-related operations to provide a consistent
    interface for searching and traversing the graph.
    """

    def __init__(self, store: GraphStore, events: EventSystem) -> None:
        """
        Initialize the query manager.

        Args:
            store: The graph store to perform queries on
            events: Event system for emitting query events
        """
        self._store = store
        self._events = events

    async def query(self, query_spec: QuerySpec) -> list[Entity]:
        """
        Execute a query against the graph.

        Args:
            query_spec: Specification of the query to execute

        Returns:
            List of entities matching the query
        """
        results = await self._store.query(query_spec)

        await self._events.emit(GraphEvent.QUERY_EXECUTED, query_spec=query_spec)

        return results

    async def traverse(self, start_entity: str, traversal_spec: TraversalSpec) -> list[Entity]:
        """
        Traverse the graph starting from a given entity.

        Args:
            start_entity: ID of the entity to start traversal from
            traversal_spec: Specification of the traversal

        Returns:
            List of entities found during traversal
        """
        results = await self._store.traverse(start_entity, traversal_spec)

        await self._events.emit(
            GraphEvent.TRAVERSAL_EXECUTED,
            start_entity=start_entity,
            traversal_spec=traversal_spec,
        )

        return results
