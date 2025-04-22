"""Caching module for graph-context.

This module provides caching functionality for the graph-context library,
including a cached implementation of the graph context.
"""

from typing import Optional, Dict, Any, List

from ..event_system import GraphEvent
from ..types.type_base import Entity, Relation
from ..context_base import BaseGraphContext
from .cache_store import CacheStore, CacheEntry
from .cache_manager import CacheManager

class CachedGraphContext(BaseGraphContext):
    """A graph context implementation with caching support."""

    def __init__(
        self,
        base_context: BaseGraphContext,
        max_size: int = 10000,
        ttl: Optional[int] = None
    ):
        """Initialize the cached graph context.

        Args:
            base_context: The base graph context to wrap
            max_size: Maximum number of entries in the cache
            ttl: Time-to-live in seconds for cache entries
        """
        super().__init__()
        self._base = base_context
        self._cache_manager = CacheManager(max_size=max_size, ttl=ttl)

        # Subscribe to all relevant events
        self._base.subscribe(GraphEvent.ENTITY_READ, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.ENTITY_WRITE, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.ENTITY_BULK_WRITE, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.ENTITY_DELETE, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.ENTITY_BULK_DELETE, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.RELATION_READ, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.RELATION_WRITE, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.RELATION_BULK_WRITE, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.RELATION_DELETE, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.RELATION_BULK_DELETE, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.QUERY_EXECUTED, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.TRAVERSAL_EXECUTED, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.SCHEMA_MODIFIED, self._cache_manager.handle_event)
        self._base.subscribe(GraphEvent.TYPE_MODIFIED, self._cache_manager.handle_event)

    def enable_caching(self) -> None:
        """Enable caching."""
        self._cache_manager.enable()

    def disable_caching(self) -> None:
        """Disable caching."""
        self._cache_manager.disable()

    async def get_entity(self, entity_type: str, entity_id: str) -> Optional[Entity]:
        """Get an entity by type and ID.

        Args:
            entity_type: The type of entity to get
            entity_id: The ID of the entity to get

        Returns:
            The entity if found, None otherwise
        """
        # Try to get from cache first
        key = self._cache_manager._make_key("entity", type=entity_type, id=entity_id)
        entry = await self._cache_manager.store.get(key)
        if entry:
            return entry.value

        # Fall back to base context
        return await self._base.get_entity(entity_type, entity_id)

    async def get_relation(self, relation_type: str, relation_id: str) -> Optional[Relation]:
        """Get a relation by type and ID.

        Args:
            relation_type: The type of relation to get
            relation_id: The ID of the relation to get

        Returns:
            The relation if found, None otherwise
        """
        # Try to get from cache first
        key = self._cache_manager._make_key("relation", type=relation_type, id=relation_id)
        entry = await self._cache_manager.store.get(key)
        if entry:
            return entry.value

        # Fall back to base context
        return await self._base.get_relation(relation_type, relation_id)

    async def query(self, query_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a query against the graph.

        Args:
            query_spec: The query specification

        Returns:
            The query results
        """
        # Try to get from cache first
        query_hash = self._cache_manager._hash_query(query_spec)
        key = self._cache_manager._make_key("query", hash=query_hash)
        entry = await self._cache_manager.store.get(key)
        if entry:
            return entry.value

        # Fall back to base context
        return await self._base.query(query_spec)

    async def traverse(self, traversal_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a traversal in the graph.

        Args:
            traversal_spec: The traversal specification

        Returns:
            The traversal results
        """
        # Try to get from cache first
        traversal_hash = self._cache_manager._hash_query(traversal_spec)
        key = self._cache_manager._make_key("traversal", hash=traversal_hash)
        entry = await self._cache_manager.store.get(key)
        if entry:
            return entry.value

        # Fall back to base context
        return await self._base.traverse(traversal_spec)

    # Delegate write operations directly to base context
    async def create_entity(self, entity_type: str, entity: Entity) -> str:
        """Create a new entity.

        Args:
            entity_type: The type of entity to create
            entity: The entity data

        Returns:
            The ID of the created entity
        """
        return await self._base.create_entity(entity_type, entity)

    async def update_entity(self, entity_type: str, entity_id: str, entity: Entity) -> None:
        """Update an existing entity.

        Args:
            entity_type: The type of entity to update
            entity_id: The ID of the entity to update
            entity: The updated entity data
        """
        await self._base.update_entity(entity_type, entity_id, entity)

    async def delete_entity(self, entity_type: str, entity_id: str) -> None:
        """Delete an entity.

        Args:
            entity_type: The type of entity to delete
            entity_id: The ID of the entity to delete
        """
        await self._base.delete_entity(entity_type, entity_id)

    async def create_relation(self, relation_type: str, relation: Relation) -> str:
        """Create a new relation.

        Args:
            relation_type: The type of relation to create
            relation: The relation data

        Returns:
            The ID of the created relation
        """
        return await self._base.create_relation(relation_type, relation)

    async def update_relation(self, relation_type: str, relation_id: str, relation: Relation) -> None:
        """Update an existing relation.

        Args:
            relation_type: The type of relation to update
            relation_id: The ID of the relation to update
            relation: The updated relation data
        """
        await self._base.update_relation(relation_type, relation_id, relation)

    async def delete_relation(self, relation_type: str, relation_id: str) -> None:
        """Delete a relation.

        Args:
            relation_type: The type of relation to delete
            relation_id: The ID of the relation to delete
        """
        await self._base.delete_relation(relation_type, relation_id)

    # Delegate bulk operations directly to base context
    async def bulk_create_entities(self, entity_type: str, entities: List[Entity]) -> List[str]:
        """Create multiple entities in bulk.

        Args:
            entity_type: The type of entities to create
            entities: The list of entities to create

        Returns:
            The list of created entity IDs
        """
        return await self._base.bulk_create_entities(entity_type, entities)

    async def bulk_update_entities(self, entity_type: str, entities: List[Dict[str, Any]]) -> None:
        """Update multiple entities in bulk.

        Args:
            entity_type: The type of entities to update
            entities: The list of entities to update, each containing an 'id' field
        """
        await self._base.bulk_update_entities(entity_type, entities)

    async def bulk_delete_entities(self, entity_type: str, entity_ids: List[str]) -> None:
        """Delete multiple entities in bulk.

        Args:
            entity_type: The type of entities to delete
            entity_ids: The list of entity IDs to delete
        """
        await self._base.bulk_delete_entities(entity_type, entity_ids)

    async def bulk_create_relations(self, relation_type: str, relations: List[Relation]) -> List[str]:
        """Create multiple relations in bulk.

        Args:
            relation_type: The type of relations to create
            relations: The list of relations to create

        Returns:
            The list of created relation IDs
        """
        return await self._base.bulk_create_relations(relation_type, relations)

    async def bulk_update_relations(self, relation_type: str, relations: List[Dict[str, Any]]) -> None:
        """Update multiple relations in bulk.

        Args:
            relation_type: The type of relations to update
            relations: The list of relations to update, each containing an 'id' field
        """
        await self._base.bulk_update_relations(relation_type, relations)

    async def bulk_delete_relations(self, relation_type: str, relation_ids: List[str]) -> None:
        """Delete multiple relations in bulk.

        Args:
            relation_type: The type of relations to delete
            relation_ids: The list of relation IDs to delete
        """
        await self._base.bulk_delete_relations(relation_type, relation_ids)

# Export public API
__all__ = ['CachedGraphContext', 'CacheStore', 'CacheEntry', 'CacheManager']