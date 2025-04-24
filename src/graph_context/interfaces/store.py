"""
Graph store interface definition.

This module defines the interface for graph data storage operations that all
concrete store implementations must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from ..types.type_base import Entity, Relation


class GraphStore(ABC):
    """
    Abstract interface for graph data storage operations.
    Concrete implementations handle the actual persistence of entities and relations.
    """

    @abstractmethod
    async def create_entity(
        self,
        entity_type: str,
        properties: Dict[str, Any]
    ) -> str:
        """Create a new entity in the graph."""
        pass

    @abstractmethod
    async def get_entity(
        self,
        entity_id: str
    ) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        pass

    @abstractmethod
    async def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Update an existing entity."""
        pass

    @abstractmethod
    async def delete_entity(
        self,
        entity_id: str
    ) -> bool:
        """Delete an entity from the graph."""
        pass

    @abstractmethod
    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new relation between entities."""
        pass

    @abstractmethod
    async def get_relation(
        self,
        relation_id: str
    ) -> Optional[Relation]:
        """Retrieve a relation by ID."""
        pass

    @abstractmethod
    async def update_relation(
        self,
        relation_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """Update an existing relation."""
        pass

    @abstractmethod
    async def delete_relation(
        self,
        relation_id: str
    ) -> bool:
        """Delete a relation from the graph."""
        pass

    @abstractmethod
    async def query(
        self,
        query_spec: Dict[str, Any]
    ) -> List[Entity]:
        """Execute a query against the graph."""
        pass

    @abstractmethod
    async def traverse(
        self,
        start_entity: str,
        traversal_spec: Dict[str, Any]
    ) -> List[Entity]:
        """Traverse the graph starting from a given entity."""
        pass

    @abstractmethod
    async def begin_transaction(self) -> None:
        """Begin a storage transaction."""
        pass

    @abstractmethod
    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        pass

    @abstractmethod
    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        pass