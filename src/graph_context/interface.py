"""
Core interface for the graph-context module.

This module defines the abstract base class that all graph context implementations
must inherit from, providing a consistent interface for graph operations.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar

from .types.type_base import Entity, QuerySpec, Relation, TraversalSpec

T = TypeVar('T')


class GraphContext(ABC):
    """
    Abstract base class defining the core graph operations interface.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the graph context.

        This method should be called before any other operations.
        It should set up any necessary connections, create indexes, etc.

        Raises:
            GraphContextError: If initialization fails
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up the graph context.

        This method should be called when the context is no longer needed.
        It should close connections, clean up resources, etc.

        Raises:
            GraphContextError: If cleanup fails
        """
        pass

    @abstractmethod
    async def create_entity(
        self,
        entity_type: str,
        properties: Dict[str, Any]
    ) -> Entity:
        """
        Create a new entity in the graph.

        Args:
            entity_type: Type of the entity to create
            properties: Dictionary of property values

        Returns:
            The created entity

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the entity type is not defined in the schema
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def get_entity(
        self,
        entity_id: str
    ) -> Optional[Entity]:
        """
        Retrieve an entity by ID.

        Args:
            entity_id: ID of the entity to retrieve

        Returns:
            The entity if found, None otherwise

        Raises:
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> Optional[Entity]:
        """
        Update an existing entity.

        Args:
            entity_id: ID of the entity to update
            properties: Dictionary of property values to update

        Returns:
            The updated entity if successful, None if the entity was not found

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the updated properties violate the schema
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def delete_entity(
        self,
        entity_id: str
    ) -> bool:
        """
        Delete an entity from the graph.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if the entity was deleted, False if it was not found

        Raises:
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Relation:
        """
        Create a new relation between entities.

        Args:
            relation_type: Type of the relation to create
            from_entity: ID of the source entity
            to_entity: ID of the target entity
            properties: Optional dictionary of property values

        Returns:
            The created relation

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the relation type is not defined in the schema
            EntityNotFoundError: If either entity does not exist
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def get_relation(
        self,
        relation_id: str
    ) -> Optional[Relation]:
        """
        Retrieve a relation by ID.

        Args:
            relation_id: ID of the relation to retrieve

        Returns:
            The relation if found, None otherwise

        Raises:
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def update_relation(
        self,
        relation_id: str,
        properties: Dict[str, Any]
    ) -> Optional[Relation]:
        """
        Update an existing relation.

        Args:
            relation_id: ID of the relation to update
            properties: Dictionary of property values to update

        Returns:
            The updated relation if successful, None if the relation was not found

        Raises:
            ValidationError: If property validation fails
            SchemaError: If the updated properties violate the schema
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def delete_relation(
        self,
        relation_id: str
    ) -> bool:
        """
        Delete a relation from the graph.

        Args:
            relation_id: ID of the relation to delete

        Returns:
            True if the relation was deleted, False if it was not found

        Raises:
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def query(
        self,
        query_spec: QuerySpec
    ) -> List[Entity]:
        """
        Execute a query against the graph.

        Args:
            query_spec: Specification of the query to execute

        Returns:
            List of entities matching the query

        Raises:
            ValidationError: If the query specification is invalid
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def traverse(
        self,
        start_entity: str,
        traversal_spec: TraversalSpec
    ) -> List[Entity]:
        """
        Traverse the graph starting from a given entity.

        Args:
            start_entity: ID of the entity to start traversal from
            traversal_spec: Specification of the traversal

        Returns:
            List of entities found during traversal

        Raises:
            EntityNotFoundError: If the start entity does not exist
            ValidationError: If the traversal specification is invalid
            GraphContextError: If the operation fails
        """
        pass

    @abstractmethod
    async def begin_transaction(self) -> None:
        """
        Begin a new transaction.

        This method should be called before a series of operations that need
        to be executed atomically.

        Raises:
            TransactionError: If a transaction is already in progress or if
                            the operation fails
        """
        pass

    @abstractmethod
    async def commit_transaction(self) -> None:
        """
        Commit the current transaction.

        This method should be called to persist the changes made during the
        current transaction.

        Raises:
            TransactionError: If no transaction is in progress or if the
                            operation fails
        """
        pass

    @abstractmethod
    async def rollback_transaction(self) -> None:
        """
        Roll back the current transaction.

        This method should be called to discard the changes made during the
        current transaction.

        Raises:
            TransactionError: If no transaction is in progress or if the
                            operation fails
        """
        pass