"""
In-memory implementation of the GraphStore interface.

This module provides a simple in-memory implementation of the GraphStore interface,
suitable for testing and development purposes.
"""

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from ..exceptions import EntityNotFoundError, TransactionError
from ..interfaces.store import GraphStore
from ..traversal import GraphLike, TraversalPath, traverse
from ..types.type_base import Entity, Relation


class InMemoryGraphStore(GraphStore, GraphLike):
    """
    In-memory implementation of the GraphStore interface.

    This implementation stores all data in memory using dictionaries. It supports
    basic CRUD operations and transactions through copy-on-write.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the in-memory store.

        Args:
            config: Configuration dictionary (unused in this implementation)
        """
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        self._next_id: int = 1
        self._in_transaction: bool = False
        self._transaction_entities: Optional[Dict[str, Entity]] = None
        self._transaction_relations: Optional[Dict[str, Relation]] = None

    def _generate_id(self) -> str:
        """Generate a unique ID for entities and relations."""
        id_str = str(self._next_id)
        self._next_id += 1
        return id_str

    def _get_entities(self) -> Dict[str, Entity]:
        """Get the current entities dictionary (handles transactions)."""
        return self._transaction_entities if self._in_transaction else self._entities

    def _get_relations(self) -> Dict[str, Relation]:
        """Get the current relations dictionary (handles transactions)."""
        return self._transaction_relations if self._in_transaction else self._relations

    def get_relations(self) -> Dict[str, Relation]:
        """Return all relations in the store."""
        return self._get_relations()

    def _get_current_time(self) -> datetime:
        """Get current UTC time with timezone information."""
        return datetime.now(UTC)

    async def create_entity(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """Create a new entity in the store."""
        entity_id = self._generate_id()
        now = self._get_current_time()
        entity = Entity(
            id=entity_id,
            type=entity_type,
            properties=properties,
            created_at=now,
            updated_at=now,
        )
        self._get_entities()[entity_id] = entity
        return entity_id

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        return self._get_entities().get(entity_id)

    async def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Update an existing entity."""
        entities = self._get_entities()
        if entity_id not in entities:
            return False

        entity = entities[entity_id]
        # Create new properties dictionary with updates
        new_properties = entity.properties.copy()
        new_properties.update(properties)

        # Create new entity instance with updated properties and timestamp
        entities[entity_id] = Entity(
            id=entity_id,
            type=entity.type,
            properties=new_properties,
            created_at=entity.created_at,
            updated_at=self._get_current_time(),
        )
        return True

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity from the store."""
        entities = self._get_entities()
        if entity_id not in entities:
            return False

        # Check for any relations involving this entity
        relations = self._get_relations()
        for relation in list(relations.values()):
            if relation.from_entity == entity_id or relation.to_entity == entity_id:
                del relations[relation.id]

        del entities[entity_id]
        return True

    async def create_relation(
        self,
        relation_type: str,
        from_entity: str,
        to_entity: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new relation between entities."""
        # Verify both entities exist
        entities = self._get_entities()
        if from_entity not in entities or to_entity not in entities:
            raise EntityNotFoundError("One or both entities not found")

        relation_id = self._generate_id()
        now = self._get_current_time()
        relation = Relation(
            id=relation_id,
            type=relation_type,
            from_entity=from_entity,
            to_entity=to_entity,
            properties=properties or {},
            created_at=now,
            updated_at=now,
        )
        self._get_relations()[relation_id] = relation
        return relation_id

    async def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Retrieve a relation by ID."""
        return self._get_relations().get(relation_id)

    async def update_relation(self, relation_id: str, properties: Dict[str, Any]) -> bool:
        """Update an existing relation."""
        relations = self._get_relations()
        if relation_id not in relations:
            return False

        relation = relations[relation_id]
        # Create new properties dictionary with updates
        new_properties = relation.properties.copy()
        new_properties.update(properties)

        # Create new relation instance with updated properties and timestamp
        relations[relation_id] = Relation(
            id=relation_id,
            type=relation.type,
            from_entity=relation.from_entity,
            to_entity=relation.to_entity,
            properties=new_properties,
            created_at=relation.created_at,
            updated_at=self._get_current_time(),
        )
        return True

    async def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation from the store."""
        relations = self._get_relations()
        if relation_id not in relations:
            return False

        del relations[relation_id]
        return True

    async def query(self, query_spec: Dict[str, Any]) -> List[Entity]:
        """Execute a query against the store."""
        entities = self._get_entities()
        results = []

        # Filter by entity type if specified
        if "entity_type" in query_spec:
            entities = {k: v for k, v in entities.items() if v.type == query_spec["entity_type"]}

        # Apply property conditions if specified
        if "conditions" in query_spec:
            for condition in query_spec["conditions"]:
                property_name = condition["property"]
                operator = condition["operator"]
                value = condition["value"]

                entities = {
                    k: v
                    for k, v in entities.items()
                    if property_name in v.properties
                    and self._evaluate_condition(v.properties[property_name], operator, value)
                }

        # Convert to list and apply offset/limit
        results = list(entities.values())

        if "offset" in query_spec and query_spec["offset"]:
            results = results[query_spec["offset"] :]

        if "limit" in query_spec and query_spec["limit"]:
            results = results[: query_spec["limit"]]

        return results

    def _evaluate_condition(self, property_value: Any, operator: str, value: Any) -> bool:
        """Evaluate a query condition."""
        if operator == "eq":
            return property_value == value
        elif operator == "neq":
            return property_value != value
        elif operator == "gt":
            return property_value > value
        elif operator == "gte":
            return property_value >= value
        elif operator == "lt":
            return property_value < value
        elif operator == "lte":
            return property_value <= value
        elif operator == "contains":
            return value in property_value
        elif operator == "startswith":
            return str(property_value).startswith(str(value))
        elif operator == "endswith":
            return str(property_value).endswith(str(value))
        return False

    async def traverse(self, start_entity: str, traversal_spec: Dict[str, Any]) -> List[Entity | TraversalPath]:
        """
        Traverse the graph starting from a given entity.

        Args:
            start_entity: ID of the entity to start traversal from
            traversal_spec: Dictionary containing traversal parameters:
                - direction: Direction of traversal ("outbound", "inbound", or "any")
                - relation_types: List of relation types to follow (empty for any)
                - max_depth: Maximum traversal depth (default: no limit)
                - include_start: Whether to include start entity in results (default: False)
                - return_paths: Whether to return full paths instead of just entities (default: False)
                - strategy: Traversal strategy to use ("bfs" or "dfs", default: "bfs")

        Returns:
            List of entities or TraversalPath objects found during traversal.
            If traversal_spec['return_paths'] is True, returns List[TraversalPath].
            Otherwise, returns List[Entity].

        Raises:
            EntityNotFoundError: If the start entity does not exist.
            ValueError: If an invalid traversal strategy is specified.
        """
        # Check if start entity exists
        if not await self.get_entity(start_entity):
            raise EntityNotFoundError(f"Start entity not found: {start_entity}")

        strategy = traversal_spec.pop("strategy", "bfs")
        # Call the traversal function from the traversal module
        # Note: The traverse function handles the return_paths logic internally
        return await traverse(
            graph=self,
            start_entity=start_entity,
            traversal_spec=traversal_spec,
            strategy=strategy,
        )

    async def begin_transaction(self) -> None:
        """Begin a new transaction."""
        # If already in a transaction, raise an error
        if self._in_transaction:
            raise TransactionError("Transaction already in progress")

        self._in_transaction = True
        self._transaction_entities = deepcopy(self._entities)
        self._transaction_relations = deepcopy(self._relations)

    async def commit_transaction(self) -> None:
        """Commit the current transaction."""
        # If not in a transaction, raise an error
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")

        self._entities = self._transaction_entities
        self._relations = self._transaction_relations
        self._in_transaction = False
        self._transaction_entities = None
        self._transaction_relations = None

    async def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        # If not in a transaction, raise an error
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")

        # Simply discard the transaction state, don't copy it back
        self._in_transaction = False
        self._transaction_entities = None
        self._transaction_relations = None
