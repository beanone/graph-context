"""Integration tests for the graph context system."""

import pytest

from graph_context.context_base import BaseGraphContext
from graph_context.types.type_base import EntityType, PropertyDefinition, RelationType


@pytest.mark.integration
@pytest.mark.asyncio
async def test_transaction_wrapping_for_entity_creation():
    """Test proper transaction wrapping for entity creation and related operations."""
    # Create a fresh context for this test
    context = BaseGraphContext()

    try:
        # Define test data
        person_type = "Person"
        company_type = "Company"
        person_props = {"name": "John Doe", "age": 30}
        company_props = {"name": "Acme Corp", "industry": "Technology"}
        relation_type = "WORKS_AT"
        relation_props = {"start_date": "2023-01-01", "role": "Engineer"}

        # Register entity types
        person_entity_type = EntityType(
            name=person_type,
            properties={
                "name": PropertyDefinition(type="string", required=True),
                "age": PropertyDefinition(type="integer", required=True),
            },
        )
        company_entity_type = EntityType(
            name=company_type,
            properties={
                "name": PropertyDefinition(type="string", required=True),
                "industry": PropertyDefinition(type="string", required=True),
            },
        )
        await context.register_entity_type(person_entity_type)
        await context.register_entity_type(company_entity_type)

        # Register relation type
        works_at_relation_type = RelationType(
            name=relation_type,
            from_types=[person_type],
            to_types=[company_type],
            properties={
                "start_date": PropertyDefinition(type="string", required=True),
                "role": PropertyDefinition(type="string", required=True),
            },
        )
        await context.register_relation_type(works_at_relation_type)

        try:
            # Start transaction
            await context.begin_transaction()

            # Create person entity
            person_id = await context.create_entity(person_type, person_props)
            assert person_id is not None

            # Create company entity
            company_id = await context.create_entity(company_type, company_props)
            assert company_id is not None

            # Create relation between person and company
            relation_id = await context.create_relation(
                relation_type, person_id, company_id, relation_props
            )
            assert relation_id is not None

            # Verify entities and relation exist within transaction
            person = await context.get_entity(person_id)
            assert person is not None
            assert person.type == person_type
            assert person.properties == person_props

            company = await context.get_entity(company_id)
            assert company is not None
            assert company.type == company_type
            assert company.properties == company_props

            relation = await context.get_relation(relation_id)
            assert relation is not None
            assert relation.type == relation_type
            assert relation.from_entity == person_id
            assert relation.to_entity == company_id
            assert relation.properties == relation_props

            # Commit transaction
            await context.commit_transaction()

            # Verify entities and relation still exist after commit
            person = await context.get_entity(person_id)
            assert person is not None
            company = await context.get_entity(company_id)
            assert company is not None
            relation = await context.get_relation(relation_id)
            assert relation is not None

        except Exception as e:
            # Rollback transaction on error
            await context.rollback_transaction()
            raise e

        # Test rollback behavior
        try:
            await context.begin_transaction()

            # Create an entity
            entity_id = await context.create_entity(person_type, person_props)
            assert entity_id is not None

            # Simulate an error condition
            raise ValueError("Simulated error")

        except ValueError:
            # Rollback should undo the entity creation
            await context.rollback_transaction()

            # Verify entity doesn't exist after rollback
            entity = await context.get_entity(entity_id)
            assert entity is None

    finally:
        # Clean up the context
        await context.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_operations():
    """Test query operations with multiple conditions."""
    # Create a fresh context for this test
    context = BaseGraphContext()

    try:
        # Register entity type
        person_type = "Person"
        person_entity_type = EntityType(
            name=person_type,
            properties={
                "name": PropertyDefinition(type="string", required=True),
                "age": PropertyDefinition(type="integer", required=True),
            },
        )
        await context.register_entity_type(person_entity_type)

        # Register relation type
        knows_type = "KNOWS"
        knows_relation_type = RelationType(
            name=knows_type,
            from_types=[person_type],
            to_types=[person_type],
            properties={
                "since": PropertyDefinition(type="integer", required=True),
            },
        )
        await context.register_relation_type(knows_relation_type)

        # Create entities and relation in a transaction
        await context.begin_transaction()
        try:
            # Create first person
            person1_id = await context.create_entity(
                person_type, {"name": "John Doe", "age": 30}
            )

            # Create second person
            person2_id = await context.create_entity(
                person_type, {"name": "Jane Doe", "age": 28}
            )

            # Create relation
            await context.create_relation(
                knows_type, person1_id, person2_id, {"since": 2023}
            )
            await context.commit_transaction()
        except Exception:
            await context.rollback_transaction()
            raise

        # Test single condition query
        query_spec = {
            "type": person_type,
            "conditions": [{"field": "name", "operator": "eq", "value": "John Doe"}],
        }
        results = await context.query(query_spec)
        assert len(results) == 1
        assert results[0].id == person1_id
        assert results[0].type == person_type
        assert results[0].properties["name"] == "John Doe"
        assert results[0].properties["age"] == 30

        # Test multiple conditions
        query_spec = {
            "type": person_type,
            "conditions": [
                {"field": "name", "operator": "eq", "value": "John Doe"},
                {"field": "age", "operator": "gt", "value": 25},
            ],
        }
        results = await context.query(query_spec)
        assert len(results) == 1
        assert results[0].id == person1_id

        # Test no results query
        query_spec = {
            "type": person_type,
            "conditions": [{"field": "name", "operator": "eq", "value": "Bob Smith"}],
        }
        results = await context.query(query_spec)
        assert len(results) == 0

    finally:
        # Clean up the context
        await context.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_with_multiple_conditions():
    """Test query with multiple conditions."""
    # Create a fresh context for this test
    context = BaseGraphContext()

    try:
        # Register entity type
        await context.register_entity_type(
            EntityType(
                name="Person",
                properties={
                    "name": PropertyDefinition(type="string", required=True),
                    "age": PropertyDefinition(type="integer", required=True),
                    "city": PropertyDefinition(type="string", required=True),
                },
            )
        )

        # Begin transaction
        await context.begin_transaction()

        # Create test entities
        await context.create_entity(
            "Person", {"name": "John Doe", "age": 30, "city": "New York"}
        )
        await context.create_entity(
            "Person", {"name": "Jane Doe", "age": 28, "city": "Boston"}
        )
        await context.create_entity(
            "Person", {"name": "John Smith", "age": 30, "city": "New York"}
        )

        # Commit transaction
        await context.commit_transaction()

        # Test query with multiple conditions
        query_spec = {
            "entity_type": "Person",
            "conditions": [
                {"field": "age", "operator": "eq", "value": 30},
                {"field": "city", "operator": "eq", "value": "New York"},
            ],
        }
        results = await context.query(query_spec)
        assert len(results) == 2
        names = {result.properties["name"] for result in results}
        assert names == {"John Doe", "John Smith"}

        # Test query with no matching results
        query_spec = {
            "entity_type": "Person",
            "conditions": [
                {"field": "age", "operator": "eq", "value": 30},
                {"field": "city", "operator": "eq", "value": "Boston"},
            ],
        }
        results = await context.query(query_spec)
        assert len(results) == 0

    finally:
        # Clean up the context
        await context.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_with_invalid_entity_type():
    """Test query with an invalid entity type."""
    # Create a fresh context for this test
    context = BaseGraphContext()

    try:
        # Register entity type
        person_type = "Person"
        person_entity_type = EntityType(
            name=person_type,
            properties={
                "name": PropertyDefinition(type="string", required=True),
                "age": PropertyDefinition(type="integer", required=True),
            },
        )
        await context.register_entity_type(person_entity_type)

        # Create test entity
        await context.begin_transaction()
        try:
            await context.create_entity(person_type, {"name": "John Doe", "age": 30})
            await context.commit_transaction()
        except Exception:
            await context.rollback_transaction()
            raise

        # Test query with invalid entity type
        query_spec = {
            "entity_type": "InvalidType",
            "conditions": [{"field": "name", "operator": "eq", "value": "John Doe"}],
        }
        results = await context.query(query_spec)
        assert len(results) == 0  # Should return empty list for invalid type

    finally:
        # Clean up the context
        await context.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_type_existence_checks():
    """Test checking if entity and relation types exist."""
    # Create a fresh context for this test
    context = BaseGraphContext()

    try:
        # Register entity type
        person_type = "Person"
        person_entity_type = EntityType(
            name=person_type,
            properties={
                "name": PropertyDefinition(type="string", required=True),
                "age": PropertyDefinition(type="integer", required=True),
            },
        )
        await context.register_entity_type(person_entity_type)

        # Register relation type
        knows_type = "KNOWS"
        knows_relation_type = RelationType(
            name=knows_type,
            from_types=[person_type],
            to_types=[person_type],
            properties={
                "since": PropertyDefinition(type="integer", required=True),
            },
        )
        await context.register_relation_type(knows_relation_type)

        # Test entity type existence
        assert await context.has_entity_type(person_type) is True
        assert await context.has_entity_type("InvalidType") is False

        # Test relation type existence
        assert await context.has_relation_type(knows_type) is True
        assert await context.has_relation_type("InvalidType") is False

    finally:
        # Clean up the context
        await context.cleanup()
