# Graph Context

A unified interface for graph operations in the Knowledge Graph Assisted Research IDE.

## Features

- Unified interface for graph operations
- Support for different backend implementations
- Strong type safety with Pydantic models
- Comprehensive validation for entities and relations
- Transaction support
- Flexible query and traversal specifications
- Clean separation of concerns

## Installation

```bash
poetry install
```

## Usage

```python
from graph_context import GraphContext, Entity, EntityType, PropertyType

# Create a graph context instance (with your chosen backend)
context = YourGraphContextImplementation()

# Define entity types
person_type = EntityType(
    name="Person",
    properties={
        "name": PropertyDefinition(type=PropertyType.STRING, required=True),
        "age": PropertyDefinition(type=PropertyType.INTEGER)
    }
)

# Register types
context.register_entity_type(person_type)

# Create entities
person = await context.create_entity(
    entity_type="Person",
    properties={
        "name": "Alice",
        "age": 30
    }
)
```

## Development

### Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   poetry install
   ```
3. Run tests:
   ```bash
   poetry run pytest
   ```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=graph_context

# Run specific test file
poetry run pytest tests/test_specific.py
```

## License

MIT License - See LICENSE file for details