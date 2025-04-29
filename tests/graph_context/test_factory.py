"""Tests for the graph store factory."""

import json
import os
from unittest import mock

import pytest

from graph_context.factory import GraphStoreFactory, StoreConfig
from graph_context.stores.memory_store import InMemoryGraphStore


class TestStoreConfig:
    """Tests for the StoreConfig dataclass."""

    def test_init(self):
        """Test initialization of StoreConfig."""
        config_data = {"type": "memory", "config": {"setting": "value"}}

        config = StoreConfig(**config_data)

        assert config.type == "memory"
        assert config.config == {"setting": "value"}


class TestGraphStoreFactory:
    """Tests for the GraphStoreFactory class."""

    def setup_method(self):
        """Setup for each test - clear the registered store types except default ones."""
        # Save the original store types dictionary
        self.original_store_types = GraphStoreFactory._store_types.copy()
        # Reset to only include the default memory store
        GraphStoreFactory._store_types = {"memory": InMemoryGraphStore}

    def teardown_method(self):
        """Teardown after each test - restore the original store types."""
        GraphStoreFactory._store_types = self.original_store_types

    def test_register_store_type(self):
        """Test registering a new store type."""
        # Register a custom store type
        GraphStoreFactory.register_store_type("custom", InMemoryGraphStore)

        # Verify it was registered
        assert "custom" in GraphStoreFactory._store_types
        assert GraphStoreFactory._store_types["custom"] == InMemoryGraphStore

    def test_create_default(self):
        """Test creating a store with default configuration."""
        # Mock _load_config to return a default configuration
        with mock.patch.object(
            GraphStoreFactory,
            "_load_config",
            return_value=StoreConfig(type="memory", config={}),
        ):
            store = GraphStoreFactory.create()

            # Verify the correct type was created
            assert isinstance(store, InMemoryGraphStore)

    def test_create_custom(self):
        """Test creating a custom store type."""
        # Register a custom store type
        GraphStoreFactory.register_store_type("custom", InMemoryGraphStore)

        # Mock _load_config to return custom configuration
        with mock.patch.object(
            GraphStoreFactory,
            "_load_config",
            return_value=StoreConfig(type="custom", config={"test": "value"}),
        ):
            store = GraphStoreFactory.create()

            # Verify the correct type was created
            assert isinstance(store, InMemoryGraphStore)

            # Since InMemoryGraphStore doesn't store the config directly, we need an indirect test
            # Register a store with a mock to verify config is passed correctly
            mock_class = mock.MagicMock()
            GraphStoreFactory.register_store_type("mock_store", mock_class)

            # Create a new store with mock class
            with mock.patch.object(
                GraphStoreFactory,
                "_load_config",
                return_value=StoreConfig(
                    type="mock_store", config={"mock_setting": "mock_value"}
                ),
            ):
                GraphStoreFactory.create()

                # Verify the mock was called with the correct config
                mock_class.assert_called_once_with({"mock_setting": "mock_value"})

    def test_create_unknown_type(self):
        """Test creating a store with unknown type raises error."""
        # Mock _load_config to return unknown configuration
        with mock.patch.object(
            GraphStoreFactory,
            "_load_config",
            return_value=StoreConfig(type="unknown", config={}),
        ):
            # Verify ValueError is raised
            with pytest.raises(ValueError, match="Unknown store type: unknown"):
                GraphStoreFactory.create()

    def test_load_config_from_env(self):
        """Test loading configuration from environment variable."""
        config_dict = {"type": "memory", "config": {"env_setting": "env_value"}}

        # Set environment variable with JSON string
        with mock.patch.dict(
            os.environ, {GraphStoreFactory._CONFIG_ENV_VAR: json.dumps(config_dict)}
        ):
            config = GraphStoreFactory._load_config()

            assert config.type == "memory"
            assert config.config == {"env_setting": "env_value"}

    def test_load_config_from_env_invalid_json(self):
        """Test loading configuration from environment with invalid JSON."""
        # Set environment variable with invalid JSON
        with mock.patch.dict(
            os.environ, {GraphStoreFactory._CONFIG_ENV_VAR: "invalid json"}
        ):
            with pytest.raises(ValueError, match="Invalid environment configuration"):
                GraphStoreFactory._load_config()

    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        config_dict = {"type": "memory", "config": {"file_setting": "file_value"}}

        # Mock environment to not have the variable
        # Mock os.path.exists to return True for the config file
        # Mock open to return a file-like object with the config JSON
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("os.path.exists", return_value=True),
            mock.patch(
                "builtins.open", mock.mock_open(read_data=json.dumps(config_dict))
            ),
        ):
            config = GraphStoreFactory._load_config()

            assert config.type == "memory"
            assert config.config == {"file_setting": "file_value"}

    def test_load_config_from_file_invalid_json(self):
        """Test loading configuration from file with invalid JSON."""
        # Mock environment to not have the variable
        # Mock os.path.exists to return True for the config file
        # Mock open to return a file-like object with invalid JSON
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("os.path.exists", return_value=True),
            mock.patch("builtins.open", mock.mock_open(read_data="invalid json")),
        ):
            with pytest.raises(ValueError, match="Invalid configuration file"):
                GraphStoreFactory._load_config()

    def test_load_config_file_not_found(self):
        """Test falling back to default when config file not found."""
        # Mock environment to not have the variable
        # Mock os.path.exists to return False for the config file
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("os.path.exists", return_value=False),
        ):
            config = GraphStoreFactory._load_config()

            # Should return default config
            assert config.type == "memory"
            assert config.config == {}

    def test_load_config_file_permission_error(self):
        """Test handling permission error when reading config file."""
        # Mock environment to not have the variable
        # Mock os.path.exists to return True for the config file
        # Mock open to raise a permission error
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("os.path.exists", return_value=True),
            mock.patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ),
        ):
            with pytest.raises(ValueError, match="Invalid configuration file"):
                GraphStoreFactory._load_config()

    @pytest.mark.asyncio
    async def test_load_config_full_flow(self):
        """Test the complete config loading flow covering all execution paths."""
        # Test with specific and comprehensive config
        config_dict = {
            "type": "custom_type",
            "config": {
                "setting1": "value1",
                "setting2": [1, 2, 3],
                "setting3": {"nested": True},
            },
        }

        # Mock store class
        mock_store = mock.MagicMock()

        # Register the mock store
        GraphStoreFactory.register_store_type("custom_type", mock_store)

        # Test end-to-end flow with environment variable
        with mock.patch.dict(
            os.environ, {GraphStoreFactory._CONFIG_ENV_VAR: json.dumps(config_dict)}
        ):
            with mock.patch.object(
                GraphStoreFactory, "_store_types"
            ) as mock_store_types:
                # Set up the mock store types
                mock_store_types.__getitem__.return_value = mock_store
                mock_store_types.__contains__.return_value = True

                # Call create() which should use the environment config
                GraphStoreFactory.create()

                # Verify mock_store was called with the correct config
                mock_store.assert_called_once_with(
                    {
                        "setting1": "value1",
                        "setting2": [1, 2, 3],
                        "setting3": {"nested": True},
                    }
                )

    def test_load_config_typeerror_handling(self):
        """Test handling TypeError during config loading from env var."""
        # Set environment variable with valid JSON but invalid config structure
        with mock.patch.dict(
            os.environ, {GraphStoreFactory._CONFIG_ENV_VAR: '{"not_type": "invalid"}'}
        ):
            with pytest.raises(ValueError, match="Invalid environment configuration"):
                GraphStoreFactory._load_config()

    def test_config_file_not_found_default(self):
        """Test default config when environment variable isn't set and file doesn't exist."""
        # Ensure environment variable is not set
        with mock.patch.dict(os.environ, {}, clear=True):
            # Ensure config file doesn't exist
            with mock.patch("os.path.exists", return_value=False):
                config = GraphStoreFactory._load_config()

                # Should return default config
                assert config.type == "memory"
                assert config.config == {}

    def test_config_file_oserror_handling(self):
        """Test handling OSError during config file reading."""
        # Ensure environment variable is not set
        with mock.patch.dict(os.environ, {}, clear=True):
            # Ensure config file exists
            with mock.patch("os.path.exists", return_value=True):
                # But opening it raises an OSError (not just PermissionError)
                with mock.patch("builtins.open", side_effect=OSError("Some OS error")):
                    with pytest.raises(ValueError, match="Invalid configuration file"):
                        GraphStoreFactory._load_config()

    def test_register_and_get_store_types(self):
        """Test registering multiple store types and retrieving them."""
        # Save original store types
        original_store_types = GraphStoreFactory._store_types.copy()

        try:
            # Start with clean store types
            GraphStoreFactory._store_types = {"memory": InMemoryGraphStore}

            # Register additional mock stores
            mock_store1 = mock.MagicMock()
            mock_store2 = mock.MagicMock()

            GraphStoreFactory.register_store_type("mock1", mock_store1)
            GraphStoreFactory.register_store_type("mock2", mock_store2)

            # Verify all store types are registered
            assert "memory" in GraphStoreFactory._store_types
            assert "mock1" in GraphStoreFactory._store_types
            assert "mock2" in GraphStoreFactory._store_types
            assert GraphStoreFactory._store_types["mock1"] == mock_store1
            assert GraphStoreFactory._store_types["mock2"] == mock_store2

            # Test with invalid store type
            with mock.patch.object(
                GraphStoreFactory,
                "_load_config",
                return_value=StoreConfig(type="nonexistent_type", config={}),
            ):
                with pytest.raises(
                    ValueError, match="Unknown store type: nonexistent_type"
                ):
                    GraphStoreFactory.create()

        finally:
            # Restore original store types
            GraphStoreFactory._store_types = original_store_types
