"""Tests for the cache configuration module."""

import pytest
from datetime import datetime, timedelta
from graph_context.caching.config import CacheMetrics, CacheConfig


class TestCacheMetrics:
    def test_default_initialization(self):
        """Test default initialization of CacheMetrics."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.total_time == 0.0
        assert isinstance(metrics.created_at, datetime)

    def test_to_dict_empty_metrics(self):
        """Test to_dict() with no hits or misses."""
        metrics = CacheMetrics()
        result = metrics.to_dict()

        assert result["hits"] == 0
        assert result["misses"] == 0
        assert result["total"] == 0
        assert result["hit_rate"] == 0.0
        assert result["total_time"] == 0.0
        assert result["avg_time"] == 0.0
        assert isinstance(result["created_at"], str)

    def test_to_dict_with_activity(self):
        """Test to_dict() with hits, misses and time."""
        metrics = CacheMetrics(hits=75, misses=25, total_time=2.0)
        result = metrics.to_dict()

        assert result["hits"] == 75
        assert result["misses"] == 25
        assert result["total"] == 100
        assert result["hit_rate"] == 0.75
        assert result["total_time"] == 2.0
        assert result["avg_time"] == 0.02  # 2.0 / 100

    def test_metrics_with_custom_creation_time(self):
        """Test metrics with a specific creation time."""
        creation_time = datetime.now() - timedelta(hours=1)
        metrics = CacheMetrics(created_at=creation_time)
        result = metrics.to_dict()

        assert result["created_at"] == creation_time.isoformat()


class TestCacheConfig:
    def test_default_initialization(self):
        """Test default initialization of CacheConfig."""
        config = CacheConfig()

        # Check default values
        assert config.max_size == 1000
        assert config.default_ttl == 300.0
        assert config.enable_metrics is True

        # Entity cache settings
        assert config.entity_cache_size == 1000
        assert config.entity_cache_ttl == 300.0

        # Relation cache settings
        assert config.relation_cache_size == 1000
        assert config.relation_cache_ttl == 300.0

        # Query cache settings
        assert config.query_cache_size == 500
        assert config.query_cache_ttl == 60.0

        # Traversal cache settings
        assert config.traversal_cache_size == 500
        assert config.traversal_cache_ttl == 60.0

    @pytest.mark.parametrize("type_name,expected_ttl", [
        ("entity", 300.0),
        ("relation", 300.0),
        ("query", 60.0),
        ("traversal", 60.0),
        ("unknown", 300.0),  # Should return default_ttl
    ])
    def test_get_ttl_for_type(self, type_name, expected_ttl):
        """Test getting TTL for different types."""
        config = CacheConfig()
        assert config.get_ttl_for_type(type_name) == expected_ttl

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = CacheConfig(
            max_size=2000,
            default_ttl=600.0,
            enable_metrics=False,
            entity_cache_size=1500,
            entity_cache_ttl=400.0,
            relation_cache_size=1500,
            relation_cache_ttl=400.0,
            query_cache_size=750,
            query_cache_ttl=120.0,
            traversal_cache_size=750,
            traversal_cache_ttl=120.0,
        )

        result = config.to_dict()

        assert result == {
            "max_size": 2000,
            "default_ttl": 600.0,
            "enable_metrics": False,
            "entity_cache_size": 1500,
            "entity_cache_ttl": 400.0,
            "relation_cache_size": 1500,
            "relation_cache_ttl": 400.0,
            "query_cache_size": 750,
            "query_cache_ttl": 120.0,
            "traversal_cache_size": 750,
            "traversal_cache_ttl": 120.0,
        }

    def test_from_dict_full_config(self):
        """Test creating config from complete dictionary."""
        config_dict = {
            "max_size": 2000,
            "default_ttl": 600.0,
            "enable_metrics": False,
            "entity_cache_size": 1500,
            "entity_cache_ttl": 400.0,
            "relation_cache_size": 1500,
            "relation_cache_ttl": 400.0,
            "query_cache_size": 750,
            "query_cache_ttl": 120.0,
            "traversal_cache_size": 750,
            "traversal_cache_ttl": 120.0,
        }

        config = CacheConfig.from_dict(config_dict)

        assert config.max_size == 2000
        assert config.default_ttl == 600.0
        assert config.enable_metrics is False
        assert config.entity_cache_size == 1500
        assert config.entity_cache_ttl == 400.0
        assert config.relation_cache_size == 1500
        assert config.relation_cache_ttl == 400.0
        assert config.query_cache_size == 750
        assert config.query_cache_ttl == 120.0
        assert config.traversal_cache_size == 750
        assert config.traversal_cache_ttl == 120.0

    def test_from_dict_partial_config(self):
        """Test creating config from partial dictionary."""
        config_dict = {
            "max_size": 2000,
            "entity_cache_ttl": 400.0,
        }

        config = CacheConfig.from_dict(config_dict)

        # Check specified values
        assert config.max_size == 2000
        assert config.entity_cache_ttl == 400.0

        # Check default values are used for unspecified fields
        assert config.default_ttl == 300.0
        assert config.enable_metrics is True
        assert config.entity_cache_size == 1000
        assert config.relation_cache_size == 1000
        assert config.relation_cache_ttl == 300.0
        assert config.query_cache_size == 500
        assert config.query_cache_ttl == 60.0
        assert config.traversal_cache_size == 500
        assert config.traversal_cache_ttl == 60.0