"""
Tests for the TransactionManager class.
"""
from unittest.mock import AsyncMock

import pytest

from graph_context.context_base import TransactionManager
from graph_context.event_system import GraphEvent
from graph_context.exceptions import TransactionError


@pytest.fixture
def mock_store():
    """Mock GraphStore for testing."""
    store = AsyncMock()
    store.begin_transaction = AsyncMock()
    store.commit_transaction = AsyncMock()
    store.rollback_transaction = AsyncMock()
    return store


@pytest.fixture
def mock_events():
    """Mock EventSystem for testing."""
    events = AsyncMock()
    events.emit = AsyncMock()
    return events


@pytest.fixture
def transaction_manager(mock_store, mock_events):
    """Transaction manager instance for testing."""
    return TransactionManager(mock_store, mock_events)


class TestTransactionManager:
    """Test cases for TransactionManager class."""

    def test_init(self, mock_store, mock_events):
        """Test TransactionManager initialization."""
        manager = TransactionManager(mock_store, mock_events)
        assert manager._store is mock_store
        assert manager._events is mock_events
        assert manager._in_transaction is False

    def test_is_in_transaction(self, transaction_manager):
        """Test is_in_transaction returns correct state."""
        assert transaction_manager.is_in_transaction() is False

        # Manually set transaction state to verify getter
        transaction_manager._in_transaction = True
        assert transaction_manager.is_in_transaction() is True

    def test_check_transaction_required_true(self, transaction_manager):
        """Test check_transaction with required=True."""
        # When not in transaction but required
        with pytest.raises(TransactionError) as exc_info:
            transaction_manager.check_transaction(required=True)
        assert "Operation requires an active transaction" in str(exc_info.value)

        # When in transaction and required
        transaction_manager._in_transaction = True
        transaction_manager.check_transaction(required=True)  # Should not raise

    def test_check_transaction_required_false(self, transaction_manager):
        """Test check_transaction with required=False."""
        # When not in transaction and not required
        transaction_manager.check_transaction(required=False)  # Should not raise

        # When in transaction but not required
        transaction_manager._in_transaction = True
        with pytest.raises(TransactionError) as exc_info:
            transaction_manager.check_transaction(required=False)
        assert "Operation cannot be performed in a transaction" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_begin_transaction_success(self, transaction_manager, mock_store, mock_events):
        """Test begin_transaction successful execution."""
        await transaction_manager.begin_transaction()

        mock_store.begin_transaction.assert_called_once()
        assert transaction_manager._in_transaction is True
        mock_events.emit.assert_called_once_with(GraphEvent.TRANSACTION_BEGIN)

    @pytest.mark.asyncio
    async def test_begin_transaction_already_active(self, transaction_manager):
        """Test begin_transaction when a transaction is already active."""
        transaction_manager._in_transaction = True

        with pytest.raises(TransactionError) as exc_info:
            await transaction_manager.begin_transaction()
        assert "Transaction already in progress" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_commit_transaction_success(self, transaction_manager, mock_store, mock_events):
        """Test commit_transaction successful execution."""
        transaction_manager._in_transaction = True

        await transaction_manager.commit_transaction()

        mock_store.commit_transaction.assert_called_once()
        assert transaction_manager._in_transaction is False
        mock_events.emit.assert_called_once_with(GraphEvent.TRANSACTION_COMMIT)

    @pytest.mark.asyncio
    async def test_commit_transaction_no_active(self, transaction_manager):
        """Test commit_transaction when no transaction is active."""
        with pytest.raises(TransactionError) as exc_info:
            await transaction_manager.commit_transaction()
        assert "No transaction in progress" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rollback_transaction_success(self, transaction_manager, mock_store, mock_events):
        """Test rollback_transaction successful execution."""
        transaction_manager._in_transaction = True

        await transaction_manager.rollback_transaction()

        mock_store.rollback_transaction.assert_called_once()
        assert transaction_manager._in_transaction is False
        mock_events.emit.assert_called_once_with(GraphEvent.TRANSACTION_ROLLBACK)

    @pytest.mark.asyncio
    async def test_rollback_transaction_no_active(self, transaction_manager):
        """Test rollback_transaction when no transaction is active."""
        with pytest.raises(TransactionError) as exc_info:
            await transaction_manager.rollback_transaction()
        assert "No transaction in progress" in str(exc_info.value)
