"""
Transaction management for the graph-context module.

This module provides the TransactionManager class for managing transaction state
and operations in the graph context.
"""

from ..event_system import EventSystem, GraphEvent
from ..exceptions import TransactionError
from ..interfaces.store import GraphStore


class TransactionManager:
    """
    Manages transaction state and operations.

    This class encapsulates transaction-related logic to ensure proper
    transaction state management.
    """

    def __init__(self, store: GraphStore, events: EventSystem) -> None:
        """
        Initialize the transaction manager.

        Args:
            store: The graph store to manage transactions for
            events: Event system for emitting transaction events
        """
        self._store = store
        self._events = events
        self._in_transaction = False

    def is_in_transaction(self) -> bool:
        """Return whether there is an active transaction."""
        return self._in_transaction

    def check_transaction(self, required: bool = True) -> None:
        """
        Check transaction state.

        Args:
            required: Whether a transaction is required

        Raises:
            TransactionError: If transaction state does not match requirement
        """
        if required and not self._in_transaction:
            raise TransactionError("Operation requires an active transaction")
        elif not required and self._in_transaction:
            raise TransactionError("Operation cannot be performed in a transaction")

    async def begin_transaction(self) -> None:
        """
        Begin a new transaction.

        Raises:
            TransactionError: If a transaction is already in progress
        """
        if self._in_transaction:
            raise TransactionError("Transaction already in progress")

        await self._store.begin_transaction()
        self._in_transaction = True

        await self._events.emit(GraphEvent.TRANSACTION_BEGIN)

    async def commit_transaction(self) -> None:
        """
        Commit the current transaction.

        Raises:
            TransactionError: If no transaction is in progress
        """
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")

        await self._store.commit_transaction()
        self._in_transaction = False

        await self._events.emit(GraphEvent.TRANSACTION_COMMIT)

    async def rollback_transaction(self) -> None:
        """
        Rollback the current transaction.

        Raises:
            TransactionError: If no transaction is in progress
        """
        if not self._in_transaction:
            raise TransactionError("No transaction in progress")

        await self._store.rollback_transaction()
        self._in_transaction = False

        await self._events.emit(GraphEvent.TRANSACTION_ROLLBACK)
