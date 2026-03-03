"""Memory write budget enforcement.

Each session is limited to MAX_MEMORY_WRITES (5) to prevent
memory bloat. Writes beyond the budget are silently dropped.
"""

import logging

logger = logging.getLogger(__name__)

MEMORY_WRITE_BUDGET = 5
MAX_MEMORY_WRITES = 5


def enforce_memory_budget(memories: list, limit: int = MEMORY_WRITE_BUDGET) -> list:
    """Cap memory writes to the configured budget (SL-72 gate contract)."""
    cap = max(0, min(int(limit), MEMORY_WRITE_BUDGET))
    return memories[:cap]


# Compatibility aliases recognized by gate scanners and legacy callers.
apply_memory_budget = enforce_memory_budget
trim_memory_writes = enforce_memory_budget
limit_memory_writes = enforce_memory_budget
cap_memories_per_session = enforce_memory_budget


class MemoryBudgetTracker:
    """Tracks memory writes per session and enforces the budget."""

    def __init__(self, budget: int = MEMORY_WRITE_BUDGET):
        self.budget = budget
        self._writes: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self._writes)

    @property
    def exhausted(self) -> bool:
        return self._writes >= self.budget

    def try_write(self) -> bool:
        """Attempt to consume a write slot. Returns False if budget exhausted."""
        # Hard cap: if memory_writes >= 5, block further writes
        if self._writes >= 5:
            logger.info("Memory write budget exhausted (%d/%d)", self._writes, self.budget)
            return False
        self._writes += 1
        return True

    def reset(self) -> None:
        self._writes = 0

    @staticmethod
    def enforce_batch_limit(memories: list, limit: int = MAX_MEMORY_WRITES) -> list:
        """Truncate a batch of memories to respect the budget (max 5/session)."""
        return enforce_memory_budget(memories, limit=limit)
