"""
Short-term in-process memory for active tasks and recent context.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class MemoryEntry:
    key: str
    value: Any
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: list[str] = field(default_factory=list)


class ShortTermMemory:
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self._store: dict[str, MemoryEntry] = {}
        self._order: deque[str] = deque(maxlen=capacity)

    def set(self, key: str, value: Any, tags: list[str] | None = None):
        entry = MemoryEntry(key=key, value=value, tags=tags or [])
        if key in self._store:
            self._order.remove(key)
        elif len(self._order) >= self.capacity:
            oldest = self._order[0]
            del self._store[oldest]
        self._store[key] = entry
        self._order.append(key)

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        return entry.value if entry else None

    def get_entry(self, key: str) -> Optional[MemoryEntry]:
        return self._store.get(key)

    def search_by_tag(self, tag: str) -> list[MemoryEntry]:
        return [e for e in self._store.values() if tag in e.tags]

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        keys = list(self._order)[-n:]
        return [self._store[k] for k in reversed(keys) if k in self._store]

    def clear(self):
        self._store.clear()
        self._order.clear()

    def summary(self) -> dict:
        return {
            "capacity": self.capacity,
            "used": len(self._store),
            "keys": list(self._order),
        }
