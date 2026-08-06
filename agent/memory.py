"""
Persistent memory utilities for the Amazon Review Agent.

Features
--------
- Persistent JSON storage
- Exact and fuzzy key lookup
- Automatic timestamps
- List merging
- Search
- Recent memories
- Memory summary
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from difflib import get_close_matches
from typing import Any

from . import config


# internal helpers


def _load() -> dict:
    if not os.path.exists(config.MEMORY_PATH):
        return {}

    with open(config.MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(config.MEMORY_PATH), exist_ok=True)

    with open(config.MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _timestamp() -> str:
    return datetime.utcnow().isoformat()


# basic API


def all() -> dict:
    return _load()


def keys() -> list[str]:
    return list(_load().keys())


def exists(key: str) -> bool:
    return key in _load()


def delete(key: str) -> bool:
    data = _load()

    if key not in data:
        return False

    del data[key]
    _save(data)
    return True


# reading


def get(key: str, default=None):
    """
    Exact lookup.

    Backward compatible with existing code.
    """
    return _load().get(key, default)


def search(query: str):
    """
    Search memory using fuzzy matching.

    Example:
        search("recent order")
        search("orders")
        search("cart")
    """

    data = _load()

    if not data:
        return None

    # exact key
    if query in data:
        return {query: data[query]}

    # partial match
    partial = {
        k: v
        for k, v in data.items()
        if query.lower() in k.lower()
    }

    if partial:
        return partial

    # fuzzy match
    matches = get_close_matches(
        query,
        list(data.keys()),
        n=5,
        cutoff=0.45,
    )

    if matches:
        return {k: data[k] for k in matches}

    return None


# writing


def set(key: str, value: Any) -> None:
    """
    Backward compatible overwrite.
    """

    data = _load()

    data[key] = {
        "value": value,
        "updated_at": _timestamp(),
    }

    _save(data)


def append(key: str, value: Any) -> None:
    """
    Append to a list instead of overwriting.
    """

    data = _load()

    if key not in data:
        data[key] = {
            "value": [],
            "updated_at": _timestamp(),
        }

    if not isinstance(data[key]["value"], list):
        data[key]["value"] = [data[key]["value"]]

    data[key]["value"].append(value)
    data[key]["updated_at"] = _timestamp()

    _save(data)


def merge(key: str, value: Any) -> None:
    """
    Merge dictionaries or append unique list items.
    """

    data = _load()

    if key not in data:
        set(key, value)
        return

    current = data[key]["value"]

    if isinstance(current, dict) and isinstance(value, dict):
        current.update(value)

    elif isinstance(current, list):

        if isinstance(value, list):
            for item in value:
                if item not in current:
                    current.append(item)

        else:
            if value not in current:
                current.append(value)

    else:
        current = value

    data[key]["value"] = current
    data[key]["updated_at"] = _timestamp()

    _save(data)


# agent helpers


def recent(limit: int = 5) -> dict:
    """
    Return the most recently updated memories.
    """

    data = _load()

    ordered = sorted(
        data.items(),
        key=lambda x: x[1].get("updated_at", ""),
        reverse=True,
    )

    return dict(ordered[:limit])


def summary() -> str:
    """
    Human-readable summary of memory.
    """

    data = _load()

    if not data:
        return "Memory is empty."

    lines = []

    for key, value in data.items():

        v = value.get("value")

        if isinstance(v, list):
            desc = f"{len(v)} item(s)"

        elif isinstance(v, dict):
            desc = f"{len(v)} field(s)"

        else:
            desc = str(v)

        lines.append(f"- {key}: {desc}")

    return "\n".join(lines)
