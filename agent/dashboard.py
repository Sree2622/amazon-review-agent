import json
import os
from datetime import datetime, timezone
from . import config


def write_entry(entry: dict) -> None:
    os.makedirs(os.path.dirname(config.DASHBOARD_PATH), exist_ok=True)
    rows = []
    if os.path.exists(config.DASHBOARD_PATH):
        with open(config.DASHBOARD_PATH, "r", encoding="utf-8") as f:
            rows = json.load(f)
    entry = {**entry, "timestamp": datetime.now(timezone.utc).isoformat()}
    rows.append(entry)
    with open(config.DASHBOARD_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def all_entries() -> list:
    if not os.path.exists(config.DASHBOARD_PATH):
        return []
    with open(config.DASHBOARD_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
