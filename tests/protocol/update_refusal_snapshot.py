"""Capture refusal diagnostics in tests/protocol/fixtures/refusal_snapshot.json.

    HF_HUB_OFFLINE=1 uv run python tests/protocol/update_refusal_snapshot.py [--check]

The script records each trigger's exception class, code, path, and message.
Runtime triggers load tiny model fixtures. Review any changed refusal before
updating its expected diagnostic; a full capture also replaces messages for
cases whose tests use ALLOWED_UPGRADES.

--check compares a new capture with the saved file and leaves the file unchanged.
--retire-only updates RETIRED entries and preserves every captured diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from causalab.protocol.rules.errors import ProtocolError

from tests._helpers import refusal_snapshot as table

BASE = "fd61afb9"


def _record(entry_id: str, layer: str, exc: BaseException) -> dict[str, Any]:
    code = exc.code if isinstance(exc, ProtocolError) else None
    path = exc.path if isinstance(exc, ProtocolError) else None
    return {
        "id": entry_id,
        "layer": layer,
        "captured": True,
        "exc_class": type(exc).__name__,
        "code": code,
        "path": path,
        "message": str(exc),
    }


def capture() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for entry_id, trigger in table.LOAD_TRIGGERS.items():
        try:
            trigger()
        except Exception as exc:  # noqa: BLE001 - the point is to record it
            entries.append(_record(entry_id, "load", exc))
        else:
            raise AssertionError(f"load trigger {entry_id} did not refuse")
    fixtures = table.Fixtures()
    for entry_id, run_trigger in table.RUN_TRIGGERS.items():
        try:
            run_trigger(fixtures)
        except Exception as exc:  # noqa: BLE001
            entries.append(_record(entry_id, "run", exc))
        else:
            raise AssertionError(f"run trigger {entry_id} did not refuse")
    for entry_id, reason in {**table.NOT_RUNNABLE, **table.RETIRED}.items():
        entries.append(
            {"id": entry_id, "layer": "run", "captured": False, "reason": reason}
        )
    entries.sort(key=lambda entry: int(entry["id"]))
    return {"base": BASE, "entries": entries}


def retire_only() -> dict[str, Any]:
    """Update retired entries without loading models or recapturing diagnostics."""
    data = json.loads(table.SNAPSHOT.read_text())
    for entry in data["entries"]:
        reason = table.RETIRED.get(entry["id"])
        if reason is not None:
            retired = {
                "id": entry["id"],
                "layer": entry["layer"],
                "captured": False,
                "reason": reason,
            }
            entry.clear()
            entry.update(retired)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify, do not write")
    parser.add_argument(
        "--retire-only",
        action="store_true",
        help="rewrite only the RETIRED rows; every other entry stays byte-identical",
    )
    args = parser.parse_args(argv)
    data = retire_only() if args.retire_only else capture()
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        current = table.SNAPSHOT.read_text() if table.SNAPSHOT.exists() else ""
        if current != text:
            print(f"{table.SNAPSHOT} is stale", file=sys.stderr)
            return 1
        print(f"{table.SNAPSHOT} is current")
        return 0
    table.SNAPSHOT.write_text(text)
    captured = sum(1 for e in json.loads(text)["entries"] if e["captured"])
    print(f"wrote {table.SNAPSHOT} ({captured} captured refusals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
