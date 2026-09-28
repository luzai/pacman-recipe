from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from pathlib import Path
from typing import Iterable


SUITE_NAME = "multi_maze_v1"
SPLIT_SIZES = {"train": 64, "validation": 16, "test": 32}
SUITE_PATH = Path(__file__).with_name("maze_suites") / f"{SUITE_NAME}.json"


def layout_hash(layout: Iterable[str]) -> str:
    payload = "\n".join(layout).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def topology_hash(layout: Iterable[str]) -> str:
    topology = ("".join("#" if cell == "#" else " " for cell in row) for row in layout)
    return layout_hash(topology)


def _validate_structure(layout: tuple[str, ...]) -> None:
    if not layout or len({len(row) for row in layout}) != 1:
        raise ValueError("layout must be a non-empty rectangle")
    if any(cell != "#" for cell in layout[0] + layout[-1]):
        raise ValueError("layout top and bottom borders must be walls")
    if any(row[0] != "#" or row[-1] != "#" for row in layout):
        raise ValueError("layout left and right borders must be walls")
    joined = "".join(layout)
    if joined.count("P") != 1 or joined.count("G") != 1:
        raise ValueError("layout must contain exactly one PacMan and one ghost")
    if joined.count(".") != 5:
        raise ValueError("multi_maze_v1 layouts must contain exactly five pellets")
    unknown = set(joined) - {"#", " ", ".", "P", "G"}
    if unknown:
        raise ValueError(f"layout contains unsupported cells: {sorted(unknown)}")

    open_cells = {
        (row, col)
        for row, cells in enumerate(layout)
        for col, cell in enumerate(cells)
        if cell != "#"
    }
    start = next(iter(open_cells))
    queue = deque([start])
    reached = {start}
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (row + dr, col + dc)
            if nxt in open_cells and nxt not in reached:
                reached.add(nxt)
                queue.append(nxt)
    if reached != open_cells:
        raise ValueError("all non-wall cells must be connected")


def load_suite(path: Path = SUITE_PATH) -> dict[str, object]:
    if not path.exists():
        return {"name": SUITE_NAME, "splits": {split: [] for split in SPLIT_SIZES}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("name") != SUITE_NAME:
        raise ValueError(f"expected maze suite {SUITE_NAME!r}")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("maze suite must contain a splits object")

    seen_names: set[str] = set()
    seen_hashes: set[str] = set()
    seen_topologies: set[str] = set()
    for split, expected_size in SPLIT_SIZES.items():
        records = splits.get(split)
        if not isinstance(records, list) or len(records) != expected_size:
            raise ValueError(f"maze split {split!r} must contain {expected_size} layouts")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("maze records must be objects")
            name = str(record.get("name", ""))
            layout = tuple(record.get("layout") or ())
            digest = str(record.get("layout_hash", ""))
            topology_digest = str(record.get("topology_hash") or topology_hash(layout))
            record["topology_hash"] = topology_digest
            _validate_structure(layout)
            if record.get("split") != split:
                raise ValueError(f"maze {name!r} has the wrong split label")
            if digest != layout_hash(layout):
                raise ValueError(f"maze {name!r} has an invalid layout hash")
            if topology_digest != topology_hash(layout):
                raise ValueError(f"maze {name!r} has an invalid topology hash")
            if name in seen_names or digest in seen_hashes or topology_digest in seen_topologies:
                raise ValueError(f"duplicate maze name, layout hash, or topology hash: {name!r}")
            if not record.get("oracle_won") or int(record.get("oracle_steps", 0)) <= 0:
                raise ValueError(f"maze {name!r} is missing a successful oracle gate")
            seen_names.add(name)
            seen_hashes.add(digest)
            seen_topologies.add(topology_digest)
    return payload


def maze_records(split: str | None = None) -> list[dict[str, object]]:
    suite = load_suite()
    splits = suite["splits"]
    if split is not None:
        if split not in SPLIT_SIZES:
            raise ValueError(f"unknown maze split {split!r}; expected one of {sorted(SPLIT_SIZES)}")
        return list(splits[split])
    return [record for name in SPLIT_SIZES for record in splits[name]]


def split_layout_names(split: str) -> tuple[str, ...]:
    return tuple(str(record["name"]) for record in maze_records(split))


def load_layout_registry() -> dict[str, tuple[str, ...]]:
    return {
        str(record["name"]): tuple(str(row) for row in record["layout"])
        for record in maze_records()
    }


def suite_summary() -> dict[str, object]:
    records = maze_records()
    return {
        "name": SUITE_NAME,
        "split_sizes": Counter(str(record["split"]) for record in records),
        "layout_count": len(records),
        "unique_layout_hashes": len({str(record["layout_hash"]) for record in records}),
        "unique_topology_hashes": len({str(record["topology_hash"]) for record in records}),
        "oracle_steps_min": min(int(record["oracle_steps"]) for record in records),
        "oracle_steps_max": max(int(record["oracle_steps"]) for record in records),
    }
