"""JSON object graph for paused pygame state; never executes serialized code.

Keep tuple/dict keys and shared sprite references intact. Surfaces are included
because ghost drawing mutates sprite pixels and drawing advances animation.
Only the seven game-owned classes below can be reconstructed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import sys
import zlib
from typing import Any

SCHEMA = "pacman-paused-state-v1"
CLASSES = ("game", "level", "pacman", "ghost", "fruit", "path_finder", "node")
ROOTS = (
    "thisGame", "thisLevel", "player", "ghosts", "thisFruit", "path",
    "tileID", "tileIDName", "tileIDImage", "img_Background",
    "GAME_EVENT_LEDGER", "GAME_LOGIC_FRAME",
)


def checksum(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


class GameStateCodec:
    def __init__(self, pygame: Any, namespace: dict[str, Any]) -> None:
        self.pygame = pygame
        self.classes = {name: namespace[name] for name in CLASSES}

    def encode(self, value: Any) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        memo: dict[int, int] = {}

        def visit(obj: Any) -> Any:
            if obj is None or type(obj) in (bool, int, float, str):
                return obj
            if id(obj) in memo:
                return {"ref": memo[id(obj)]}
            index = len(nodes)
            memo[id(obj)] = index
            node: dict[str, Any] = {}
            nodes.append(node)
            if type(obj) in (list, tuple):
                node.update(kind=type(obj).__name__, items=[visit(x) for x in obj])
            elif type(obj) is dict:
                node.update(kind="dict", items=[[visit(k), visit(v)] for k, v in obj.items()])
            elif isinstance(obj, self.pygame.Surface):
                node.update(
                    kind="surface", size=list(obj.get_size()),
                    alpha=obj.get_alpha(), colorkey=obj.get_colorkey(),
                    per_pixel_alpha=bool(obj.get_flags() & self.pygame.SRCALPHA),
                    pixels=base64.b64encode(zlib.compress(
                        self.pygame.image.tostring(obj, "RGBA"),
                    )).decode("ascii"),
                )
            elif type(obj) in self.classes.values():
                node.update(kind="object", name=type(obj).__name__, attrs=visit(vars(obj)))
            else:
                raise ValueError(f"unsupported game state type: {type(obj).__name__}")
            return {"ref": index}

        root = visit(value)
        return {"root": root, "nodes": nodes}

    def decode(self, graph: dict[str, Any]) -> Any:
        memo: dict[int, Any] = {}

        def visit(value: Any) -> Any:
            if not isinstance(value, dict):
                if value is None or type(value) in (bool, int, float, str):
                    return value
                raise ValueError("invalid graph value")
            index = value["ref"]
            if index in memo:
                return memo[index]
            node = graph["nodes"][index]
            kind = node["kind"]
            if kind == "dict":
                obj: Any = {}
                memo[index] = obj
                obj.update((visit(k), visit(v)) for k, v in node["items"])
            elif kind == "list":
                obj = []
                memo[index] = obj
                obj.extend(visit(x) for x in node["items"])
            elif kind == "tuple":
                obj = tuple(visit(x) for x in node["items"])
            elif kind == "object":
                obj = self.classes[node["name"]].__new__(self.classes[node["name"]])
                memo[index] = obj
                attrs = visit(node["attrs"])
                if any(not isinstance(k, str) or k.startswith("__") for k in attrs):
                    raise ValueError("invalid object attributes")
                obj.__dict__.update(attrs)
            elif kind == "surface":
                raw = zlib.decompress(base64.b64decode(node["pixels"], validate=True))
                obj = self.pygame.image.fromstring(raw, node["size"], "RGBA")
                obj = obj.convert_alpha() if node["per_pixel_alpha"] else obj.convert()
                obj.set_colorkey(node["colorkey"])
                obj.set_alpha(node["alpha"])
            else:
                raise ValueError(f"invalid graph node kind: {kind}")
            memo[index] = obj
            return obj

        return visit(graph["root"])


def runtime_identity(pygame: Any) -> dict[str, Any]:
    return {"python": list(sys.version_info[:3]), "pygame": pygame.version.ver,
            "sdl": list(pygame.get_sdl_version())}


def capture_game(pygame: Any, namespace: dict[str, Any]) -> dict[str, Any]:
    roots = {name: namespace[name] for name in ROOTS}
    roots["rng"] = random.getstate()
    roots["screen"] = pygame.display.get_surface()
    return {"runtime": runtime_identity(pygame),
            "graph": GameStateCodec(pygame, namespace).encode(roots)}


def restore_game(pygame: Any, namespace: dict[str, Any], saved: dict[str, Any]) -> None:
    if saved["runtime"] != runtime_identity(pygame):
        raise ValueError("saved state runtime mismatch")
    roots = GameStateCodec(pygame, namespace).decode(saved["graph"])
    if set(roots) != {*ROOTS, "rng", "screen"}:
        raise ValueError("saved state roots mismatch")
    # Validate RNG before changing the running game. No seed replay is needed.
    random.Random().setstate(roots["rng"])
    surface = pygame.display.get_surface()
    if roots["screen"].get_size() != surface.get_size():
        raise ValueError("saved state screen size mismatch")
    namespace.update({name: roots[name] for name in ROOTS})
    random.setstate(roots["rng"])
    # Do not draw: Draw() changes animation counters and pellet blink timers.
    surface.blit(roots["screen"], (0, 0))
    namespace["agentPaused"] = True
    pygame.event.clear()
