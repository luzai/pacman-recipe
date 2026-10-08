"""Pacman image and live-state prompt construction."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
from io import BytesIO
from typing import Any, Mapping, Sequence

import numpy as np
from pacman_env.planner import validate_fallback_mode
from .vision_prompt import VISION_LAYOUT_VERSION, vision_content
from .legacy_prompts_v1 import (
    EDWARD_OPTION_CODE_V1_SYSTEM_PROMPT,
    EDWARD_OPTION_CODE_V1_USER_TEMPLATE,
)


MINIMAL_V1_SYSTEM_PROMPT = (
    "You control Pacman in the supplied game image. Collect pellets and avoid "
    "wasting moves. Respond with exactly one action token: U, D, L, R, or S. "
    "Do not explain the action."
)
MINIMAL_V1_USER_INSTRUCTION = "Choose the next action. Return exactly U, D, L, R, or S."

WALL_AVOIDANCE_V1_SYSTEM_PROMPT = (
    "You control Pacman from one screenshot. Pacman is the yellow circle and "
    "blue lines are walls. Choose one movement direction that does not hit a "
    "wall. Respond with exactly one token: U, D, L, or R. Do not explain."
)
WALL_AVOIDANCE_V1_USER_INSTRUCTION = (
    "Choose one open movement direction. Return exactly U, D, L, or R."
)

WALL_AVOIDANCE_LOCAL_V2_SYSTEM_PROMPT = (
    "You control Pacman from a local screenshot centered on Pacman. Pacman is "
    "the large yellow circle near the center and cyan lines are walls. Choose "
    "one movement direction with an open corridor immediately next to Pacman. "
    "Directions are screen-absolute: U=up, D=down, L=left, R=right. Respond "
    "with exactly one token: U, D, L, or R. Do not explain."
)
WALL_AVOIDANCE_LOCAL_V2_USER_INSTRUCTION = (
    "Inspect the walls immediately around the centered Pacman. Choose one open "
    "direction and return exactly U, D, L, or R."
)

WALL_AVOIDANCE_AXIS_V3_SYSTEM_PROMPT = (
    "You control Pacman from a local screenshot centered on Pacman. Pacman is "
    "the large yellow circle near the center and cyan lines are walls. Read the "
    "corridor through Pacman's center: if it continues horizontally, choose L "
    "or R; if it continues vertically, choose U or D. Never choose a direction "
    "that crosses a cyan wall. Directions are screen-absolute: U=up, D=down, "
    "L=left, R=right. Respond with exactly one token: U, D, L, or R. Do not "
    "explain."
)
WALL_AVOIDANCE_AXIS_V3_USER_INSTRUCTION = (
    "Use the visible corridor axis through centered Pacman to choose an open "
    "direction. Return exactly U, D, L, or R."
)

LIVE_STATIC_V2_SYSTEM_PROMPT = (
    "You are playing Classic Pacman from one current screenshot. Pacman is the "
    "yellow circle with a mouth. Blue lines are walls. Small gold dots are "
    "pellets to eat. Directions are screen-absolute: U=up, D=down, L=left, "
    "R=right. Choose a direction that is visually open and moves Pacman along "
    "a path toward a nearby pellet; never move through a blue wall. Re-evaluate "
    "from every new screenshot. Prefer U, D, L, or R; use S only if no movement "
    "direction appears safe. Respond with exactly one action token: U, D, L, R, "
    "or S. Do not explain the action."
)
LIVE_STATIC_V2_USER_INSTRUCTION = (
    "Inspect only this screenshot and choose the next action toward a reachable "
    "nearby pellet. Return exactly U, D, L, R, or S."
)

SHARED_GAME_RULES = (
    "Use only the latest screenshot and authoritative structured state; if they conflict, "
    "trust the structured state for coordinates and ghost status. Do not import rules "
    "from other Pac-Man games. Colors can vary; use shape and maze context. Pac-Man "
    "cannot cross walls or the ghost door, but can use a level/tunnel door; ghosts "
    "cannot use that tunnel. OPEN means passable, not safe from ghosts. "
    "Normal ghosts are lethal. Use the structured ghost state "
    "to determine whether a ghost is vulnerable; do not infer flashing from a single "
    "screenshot. A ghost is edible only when its structured state is vulnerable and "
    "edible_ticks allows a safe interception. edible_ticks is the remaining "
    "vulnerability duration in game logic ticks, not seconds or action count. "
    "Each movement usually consumes 16 game logic ticks. When judging the remaining "
    "edible time, leave a safety margin and account for ghosts moving. Eyes and gone "
    "ghosts are nonlethal and must not be targeted. Eating all normal pellets clears the "
    "level; power pellets are optional. The episode ends on the first death. "
)

LIVE_STATE_V3_SYSTEM_PROMPT = (
    "You control primitive movement in this Pacman simulator. "
    + SHARED_GAME_RULES
    + "Choose an OPEN direction. First avoid death, then make progress collecting "
    "normal pellets. When directions are similarly safe and useful for collection, "
    "prefer an untried exit. Reversing or repeating an exit is allowed; avoid "
    "back-and-forth movement that makes no progress. "
    "Output exactly one action letter: U, D, L, or R. "
    "Do not explain the action."
)
LIVE_STATE_V3_USER_INSTRUCTION = (
    "You are playing Classic Pacman. One current screenshot.\n"
    "Yellow circle with a mouth = Pac-Man. Blue lines = walls. Small gold "
    "coins = pellets to eat.\n"
    "Directions are screen-absolute: U=top, D=bottom, L=left, R=right."
)

EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT = (
    "You are the tactical objective policy for this Pacman simulator. "
    + SHARED_GAME_RULES
    + "Choose one reachable "
    "system-generated candidate; never invent coordinates. COLLECT is the default "
    "without a meaningful lethal threat or safe edible opportunity. Use AVOID for a "
    "threatened route, reduced escape capacity, or a trap; prefer larger safety and "
    "exits. Use ELIMINATE only for an edible ghost reachable before vulnerability "
    "expires via a nonlethal route. After selection, the navigator executes at most "
    "commit moves and rechecks safety after every move. Prioritize survival. Rows begin "
    "[code,id]. Codes map to "
    "fixed objective ids; only "
    "advertised candidates and their targets are valid this turn. Output exactly one "
    "advertised uppercase option code, not a movement action; no other text."
)
EDWARD_OBJECTIVE_V1_SYSTEM_PROMPT = (
    "You select one authoritative Edward planning objective for Classic Pacman. "
    "Use the current screenshot, authoritative game-state context, and provided "
    "candidate list. Return exactly one canonical JSON object in the form "
    '{"objective_id":"ID"}, where ID is one of the provided candidate IDs. '
    "Output no action letter, reasoning, Markdown, or other text."
)

EDWARD_OPTION_CODE_V2_USER_TEMPLATE = (
    "Choose one tactical objective. Keys: p=Pac-Man [row,column], f=facing, "
    "pellets=normal+power pellets remaining, maze=[rows,columns], "
    "ghosts=[[id,state,position]], edible_ticks=vulnerability time, "
    "last=previous action, c=candidates. Candidate row: "
    "[code,id,strategy,target,first_action,distance,commit,safety,exits,entity].\n"
    "Each candidate describes a target and its navigation plan. "
    "first_action=the first move the navigator executes if you select this candidate; "
    "screen-absolute U=up, D=down, L=left, R=right. Select its code. Metrics: distance=route steps, commit=max executed moves, larger "
    "safety/exits are better, entity=ELIMINATE ghost id.\n"
    "{decision_state}\nUse only the candidates shown "
    "for this turn. Return exactly one "
    "code from [{option_codes}]; nothing else."
)


def compact_edward_decision_prompt(
    state_context: Mapping[str, Any], candidates: Sequence[Any], constraint: Any
) -> str:
    """Render the actual bounded Edward option-code user message."""
    candidate_rows = [
        [
            constraint.code_for_option(candidate.option_id),
            candidate.option_id,
            candidate.strategy,
            list(candidate.target),
            candidate.first_action,
            candidate.route_distance,
            candidate.commit_moves,
            candidate.safety_margin,
            candidate.future_safe_exits,
            candidate.entity_id,
        ]
        for candidate in candidates
    ]
    decision_state = {
        "p": state_context.get("pacman_position"),
        "f": state_context.get("facing"),
        "pellets": state_context.get("pellets_remaining"),
        "maze": state_context.get("maze_size"),
        "ghosts": [
            [ghost.get("id"), ghost.get("state"), ghost.get("position")]
            for ghost in state_context.get("ghosts") or []
            if isinstance(ghost, Mapping)
        ],
        "edible_ticks": state_context.get("edible_ticks"),
        "last": state_context.get("last_action"),
        "c": candidate_rows,
    }
    return EDWARD_OPTION_CODE_V2_USER_TEMPLATE.format(
        decision_state=json.dumps(
            decision_state, separators=(",", ":"), allow_nan=False
        ),
        option_codes=",".join(constraint.rendered_choices),
    )


EDWARD_RISK_NOTICE = (
    " RISK_FALLBACK is not safety-approved. Only when normal C/A/E options are "
    "absent, choose from ALL open directions ranked A0..A3 by increasing estimated "
    "risk. Execute one move, then replan. Unknown motion does not prove safety."
)
EDWARD_RISK_USER_SUFFIX = EDWARD_RISK_NOTICE + (
    "\nFallback risk by code: {risk_state}\n"
    "Risk row=[rank,motion,ghost_clearance,route_margin,safe_next_cells,dead_end,reverse]. "
    "motion=clear_estimate/unknown/collision_predicted for 16 frames, ignoring "
    "power-pellet effects. Estimates, not probabilities. Larger clearance/margin "
    "are better; safe_next_cells counts adjacent tiles, not guaranteed escapes. "
    "Rank 1 is lowest estimated risk. Return one advertised code."
)


def edward_system_prompt(fallback_mode: str = "refuse") -> str:
    """Keep normal requests unchanged; risk instructions appear only on fallback."""
    validate_fallback_mode(fallback_mode)
    return EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT


def risk_ranked_edward_decision_prompt(
    state_context: Mapping[str, Any], candidates: Sequence[Any], constraint: Any
) -> str:
    """Separate renderer keeps archived v1 source fingerprints unchanged."""
    risk_fields = (
        "rank", "motion", "ghost_clearance", "route_margin", "safe_next_cells",
        "dead_end", "reverse",
    )
    risks = {
        constraint.code_for_option(candidate.option_id): [
            candidate.risk[field] for field in risk_fields
        ]
        for candidate in candidates
        if candidate.strategy == "RISK_FALLBACK"
    }
    base = compact_edward_decision_prompt(state_context, candidates, constraint)
    if not risks:
        return base
    return base + (
        EDWARD_RISK_USER_SUFFIX.format(
            risk_state=json.dumps(risks, separators=(",", ":"), allow_nan=False)
        )
    )


def render_edward_decision_prompt(
    state_context: Mapping[str, Any], candidates: Sequence[Any], constraint: Any,
    *, fallback_mode: str = "refuse",
) -> str:
    validate_fallback_mode(fallback_mode)
    renderer = (
        risk_ranked_edward_decision_prompt if fallback_mode == "risk_ranked"
        else compact_edward_decision_prompt
    )
    return renderer(state_context, candidates, constraint)


PROMPT_STYLES = (
    "minimal_v1",
    "wall_avoidance_v1",
    "wall_avoidance_local_v2",
    "wall_avoidance_axis_v3",
    "live_static_v2",
    "live_state_v3",
)
SYSTEM_PROMPT = MINIMAL_V1_SYSTEM_PROMPT
USER_INSTRUCTION = MINIMAL_V1_USER_INSTRUCTION


def prompt_text(prompt_style: str) -> tuple[str, str]:
    if prompt_style == "ascii_edward_v1":
        return ASCII_EDWARD_SYSTEM_PROMPT, ASCII_EDWARD_USER_TEMPLATE
    if prompt_style == "minimal_v1":
        return MINIMAL_V1_SYSTEM_PROMPT, MINIMAL_V1_USER_INSTRUCTION
    if prompt_style == "wall_avoidance_v1":
        return WALL_AVOIDANCE_V1_SYSTEM_PROMPT, WALL_AVOIDANCE_V1_USER_INSTRUCTION
    if prompt_style == "wall_avoidance_local_v2":
        return (
            WALL_AVOIDANCE_LOCAL_V2_SYSTEM_PROMPT,
            WALL_AVOIDANCE_LOCAL_V2_USER_INSTRUCTION,
        )
    if prompt_style == "wall_avoidance_axis_v3":
        return (
            WALL_AVOIDANCE_AXIS_V3_SYSTEM_PROMPT,
            WALL_AVOIDANCE_AXIS_V3_USER_INSTRUCTION,
        )
    if prompt_style == "live_static_v2":
        return LIVE_STATIC_V2_SYSTEM_PROMPT, LIVE_STATIC_V2_USER_INSTRUCTION
    if prompt_style == "live_state_v3":
        return LIVE_STATE_V3_SYSTEM_PROMPT, LIVE_STATE_V3_USER_INSTRUCTION
    raise ValueError(
        f"unsupported image prompt style {prompt_style!r}; "
        f"expected one of {PROMPT_STYLES}"
    )


def _action_text(actions: Sequence[Any]) -> str:
    tokens = [str(action) for action in actions]
    return ", ".join(tokens) if tokens else "NONE"


def live_state_instruction(context: Mapping[str, Any]) -> str:
    """Build the bounded dynamic block used by the live-demo-style policy."""
    position = context.get("pacman_position")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes))
        or len(position) != 2
    ):
        raise ValueError("live-state prompt requires a two-item pacman_position")
    row, col = (int(position[0]), int(position[1]))
    facing = str(context.get("facing") or "S")
    pellets_remaining = int(context.get("pellets_remaining", -1))
    open_actions = list(context.get("open_actions") or [])
    blocked_actions = list(context.get("blocked_actions") or [])
    exits = list(context.get("current_cell_exit_history") or [])
    last_action = context.get("last_action")

    history_text = _action_text(exits)
    pellets_line = (
        f" - Remaining pellets: {pellets_remaining}\n" if pellets_remaining >= 0 else ""
    )
    last_action_line = f"Last move: {last_action}.\n" if last_action else ""
    ghost_line = (
        " - Ghosts [id,state,position]: "
        + json.dumps(
            [
                [ghost.get("id"), ghost.get("state"), ghost.get("position")]
                for ghost in context.get("ghosts") or []
            ],
            separators=(",", ":"),
            allow_nan=False,
        )
        + f"; edible_ticks={int(context.get('edible_ticks', 0))}\n"
        if "ghosts" in context
        else ""
    )
    return (
        f"{LIVE_STATE_V3_USER_INSTRUCTION}\n\n"
        "GAME STATE (authoritative from Pacman engine):\n"
        f" - Grid position: row={row}, col={col}\n"
        f" - Facing: {facing}\n"
        f"{pellets_line}"
        f"{ghost_line}"
        f" - BLOCKED dirs here: {_action_text(blocked_actions)}\n"
        f" - OPEN dirs here: {_action_text(open_actions)}\n"
        f"At this cell ({row},{col}), directions already taken before: "
        f"{history_text}.\n"
        f"{last_action_line}\n"
        f"Choose ONE ACTION from [{_action_text(open_actions)}]."
    )


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_user_template(
    prompt_style: str, *, edward_options: bool, fallback_mode: str = "refuse"
) -> str:
    validate_fallback_mode(fallback_mode, edward_options=edward_options)
    if prompt_style == "ascii_edward_v1":
        if not edward_options:
            raise ValueError("ASCII Edward requires Edward options")
        return ASCII_EDWARD_USER_TEMPLATE + (EDWARD_RISK_USER_SUFFIX if fallback_mode == "risk_ranked" else "")
    return (
        VISION_EDWARD_USER_TEMPLATE + (
            EDWARD_RISK_USER_SUFFIX if fallback_mode == "risk_ranked" else ""
        )
        if edward_options
        else prompt_text(prompt_style)[1]
    )


def prompt_contract_metadata(
    prompt_style: str, *, edward_options: bool, fallback_mode: str = "refuse"
) -> dict[str, str]:
    """Fingerprint the actual static text AND dynamic rendering implementation.

    This is a template fingerprint, not a claim that per-turn prompts are equal.
    Trajectories separately fingerprint the exact rendered messages and image.
    """
    if prompt_style == "ascii_edward_v1":
        if not edward_options:
            raise ValueError("ASCII Edward requires Edward options")
        return ascii_edward_prompt_contract_metadata(fallback_mode=fallback_mode)
    system, _ = prompt_text(prompt_style)
    if edward_options:
        system = edward_system_prompt(fallback_mode)
        renderer = (
            risk_ranked_edward_decision_prompt if fallback_mode == "risk_ranked"
            else compact_edward_decision_prompt
        )
        protocol = "edward-option-code-v1"
        version = "edward-option-code-v2"
    else:
        renderer = (
            live_state_instruction if prompt_style == "live_state_v3" else prompt_text
        )
        protocol = "direct-open-action-token-v1"
        version = (
            "live-state-direct-action-v3"
            if prompt_style == "live_state_v3"
            else prompt_style
        )
    template = prompt_user_template(
        prompt_style, edward_options=edward_options, fallback_mode=fallback_mode
    )
    fingerprint = {
        "system": system,
        "user_template": template,
        "renderer_source": inspect.getsource(renderer).replace("\r\n", "\n"),
        "vision_layout": VISION_LAYOUT_VERSION,
        "vision_layout_source": inspect.getsource(layout_image_user_content),
        "vision_content_source": inspect.getsource(vision_content),
        "vision_edward_renderer_source": inspect.getsource(render_vision_edward_decision_prompt) if edward_options else None,
    }
    if fallback_mode == "risk_ranked":
        fingerprint["base_renderer_source"] = inspect.getsource(
            compact_edward_decision_prompt
        ).replace("\r\n", "\n")
    return {
        "action_protocol": protocol,
        "prompt_version": f"{version}+{VISION_LAYOUT_VERSION}",
        "prompt_template_sha256": text_sha256(
            json.dumps(fingerprint, sort_keys=True, allow_nan=False)
        ),
        "system_prompt_sha256": text_sha256(system),
        "user_prompt_template_sha256": text_sha256(template),
    }


def sent_prompt_sha256(system: str, user: str, image_sha256: str) -> str:
    """Canonical identity of the actual two text messages and attached image."""
    return text_sha256(
        json.dumps(
            {"system": system, "user": user, "observation_png_sha256": image_sha256,
             "vision_layout": VISION_LAYOUT_VERSION},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def encode_png(image: np.ndarray) -> bytes:
    """Encode one Pacman RGB observation as deterministic PNG bytes."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image-only rollouts") from exc
    buffer = BytesIO()
    # Level 1 (user decision 2026-10-02): lossless like 9, ~2.6 ms instead of ~14 ms per 400x336 frame
    # (9 KB vs 5 KB). These bytes stay in process; slime re-encodes for SGLang. The level changes PNG bytes
    # and their sha256, not pixels. legacy_prompts_v1 keeps level 9 for frozen evidence.
    Image.fromarray(image, mode="RGB").save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=1,
    )
    return buffer.getvalue()


def crop_pacman_local_view(
    image: np.ndarray,
    *,
    radius: int = 48,
    upscale: int = 3,
) -> np.ndarray:
    """Find Pacman from yellow pixels and return a centered local RGB view."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")
    if radius < 8 or upscale < 1:
        raise ValueError("invalid local-view dimensions")

    # Pacman and pellets are pure yellow. Pacman is the largest connected
    # yellow component; orange ghosts are excluded by the strict green cutoff.
    yellow = (image[:, :, 0] > 240) & (image[:, :, 1] > 220) & (image[:, :, 2] < 40)
    # Ignore the small yellow score/lives glyphs at the very bottom.
    yellow[max(0, image.shape[0] - 20) :, :] = False
    seen = np.zeros(yellow.shape, dtype=bool)
    largest: list[tuple[int, int]] = []
    height, width = yellow.shape
    for row, col in zip(*np.where(yellow), strict=True):
        row = int(row)
        col = int(col)
        if seen[row, col]:
            continue
        component: list[tuple[int, int]] = []
        stack = [(row, col)]
        seen[row, col] = True
        while stack:
            current_row, current_col = stack.pop()
            component.append((current_row, current_col))
            for row_delta in (-1, 0, 1):
                for col_delta in (-1, 0, 1):
                    neighbor_row = current_row + row_delta
                    neighbor_col = current_col + col_delta
                    if (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_col < width
                        and yellow[neighbor_row, neighbor_col]
                        and not seen[neighbor_row, neighbor_col]
                    ):
                        seen[neighbor_row, neighbor_col] = True
                        stack.append((neighbor_row, neighbor_col))
        if len(component) > len(largest):
            largest = component
    if len(largest) < 100:
        raise ValueError("could not locate Pacman in RGB observation")

    center_row = int(round(sum(row for row, _ in largest) / len(largest)))
    center_col = int(round(sum(col for _, col in largest) / len(largest)))
    side = radius * 2
    local = np.zeros((side, side, 3), dtype=np.uint8)
    source_top = max(0, center_row - radius)
    source_bottom = min(height, center_row + radius)
    source_left = max(0, center_col - radius)
    source_right = min(width, center_col + radius)
    target_top = source_top - (center_row - radius)
    target_left = source_left - (center_col - radius)
    local[
        target_top : target_top + source_bottom - source_top,
        target_left : target_left + source_right - source_left,
    ] = image[source_top:source_bottom, source_left:source_right]
    if upscale > 1:
        local = np.repeat(np.repeat(local, upscale, axis=0), upscale, axis=1)
    return local


def png_sha256(png: bytes) -> str:
    return hashlib.sha256(png).hexdigest()


def png_data_url(png: bytes) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


VISION_EDWARD_FIXED_PREFIX = (
    EDWARD_OPTION_CODE_V2_USER_TEMPLATE.split("{decision_state}", 1)[0]
    + "Coordinates are [row,column], starting at 0. If the screenshot and "
    "structured state disagree, trust the structured state.\n\n[CURRENT IMAGE]\n"
)
VISION_EDWARD_USER_TEMPLATE = (
    VISION_EDWARD_FIXED_PREFIX
    + "[CURRENT STATE]\n{decision_state}\n\n"
    "[CANDIDATE OBJECTIVES]\n{candidate_rows}\n\n"
    "[OUTPUT]\nUse only the candidates shown for this turn. Return exactly one "
    "code from [{option_codes}]; nothing else."
)


def render_vision_edward_decision_prompt(
    state_context, candidates, constraint, *, fallback_mode="refuse"
):
    """Preserve decision facts and fallback evidence in separate visual sections."""
    base = render_edward_decision_prompt(
        state_context, candidates, constraint, fallback_mode=fallback_mode
    )
    prefix = EDWARD_OPTION_CODE_V2_USER_TEMPLATE.split("{decision_state}", 1)[0]
    serialized_state, tail = base[len(prefix):].split("\n", 1)
    state = json.loads(serialized_state)
    rows = state.pop("c")
    footer = EDWARD_OPTION_CODE_V2_USER_TEMPLATE.split("{decision_state}\n", 1)[1].format(
        option_codes=",".join(constraint.rendered_choices)
    )
    if not tail.startswith(footer):
        raise ValueError("Edward output instructions changed")
    return VISION_EDWARD_USER_TEMPLATE.format(
        decision_state=json.dumps(state, separators=(",", ":"), allow_nan=False),
        candidate_rows="\n".join(json.dumps(row, separators=(",", ":"), allow_nan=False) for row in rows),
        option_codes=",".join(constraint.rendered_choices),
    ) + tail[len(footer):]


def layout_image_user_content(png, instruction, *, prompt_style, edward_options=False):
    """Split only known invariant prefixes; preserve exact concatenated text."""
    fixed = ""
    if edward_options:
        fixed = VISION_EDWARD_FIXED_PREFIX
    elif prompt_style == "live_state_v3":
        fixed = LIVE_STATE_V3_USER_INSTRUCTION + "\n\n"
    else:
        # These primitive styles contain only invariant instructions.
        fixed = instruction
    if fixed and not instruction.startswith(fixed):
        raise ValueError("vision instruction does not match its fixed prefix")
    return vision_content(
        {"type": "image_url", "image_url": {"url": png_data_url(png)}},
        fixed_text=fixed, dynamic_text=instruction[len(fixed):])


def build_image_messages(
    png: bytes,
    *,
    prompt_style: str = "minimal_v1",
    state_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the two-message contract containing exactly one image."""
    system_prompt, user_instruction = prompt_text(prompt_style)
    if prompt_style == "live_state_v3":
        if state_context is None:
            raise ValueError("live_state_v3 requires state_context")
        user_instruction = live_state_instruction(state_context)
    elif state_context is not None:
        raise ValueError(f"{prompt_style} does not accept state_context")
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": layout_image_user_content(
                png, user_instruction, prompt_style=prompt_style),
        },
    ]


def image_count(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            count += sum(
                1
                for item in content
                if isinstance(item, dict) and item.get("type") == "image_url"
            )
    return count

# Separate ASCII identity: archived image constants and renderers remain unchanged.
ASCII_EDWARD_SYSTEM_PROMPT = EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT.replace(
    "Use only the latest screenshot and authoritative structured state; if they conflict, "
    "trust the structured state for coordinates and ghost status.",
    "The board is an ASCII map of the current frame. Legend: # wall, . pellet, "
    "o power pellet, P Pac-Man, G normal ghost, V vulnerable ghost, E ghost eyes, "
    "- ghost door, = tunnel/level door, space empty. Map row r and column c equal "
    "[r,c] in the state JSON. If the map and the state JSON ever disagree, trust the state JSON.",
).replace("Colors can vary; use shape and maze context. ", "").replace(
    "Use the structured ghost state to determine whether a ghost is vulnerable; "
    "do not infer flashing from a single screenshot.",
    "Use the structured ghost state to determine whether a ghost is vulnerable.",
)
ASCII_EDWARD_LAYOUT_VERSION = "fixed-map-dynamic-v1"
# Keep the same fixed-instructions-first topology as VLM. Only the observation
# terminology differs; no padding or cache-specific words enter either prompt.
ASCII_EDWARD_FIXED_PREFIX = VISION_EDWARD_FIXED_PREFIX.replace(
    "screenshot", "map"
).replace("[CURRENT IMAGE]", "[CURRENT MAP]")
ASCII_EDWARD_USER_TEMPLATE = (
    ASCII_EDWARD_FIXED_PREFIX
    + "{ascii_map}\n\n"
    "[CURRENT STATE]\n{decision_state}\n\n"
    "[CANDIDATE OBJECTIVES]\n{candidate_rows}\n\n"
    "[OUTPUT]\n"
    "Use only the candidates shown for this turn. Return exactly one "
    "code from [{option_codes}]; nothing else."
)


def ascii_edward_system_prompt(fallback_mode: str = "refuse") -> str:
    validate_fallback_mode(fallback_mode)
    return ASCII_EDWARD_SYSTEM_PROMPT


def compact_ascii_edward_decision_prompt(
    state_context: Mapping[str, Any], candidates: Sequence[Any], constraint: Any,
    map_text: str, *, fallback_mode: str = "refuse",
) -> str:
    """Separate map, state and candidate rows without changing their evidence."""
    from .ascii_observation import ASCII_MAP_HEADER
    if not map_text.startswith(ASCII_MAP_HEADER + "\n"):
        raise ValueError("expected rendered current-frame ASCII map")
    base = render_edward_decision_prompt(
        state_context, candidates, constraint, fallback_mode=fallback_mode
    )
    prefix = EDWARD_OPTION_CODE_V2_USER_TEMPLATE.split("{decision_state}", 1)[0]
    if not base.startswith(prefix):
        raise ValueError("Edward template prefix changed")
    serialized_state, tail = base[len(prefix):].split("\n", 1)
    state = json.loads(serialized_state)
    candidate_rows = state.pop("c")
    footer = EDWARD_OPTION_CODE_V2_USER_TEMPLATE.split("{decision_state}\n", 1)[1].format(
        option_codes=",".join(constraint.rendered_choices)
    )
    if not tail.startswith(footer):
        raise ValueError("Edward output instructions changed")
    rendered = ASCII_EDWARD_USER_TEMPLATE.format(
        ascii_map=map_text,
        decision_state=json.dumps(state, separators=(",", ":"), allow_nan=False),
        candidate_rows="\n".join(
            json.dumps(row, separators=(",", ":"), allow_nan=False)
            for row in candidate_rows
        ),
        option_codes=",".join(constraint.rendered_choices),
    )
    return rendered + tail[len(footer):]


def ascii_edward_prompt_contract_metadata(*, fallback_mode: str = "refuse") -> dict[str, str]:
    from . import ascii_observation
    template = prompt_user_template("ascii_edward_v1", edward_options=True, fallback_mode=fallback_mode)
    fingerprint = {
        "system": ascii_edward_system_prompt(fallback_mode), "user_template": template,
        "ascii_renderer_module": inspect.getsource(ascii_observation).replace("\r\n", "\n"),
        "renderer_sources": [inspect.getsource(fn).replace("\r\n", "\n") for fn in (
            compact_ascii_edward_decision_prompt, render_edward_decision_prompt,
            compact_edward_decision_prompt, risk_ranked_edward_decision_prompt,
        )],
    }
    return {
        "action_protocol": "edward-option-code-v1",
        "prompt_version": f"edward-ascii-option-code-v2+{ASCII_EDWARD_LAYOUT_VERSION}",
        "prompt_template_sha256": text_sha256(json.dumps(fingerprint, sort_keys=True, allow_nan=False)),
        "system_prompt_sha256": text_sha256(ASCII_EDWARD_SYSTEM_PROMPT),
        "user_prompt_template_sha256": text_sha256(template),
    }
