"""Shared rules and survival priority in the active policy prompts."""
from types import SimpleNamespace

from pacman_recipe.level1.prompts import (
    SHARED_GAME_RULES,
    LIVE_STATE_V3_SYSTEM_PROMPT,
    EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT,
    EDWARD_OPTION_CODE_V2_USER_TEMPLATE,
    compact_edward_decision_prompt,
    live_state_instruction,
)


def test_active_policies_share_rules_but_keep_distinct_output_contracts():
    for prompt in (LIVE_STATE_V3_SYSTEM_PROMPT, EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT):
        assert prompt.count(SHARED_GAME_RULES) == 1
        for rule in ("latest screenshot", "structured state", "ghost door",
                     "OPEN means passable, not safe from ghosts",
                     "Normal ghosts are lethal", "edible_ticks", "Eyes and gone",
                     "Each movement usually consumes 16 game logic ticks",
                     "leave a safety margin and account for ghosts moving",
                     "power pellets are optional", "episode ends on the first death"):
            assert rule in prompt
    assert "one action letter: U, D, L, or R" in LIVE_STATE_V3_SYSTEM_PROMPT
    assert "COLLECT" not in LIVE_STATE_V3_SYSTEM_PROMPT
    assert "commit" not in LIVE_STATE_V3_SYSTEM_PROMPT
    assert "advertised uppercase option code" in EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT


def test_primitive_can_reverse_or_repeat_for_safety():
    text = live_state_instruction({
        "pacman_position": [1, 2], "open_actions": ["L", "R"],
        "blocked_actions": ["U", "D"], "preferred_open_actions": ["R"],
        "current_cell_exit_history": ["L"], "last_action": "R",
        "episode_life_mode": "single_death",
        "ghosts": [{"id": 1, "state": "vulnerable", "position": [1, 4]}],
        "edible_ticks": 3,
    })
    assert "Choose ONE ACTION from [L, R]" in text
    assert "History preference" not in text
    assert "directions already taken before: L" in text
    assert "Last move: R." in text
    assert "Reversing or repeating an exit is allowed" in LIVE_STATE_V3_SYSTEM_PROMPT
    assert "similarly safe and useful for collection" in LIVE_STATE_V3_SYSTEM_PROMPT
    assert "Do NOT reverse" not in text
    assert "life_mode" not in text
    assert '[1,"vulnerable",[1,4]]' in text
    assert "edible_ticks=3" in text


def test_options_explain_first_move_in_user_prompt_and_omit_life_mode():
    assert "first_action" not in EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT
    assert "first move the navigator executes if you select this candidate" in EDWARD_OPTION_CODE_V2_USER_TEMPLATE
    assert "screen-absolute U=up, D=down, L=left, R=right" in EDWARD_OPTION_CODE_V2_USER_TEMPLATE
    assert "game logic ticks, not seconds or action count" in SHARED_GAME_RULES
    text = compact_edward_decision_prompt(
        {"episode_life_mode": "single_death"}, [],
        SimpleNamespace(rendered_choices=[]),
    )
    assert "life_mode" not in text
    for prompt in (LIVE_STATE_V3_SYSTEM_PROMPT, EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT):
        assert "life_mode" not in prompt
        assert "original_three_lives" not in prompt


def test_history_preference_does_not_change_primitive_prompt():
    context = {"pacman_position": [1, 2], "open_actions": ["L", "R"]}
    assert live_state_instruction(context) == live_state_instruction(
        {**context, "preferred_open_actions": ["R"]}
    )
