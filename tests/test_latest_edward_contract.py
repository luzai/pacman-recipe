"""Default selection and route-summary prompt contracts, without GPU dependencies."""
from types import SimpleNamespace
import pytest
from slime_pacman.config import PacmanConfig, runner_options
from pacman_recipe.level1.prompts import (
    render_edward_decision_prompt, render_vision_edward_decision_prompt,
)


def test_default_and_explicit_legacy_config():
    config = PacmanConfig()
    options = runner_options(config)
    assert options["edward_harness"] == "latest"
    assert options["edward_fallback_mode"] == "risk_ranked"
    assert runner_options(PacmanConfig(edward_harness="legacy"))["edward_harness"] == "legacy"
    with pytest.raises(ValueError, match="edward_harness"):
        PacmanConfig(edward_harness="unknown")


def test_routes_are_distinguishable_and_shared_with_audit_renderer():
    state = {"pacman_position": [1, 2], "route_menu": {
        "C0": {"route_id": "abc", "junctions": [[1, 3, "U"]]},
        "C1": {"route_id": "def", "junctions": [[1, 3, "D"]]},
    }}
    constraint = SimpleNamespace(rendered_choices=["B", "C"])
    compact = render_edward_decision_prompt(state, [], constraint)
    visual = render_vision_edward_decision_prompt(state, [], constraint)
    marker = "\nBound routes (coordinates row,col): "
    assert compact[compact.index(marker):] == visual[visual.index(marker):]
    assert visual.count(marker) == 1
    assert '"abc"' in visual and '"def"' in visual
    assert "does not guarantee safety" in visual
    assert marker not in render_vision_edward_decision_prompt({}, [], constraint)
