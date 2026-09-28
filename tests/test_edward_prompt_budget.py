import sys

import pytest

from scripts.level1.train import check_edward_prompt_budget as budget
from pacman_recipe.level1.prompts import EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT
from test_level1_recipe import FakeObjectiveTokenizer


def test_budget_default_keeps_legacy_prompt_and_all_ten_options():
    messages, user = budget.worst_case_messages(FakeObjectiveTokenizer())
    assert messages[0]["content"] == EDWARD_OPTION_CODE_V2_SYSTEM_PROMPT
    assert "RISK_FALLBACK" not in user
    assert '"C3"' in user and '"A3"' in user and '"E1"' in user


def test_budget_includes_four_risk_directions():
    messages, user = budget.worst_case_messages(
        FakeObjectiveTokenizer(), fallback_mode="risk_ranked", risk_fallback=True,
    )
    assert "not safety-approved" in user
    assert user.count('"RISK_FALLBACK"') == 4
    assert "collision_predicted" in user
    assert '"C0"' not in user and '"E0"' not in user


@pytest.mark.parametrize("mode,scenarios", [("refuse", [False]), ("risk_ranked", [False, True])])
def test_budget_cli_reads_mode_from_config(tmp_path, monkeypatch, mode, scenarios):
    config = tmp_path / "recipe.yaml"
    config.write_text(f"edward_options: true\nedward_fallback_mode: {mode}\n")
    monkeypatch.setattr(sys, "argv", ["budget", "--config", str(config), "--model-path", str(tmp_path)])
    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", lambda *a, **kw: object())
    calls = []

    def measure(_processor, *, fallback_mode, risk_fallback=False):
        assert fallback_mode == mode
        calls.append(risk_fallback)
        return {"input_tokens": 1000}

    monkeypatch.setattr(budget, "measure", measure)
    budget.main()
    assert calls == scenarios


def test_budget_cli_refuses_overflow_in_fallback_only(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["budget", "--fallback-mode", "risk_ranked", "--model-path", str(tmp_path)])
    monkeypatch.setattr("transformers.AutoProcessor.from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(budget, "measure", lambda _p, **kw: {"input_tokens": 1024 if kw.get("risk_fallback") else 1000})
    with pytest.raises(SystemExit, match="risk_fallback worst-case"):
        budget.main()
