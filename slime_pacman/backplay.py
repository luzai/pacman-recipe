"""Bind immutable full simulator restarts to the options rollout contract."""

from copy import deepcopy
from pathlib import Path

from pacman_recipe.level1.backplay import load_restart_bank, load_restart_state
from pacman_recipe.level1.contracts import validate_episode_record


def bind_restart_record(template, bank_dir, state_id):
    validate_episode_record(template)
    bank = load_restart_bank(bank_dir)
    expected_environment = {key: template["environment"][key]
                            for key in ("ghost_mode", "episode_life_mode", "max_steps")}
    if bank.get("environment") != expected_environment:
        raise ValueError("restart bank environment is absent or differs from learner")
    entries = [e for e in bank["restart_states"] if e["restart_state_id"] == state_id]
    if len(entries) != 1:
        raise ValueError("restart ID must identify exactly one bank entry")
    entry = entries[0]
    saved = load_restart_state(bank_dir, entry)
    identity = saved["payload"]["identity"]
    for key in ("max_steps", "episode_life_mode"):
        if identity[key] != template["environment"][key]:
            raise ValueError(f"restart {key} differs from the learner contract")
    record = deepcopy(template)
    record["id"] = state_id
    record["environment"]["seed"] = entry["seed"]
    record["restart"] = dict(
        path=str((Path(bank_dir) / entry["state_path"]).resolve()),
        sha256=entry["state_file_sha256"], id=state_id,
    )
    validate_episode_record(record)
    return record
