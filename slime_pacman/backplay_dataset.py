"""One-update dataset with explicit restart membership and probe provenance."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from pacman_recipe.level1.backplay import load_restart_bank
from pacman_recipe.level1.contracts import (make_episode_record, repository_identity,
                                            validate_episode_record, write_json_new)
from .backplay import bind_restart_record
from .config import load_config, runner_options

SCHEMA = "pacman-backplay-smoke-dataset-v1"
SELECTION_SUCCESS_RANGE = (0.1, 0.9)
SELECTION_MODES = ("smoke", "curriculum")
PREPARATION_ONLY_SOURCE_FILES = {"slime_pacman/backplay_dataset.py", "slime_pacman/preflight.py"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_provenance(template, config, bank_id, server_runtime_sha256):
    return dict(source_revisions=template["source_revisions"], prompt=template["prompt"],
                rollout_config=dict(runner_options(config), max_input_tokens=config.max_input_tokens),
                bank_id=bank_id, server_runtime_sha256=server_runtime_sha256)


def _source_files(root):
    root = Path(root).resolve()
    files = {}
    for name in ("pacman_recipe", "pacman_env", "slime_pacman", "slime", "slime_plugins",
                 "pacman", "configs", "scripts"):
        directory = root / name
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix in {".py", ".pyw", ".json", ".yaml", ".txt", ".sh"}:
                    files[path.relative_to(root).as_posix()] = hashlib.sha256(
                        path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    for path in root.iterdir():
        if path.is_file() and path.suffix in {".py", ".toml"}:
            files[path.name] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return files


def _check_probe_recipe(current_sources, probe_sources, current_recipe, probe_recipe):
    for name in ("pacman-python", "slime"):
        if current_sources[name] != probe_sources[name]:
            raise ValueError(f"probe {name} source differs from training")
    if current_sources["pacman-recipe"] == probe_sources["pacman-recipe"]:
        return None
    if probe_recipe is None:
        raise ValueError("changed recipe requires the frozen probe recipe path")
    probe_recipe = Path(probe_recipe).resolve()
    if repository_identity(probe_recipe) != probe_sources["pacman-recipe"]:
        raise ValueError("frozen probe recipe identity differs from finalist evidence")
    if current_sources["pacman-recipe"]["commit"] != probe_sources["pacman-recipe"]["commit"]:
        raise ValueError("probe and training recipe base commits differ")
    current, frozen = _source_files(current_recipe), _source_files(probe_recipe)
    differences = {name for name in current.keys() | frozen.keys()
                   if current.get(name) != frozen.get(name)}
    if "slime_pacman/backplay_dataset.py" not in differences or not differences <= PREPARATION_ONLY_SOURCE_FILES:
        raise ValueError(f"probe and training recipe source differs outside selection policy: {sorted(differences)}")
    return str(probe_recipe)


def validate_selection(bank_dir, state_ids, reports, weight_version, expected_provenance, allow_true_initial=False):
    if len(state_ids) != 4 or len(set(state_ids)) != 4:
        raise ValueError("smoke requires four distinct restart states")
    bank = load_restart_bank(bank_dir)
    if expected_provenance["bank_id"] != bank["bank_id"]:
        raise ValueError("probe provenance bank differs")
    entries = {entry["restart_state_id"]: entry for entry in bank["restart_states"]}
    selected = [entries[state_id] for state_id in state_ids]
    # Curriculum stages whose frontier has reached the start train from the true initial state.
    if not allow_true_initial and any(entry["is_true_initial_state"] for entry in selected):
        raise ValueError("backplay smoke requires noninitial restart states")
    trajectory_ids = {entry["trajectory_id"] for entry in selected}
    if len(trajectory_ids) != 4:
        raise ValueError("smoke requires four distinct teacher trajectories")
    routes = set()
    root = Path(bank_dir).resolve()
    for trajectory in bank["trajectories"]:
        if trajectory["trajectory_id"] in trajectory_ids:
            path = (root / trajectory["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("teacher path escapes bank")
            payload = json.loads(path.read_text())
            routes.add(tuple(step["action"] for step in payload["actions"]))
    if len(routes) != 4:
        raise ValueError("teacher action routes are duplicated")
    summaries = {}
    for report in reports:
        if report.get("provenance") != expected_provenance:
            raise ValueError("finalist source/prompt/rollout provenance differs")
        if report.get("purpose") != "finalist" or report.get("training_data") is not False:
            raise ValueError("selection requires independent finalist evaluation")
        if report.get("weight_version") != weight_version or report.get("optimizer_updates") != 0:
            raise ValueError("selection policy version differs or was updated")
        for summary in report["summaries"]:
            state_id = summary["state_id"]
            if state_id in summaries:
                raise ValueError("duplicate finalist state evidence")
            summaries[state_id] = summary
    for state_id in state_ids:
        summary = summaries[state_id]
        n, wins = summary["samples"], summary["successes"]
        low, high = SELECTION_SUCCESS_RANGE
        if type(n) is not int or type(wins) is not int or n < 24 or not low <= wins / n <= high:
            raise ValueError("finalist requires >=24 samples and 10–90% success point estimate")
        if abs(summary["success_rate"] - wins / n) > 1e-12:
            raise ValueError("finalist success accounting differs")
    return bank


def check_dataset(directory, config, sources):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("training_backend") != "slime":
        raise ValueError("invalid backplay smoke manifest")
    if config.updates != 1 or manifest["config"] != config.as_dict() or manifest["source_revisions"] != sources:
        raise ValueError("backplay smoke config/source changed or updates is not one")
    if manifest.get("validation_scope") != "reused_restart_infrastructure_only":
        raise ValueError("missing smoke validation scope")
    if manifest.get("selection_success_range") != list(SELECTION_SUCCESS_RANGE):
        raise ValueError("smoke selection policy differs")
    reports = []
    for evidence in manifest["finalist_reports"]:
        path = directory / evidence["file"]
        if path.parent.resolve() != directory.resolve() or _sha(path) != evidence["sha256"]:
            raise ValueError("finalist evidence checksum/path changed")
        reports.append(json.loads(path.read_text()))
    from pacman_recipe.level1.prompts import prompt_contract_metadata
    current_provenance = probe_provenance(dict(source_revisions=sources,
        prompt=prompt_contract_metadata("live_state_v3", edward_options=True)), config, manifest["bank_id"],
        manifest["server_runtime_sha256"])
    provenance = manifest["probe_provenance"]
    for key in ("prompt", "rollout_config", "bank_id", "server_runtime_sha256"):
        if provenance[key] != current_provenance[key]:
            raise ValueError(f"probe {key} differs from training")
    _check_probe_recipe(sources, provenance["source_revisions"], Path(__file__).resolve().parents[1],
                        manifest.get("probe_recipe_path"))
    selection_mode = manifest.get("selection_mode", "smoke")
    if selection_mode not in SELECTION_MODES:
        raise ValueError("unknown selection mode")
    bank = validate_selection(manifest["bank_path"], manifest["state_ids"], reports, manifest["weight_version"], provenance,
                              allow_true_initial=selection_mode == "curriculum")
    if bank["bank_id"] != manifest["bank_id"]:
        raise ValueError("restart bank changed")
    if set(manifest["files"]) != {"train.jsonl", "validation.jsonl"}:
        raise ValueError("unexpected smoke dataset splits")
    for split, ids in (("train", manifest["state_ids"]), ("validation", manifest["state_ids"][:1])):
        path = directory / f"{split}.jsonl"
        evidence = manifest["files"][path.name]
        if _sha(path) != evidence["sha256"]:
            raise ValueError("smoke dataset checksum changed")
        rows = [json.loads(line)["metadata"]["episode_record"] for line in path.read_text().splitlines()]
        if len(rows) != evidence["rows"] or [row["id"] for row in rows] != ids:
            raise ValueError("smoke restart membership changed")
        for row in rows:
            validate_episode_record(row, expected_sources=sources)
            if row["split"] != split or row["environment"]["max_steps"] != config.max_steps:
                raise ValueError("smoke split/horizon changed")
            if bind_restart_record(row, manifest["bank_path"], row["id"]) != row:
                raise ValueError("smoke restart binding differs")
    return manifest


def prepare(args):
    config = load_config(args.config)
    if config.updates != 1:
        raise ValueError("backplay smoke requires updates=1")
    reports = [json.loads(path.read_text()) for path in args.finalist_report]
    template = make_episode_record(0, split="train", recipe_root=args.recipe, game_root=args.game,
                                   backend_root=args.slime, max_steps=config.max_steps)
    bank = load_restart_bank(args.bank)
    current_provenance = probe_provenance(template, config, bank["bank_id"], _sha(args.server_manifest))
    provenance = reports[0]["provenance"]
    for key in ("prompt", "rollout_config", "bank_id", "server_runtime_sha256"):
        if provenance[key] != current_provenance[key]:
            raise ValueError(f"probe {key} differs from training")
    probe_recipe_path = _check_probe_recipe(template["source_revisions"], provenance["source_revisions"],
                                            args.recipe, getattr(args, "probe_recipe", None))
    allow_true_initial = getattr(args, "allow_true_initial", False)
    bank = validate_selection(args.bank, args.candidate_id, reports, args.weight_version, provenance,
                              allow_true_initial=allow_true_initial)
    rows = [bind_restart_record(template, args.bank, state_id) for state_id in args.candidate_id]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=SCHEMA, training_backend="slime", config=config.as_dict(),
                    source_revisions=template["source_revisions"], bank_path=str(args.bank.resolve()),
                    bank_id=bank["bank_id"], state_ids=args.candidate_id, weight_version=args.weight_version,
                    probe_provenance=provenance, probe_recipe_path=probe_recipe_path,
                    selection_success_range=list(SELECTION_SUCCESS_RANGE),
                    server_runtime_sha256=provenance["server_runtime_sha256"],
                    validation_scope="reused_restart_infrastructure_only", finalist_reports=[], files={})
    if allow_true_initial:
        # Recorded only when set, so manifests of existing smoke datasets stay byte-identical.
        manifest["selection_mode"] = "curriculum"
    for i, report in enumerate(reports):
        path = args.output / f"finalist-{i}.json"
        write_json_new(path, report)
        manifest["finalist_reports"].append(dict(file=path.name, sha256=_sha(path)))
    validation = deepcopy(rows[:1])
    validation[0]["split"] = "validation"
    for split, records in (("train", rows), ("validation", validation)):
        path = args.output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as out:
            for record in records:
                out.write(json.dumps(dict(prompt="Pacman episode", metadata=dict(episode_record=record)), sort_keys=True) + "\n")
        manifest["files"][path.name] = dict(rows=len(records), sha256=_sha(path))
    write_json_new(args.output / "manifest.json", manifest)
    return check_dataset(args.output, config, template["source_revisions"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output", "bank", "recipe", "game", "slime", "server-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--probe-recipe", type=Path,
                        help="frozen finalist recipe when only selection policy changed")
    parser.add_argument("--candidate-id", action="append", required=True)
    parser.add_argument("--finalist-report", action="append", type=Path, required=True)
    parser.add_argument("--weight-version", required=True)
    parser.add_argument("--allow-true-initial", action="store_true",
                        help="curriculum stage: selected states may include true initial states")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
