"""Dataset for the dynamic-bank curriculum: true-start templates of the training seeds.

Rows only seed slime's data source (indices) and provide the episode-record template; the curriculum
replaces each group's start record. Provenance (config, source identities, bank, file hashes and
restart binding) is checked by preflight like the other dataset schemas.

Fixed-states mode (--selected-from) rebinds the restart states of an earlier, already validated
dataset under the current sources, e.g. to rerun an overfit experiment on new code. The source
manifest is copied in as selection evidence; its finalist probes are not revalidated, since they
were collected with the earlier sources.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from pacman_recipe.level1.backplay import load_restart_bank
from pacman_recipe.level1.contracts import make_episode_record, validate_episode_record, write_json_new
from .backplay import bind_restart_record
from .config import load_config

SCHEMA = "pacman-dynamic-bank-dataset-v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def true_start_ids(bank, seeds):
    ids = {}
    for seed in seeds:
        matches = [e["restart_state_id"] for e in bank["restart_states"]
                   if e["seed"] == seed and e["is_true_initial_state"]]
        if len(matches) != 1:
            raise ValueError(f"bank needs exactly one true start for seed {seed}")
        ids[seed] = matches[0]
    return ids


def fixed_state_ids(bank, source_manifest):
    if source_manifest.get("bank_id") != bank["bank_id"]:
        raise ValueError("selected-from dataset uses a different teacher bank")
    known = {e["restart_state_id"] for e in bank["restart_states"]}
    ids = list(source_manifest["state_ids"])
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= known:
        raise ValueError("selected-from state ids are not distinct teacher-bank states")
    return ids


def prepare(args):
    config = load_config(args.config)
    template = make_episode_record(0, split="train", recipe_root=args.recipe, game_root=args.game,
                                   backend_root=args.slime, max_steps=config.max_steps)
    bank = load_restart_bank(args.bank)
    selected_from = getattr(args, "selected_from", None)
    if selected_from is not None:
        if args.train_seed:
            raise ValueError("--selected-from and --train-seed are exclusive")
        source = json.loads(Path(selected_from).read_text(encoding="utf-8"))
        state_ids = fixed_state_ids(bank, source)
    else:
        ids = true_start_ids(bank, args.train_seed)
        state_ids = [ids[seed] for seed in args.train_seed]
    rows = [bind_restart_record(template, args.bank, state_id) for state_id in state_ids]
    validation = deepcopy(rows[:1])
    validation[0]["split"] = "validation"
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=SCHEMA, training_backend="slime", config=config.as_dict(),
                    source_revisions=template["source_revisions"], bank_path=str(args.bank.resolve()),
                    bank_id=bank["bank_id"], train_seeds=list(args.train_seed or []), state_ids=state_ids,
                    files={})
    if selected_from is not None:
        # Recorded only in this mode, so true-start manifests stay byte-identical.
        evidence = args.output / "selected-from.json"
        evidence.write_bytes(Path(selected_from).read_bytes())
        manifest["selection"] = "fixed_states"
        manifest["selected_from"] = dict(file=evidence.name, sha256=_sha(evidence), path=str(selected_from))
    for split, records in (("train", rows), ("validation", validation)):
        path = args.output / f"{split}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as out:
            for record in records:
                out.write(json.dumps(dict(prompt="Pacman episode", metadata=dict(episode_record=record)),
                                     sort_keys=True) + "\n")
        manifest["files"][path.name] = dict(rows=len(records), sha256=_sha(path))
    write_json_new(args.output / "manifest.json", manifest)
    return check_dataset(args.output, config, template["source_revisions"])


def check_dataset(directory, config, sources):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("training_backend") != "slime":
        raise ValueError("invalid dynamic-bank dataset manifest")
    if manifest["config"] != config.as_dict() or manifest["source_revisions"] != sources:
        raise ValueError("dynamic-bank dataset config/source changed; generate a fresh dataset")
    bank = load_restart_bank(manifest["bank_path"])
    if bank["bank_id"] != manifest["bank_id"]:
        raise ValueError("teacher bank changed")
    if manifest.get("selection", "true_starts") == "fixed_states":
        evidence = directory / manifest["selected_from"]["file"]
        if evidence.parent.resolve() != directory.resolve() or _sha(evidence) != manifest["selected_from"]["sha256"]:
            raise ValueError("selected-from evidence checksum/path changed")
        source = json.loads(evidence.read_text(encoding="utf-8"))
        if manifest["train_seeds"] or fixed_state_ids(bank, source) != manifest["state_ids"]:
            raise ValueError("fixed-state membership changed")
    else:
        ids = true_start_ids(bank, manifest["train_seeds"])
        if [ids[s] for s in manifest["train_seeds"]] != manifest["state_ids"]:
            raise ValueError("true-start membership changed")
    if set(manifest["files"]) != {"train.jsonl", "validation.jsonl"}:
        raise ValueError("unexpected dataset splits")
    for split, expected in (("train", manifest["state_ids"]), ("validation", manifest["state_ids"][:1])):
        path = directory / f"{split}.jsonl"
        evidence = manifest["files"][path.name]
        if _sha(path) != evidence["sha256"]:
            raise ValueError("dynamic-bank dataset checksum changed")
        rows = [json.loads(line)["metadata"]["episode_record"] for line in path.read_text().splitlines()]
        if len(rows) != evidence["rows"] or [row["id"] for row in rows] != expected:
            raise ValueError("dynamic-bank dataset membership changed")
        for row in rows:
            validate_episode_record(row, expected_sources=sources)
            if row["split"] != split or row["environment"]["max_steps"] != config.max_steps:
                raise ValueError("dynamic-bank dataset split/horizon changed")
            template = dict(row, split="train")
            rebound = bind_restart_record(template, manifest["bank_path"], row["id"])
            if dict(rebound, split=split) != row:
                raise ValueError("dynamic-bank restart binding differs")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output", "bank", "recipe", "game", "slime"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--train-seed", type=int, action="append", default=[])
    parser.add_argument("--selected-from", type=Path,
                        help="manifest of an earlier dataset whose restart states are rebound (fixed-states mode)")
    print(json.dumps(prepare(parser.parse_args()), indent=1))


if __name__ == "__main__":
    main()
