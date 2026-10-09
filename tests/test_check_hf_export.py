"""check_hf_export: catches float32 config leaks, stale shard indexes, wrong image resolution."""
import json
import struct

from scripts.level1.report.check_hf_export import check, main


def write_safetensors(path, dtypes):
    header = {name: {"dtype": dtype, "shape": [1], "data_offsets": [2 * i, 2 * i + 2]} for i, (name, dtype) in enumerate(dtypes.items())}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * (2 * len(dtypes)))


def export(tmp_path, *, dtype="bfloat16", tensor="BF16", edge=537600, index=False, drop=()):
    ck = tmp_path / "ck"; ck.mkdir()
    (ck / "config.json").write_text(json.dumps({"dtype": dtype, "text_config": {"dtype": dtype, "mamba_ssm_dtype": "float32"}}))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (ck / name).write_text("{}")
    (ck / "preprocessor_config.json").write_text(json.dumps({"size": {"shortest_edge": edge, "longest_edge": 16777216}}))
    write_safetensors(ck / "model.safetensors", {"a.weight": tensor, "b.weight": "BF16"})
    if index:
        (ck / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a.weight": "model-00001-of-00004.safetensors"}}))
    for name in drop:
        (ck / name).unlink()
    return ck


def test_clean_export_passes_and_mamba_state_dtype_is_allowed(tmp_path):
    ck = export(tmp_path)
    assert check(ck) == []
    assert main([str(ck)]) == 0


def test_each_known_failure_is_reported(tmp_path):
    cases = {
        "float32 config": dict(dtype="float32"),
        "fp32 tensor": dict(tensor="F32"),
        "stale preprocessor": dict(edge=65536),
        "stale base index": dict(index=True),
        "missing template": dict(drop=("chat_template.jinja",)),
    }
    for i, (label, kwargs) in enumerate(cases.items()):
        root = tmp_path / str(i); root.mkdir()
        problems = check(export(root, **kwargs))
        assert problems, label
        assert main([str(root / "ck")]) == 1
