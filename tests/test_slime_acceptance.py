import copy
import math
import json
import sys

import pytest
import torch

from slime_pacman.acceptance import compare_probabilities, probe_sequence_isolation


def record(probabilities=(0.25, 0.75)):
    return {
        "fixture_sha256": "a" * 64,
        "weights_sha256": "b" * 64,
        "processor_sha256": "c" * 64,
        "image_sha256": "d" * 64,
        "input_ids": [1, 2],
        "image_grid_thw": [[1, 2, 2]],
        "allowed_token_ids": [4, 7],
        "temperature": 0.7,
        "outside_support_probability": 0,
        "log_probs": [math.log(x) for x in probabilities],
    }


def compare(a, b):
    return compare_probabilities(a, b, max_abs_logp=0.01, max_kl=0.001)


def test_probability_metrics_detect_shift():
    assert compare(record(), record())["passed"]
    result = compare(record(), record((0.5, 0.5)))
    assert not result["passed"]
    assert result["kl_reference_candidate"] == pytest.approx(0.13081203594)
    assert result["ratio_outside_clip_fraction"] == 1
    assert result["max_probability_difference"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "field",
    [
        "weights_sha256",
        "input_ids",
        "allowed_token_ids",
        "processor_sha256",
        "temperature",
        "image_sha256",
    ],
)
def test_identity_mismatch_is_not_numerical_noise(field):
    candidate = copy.deepcopy(record())
    candidate[field] = None
    with pytest.raises(ValueError, match="identity"):
        compare(record(), candidate)


@pytest.mark.parametrize(
    "update",
    [
        {"log_probs": [0.0, 0.0]},
        {"log_probs": [float("nan"), -1.0]},
        {"log_probs": [-1.0]},
        {"outside_support_probability": 0.01},
    ],
)
def test_invalid_probabilities_fail_closed(update):
    candidate = record()
    candidate.update(update)
    with pytest.raises(ValueError):
        compare(record(), candidate)


class RecurrentToy(torch.nn.Module):
    def __init__(self, isolate):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7))
        self.isolate = isolate

    def forward(self, x, cu_seqlens):
        if not self.isolate:
            return x.cumsum(1) * self.weight
        return (
            torch.cat(
                [
                    x[:, start:end].cumsum(1)
                    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
                ],
                dim=1,
            )
            * self.weight
        )


@pytest.mark.parametrize("isolate", [True, False])
def test_probe_detects_cross_sequence_recurrence(isolate):
    module = RecurrentToy(isolate).eval()
    a = torch.arange(6.0).reshape(1, 3, 2)
    b = torch.ones(1, 5, 2)
    result = probe_sequence_isolation(module, a, b, atol=1e-6, rtol=1e-6)
    assert result["passed"] is isolate
    if not isolate:
        assert not result["cases"]["b_a"]["zero_cross_sample_gradient"]
    assert module.weight.grad is None


def test_probe_rejects_training_mode():
    with pytest.raises(ValueError, match="eval mode"):
        probe_sequence_isolation(
            RecurrentToy(True), torch.ones(1, 2, 2), torch.ones(1, 3, 2), atol=0, rtol=0
        )


def test_extreme_finite_logp_does_not_make_kl_nan():
    left, right = record(), record()
    left["log_probs"], right["log_probs"] = [0.0, -1000.0], [0.0, -1001.0]
    result = compare(left, right)
    assert math.isfinite(result["kl_reference_candidate"])
    assert not result["passed"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_ids", []),
        ("image_grid_thw", [[0, 1, 1]]),
        ("weights_sha256", "version-0"),
    ],
)
def test_matching_invalid_identity_still_fails(field, value):
    a = record()
    a[field] = value
    with pytest.raises(ValueError):
        compare(a, a)


@pytest.mark.parametrize("shifted", [False, True])
def test_cli_records_failure_and_refuses_overwrite(tmp_path, monkeypatch, shifted):
    from slime_pacman.compare_evidence import main

    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    limits = tmp_path / "limits.json"
    output = tmp_path / "result.json"
    reference.write_text(json.dumps(record()))
    candidate.write_text(json.dumps(record((0.5, 0.5)) if shifted else record()))
    limits.write_text(
        json.dumps(
            {
                "max_abs_logp": 0.01,
                "max_kl": 0.001,
                "calibration_evidence": "test-only-calibration",
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--tolerances",
            str(limits),
            "--output",
            str(output),
        ],
    )
    if shifted:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
    else:
        main()
    result = json.loads(output.read_text())
    assert result["passed"] is not shifted
    assert len(result["input_sha256"]["candidate"]) == 64
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        main()
    assert output.read_bytes() == before
