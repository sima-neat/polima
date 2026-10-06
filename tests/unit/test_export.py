"""The export path's torch-free logic.

The torch-dependent part is covered by reproduction instead: re-exporting the
checkpoint that produced the deployed bundle regenerates all six ONNX graphs,
all six calibration npz, and every fixture byte for byte, with the verify report
matching the recorded 1.430511474609375e-06 exactly.

Left here is what can go wrong without torch being involved: entry-point
resolution, dataset resolution from the checkpoint, and the two indirections
LeRobot puts between a checkpoint and its normalization statistics.
"""

from __future__ import annotations

import json

import pytest

from polima.export import normalization
from polima.export import samples as sampling
from polima.export.driver import CONTRACT_FILE, ExportResult, _write_contract, resolve
from polima.policies.registry import get_policy


# ------------------------------------------------------------ entry points


def test_resolve_finds_a_dotted_attribute():
    assert resolve("json:dumps") is json.dumps


def test_resolve_requires_the_colon_form():
    with pytest.raises(ValueError, match="module:attribute"):
        resolve("polima.export.driver")


def test_resolve_reports_a_missing_attribute():
    with pytest.raises(AttributeError):
        resolve("json:not_a_function")


def test_every_act_entry_point_is_a_colon_path():
    """A typo here fails at export time, minutes into a run, so it is worth
    asserting the shape up front."""
    plan = get_policy("act").compile
    entries = [plan.export_entry, plan.verify_entry, plan.fixture_entry,
               plan.normalization_entry]
    for entry in entries:
        assert entry and ":" in entry, entry


# ------------------------------------------------------- dataset resolution


def _checkpoint(tmp_path, root="/ml_datasets/rcwb_f_t", repo_id="local/rcwb_f_t"):
    (tmp_path / "train_config.json").write_text(
        json.dumps({"dataset": {"repo_id": repo_id, "root": root}})
    )
    return tmp_path


def test_dataset_comes_from_the_checkpoint(tmp_path):
    """So calibration cannot silently use a different dataset than training."""
    dataset = tmp_path / "ml_datasets" / "rcwb_f_t"
    repo_id, root = sampling.resolve_dataset(_checkpoint(tmp_path, root=str(dataset)))
    assert repo_id == "local/rcwb_f_t"
    # Resolved, so a symlinked dataset parent names the real directory.
    assert root == dataset.resolve()


def test_dataset_root_can_be_overridden(tmp_path):
    """Datasets get moved; the repo_id still comes from the checkpoint."""
    repo_id, root = sampling.resolve_dataset(_checkpoint(tmp_path), tmp_path / "elsewhere")
    assert repo_id == "local/rcwb_f_t"
    assert root == tmp_path / "elsewhere"


def test_missing_train_config_suggests_the_flag(tmp_path):
    with pytest.raises(FileNotFoundError, match="--dataset-root"):
        sampling.resolve_dataset(tmp_path)


# --------------------------------------------------------- normalization io


def _processor(tmp_path, name, registry, state_file):
    (tmp_path / name).write_text(json.dumps({"steps": [
        {"registry_name": "other_processor", "state_file": "unused.safetensors"},
        {"registry_name": registry, "state_file": state_file},
    ]}))


def test_state_file_follows_the_registry_indirection(tmp_path):
    _processor(tmp_path, normalization.PREPROCESSOR, "normalizer_processor", "pre.safetensors")
    found = normalization._state_file(
        tmp_path, normalization.PREPROCESSOR, "normalizer_processor"
    )
    assert found == tmp_path / "pre.safetensors"


def test_missing_manifest_names_the_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="pretrained_model"):
        normalization._state_file(tmp_path, normalization.PREPROCESSOR, "normalizer_processor")


def test_absent_step_lists_what_was_found(tmp_path):
    _processor(tmp_path, normalization.PREPROCESSOR, "normalizer_processor", "pre.safetensors")
    with pytest.raises(KeyError) as caught:
        normalization._state_file(tmp_path, normalization.PREPROCESSOR, "unnormalizer_processor")
    assert "other_processor" in str(caught.value)


# ---------------------------------------------------------------- contract


def test_contract_records_camera_order(tmp_path):
    """The board addresses cameras by slot, so a swapped order produces a policy
    that runs perfectly and reaches for the wrong place."""
    spec = get_policy("act")
    written = [tmp_path / f"{g.name}.onnx" for g in spec.compile.graphs]
    _write_contract(spec, tmp_path, tmp_path / "ckpt", tmp_path / "data",
                    ["observation.images.overhead", "observation.images.wrist"], written)
    contract = json.loads((tmp_path / CONTRACT_FILE).read_text())
    assert contract["camera_order"] == [
        "observation.images.overhead", "observation.images.wrist"
    ]
    assert contract["policy"] == "act"
    assert contract["graphs"] == [f"{g.name}.onnx" for g in spec.compile.graphs]


def test_contract_shapes_come_from_the_spec(tmp_path):
    spec = get_policy("act")
    written = [tmp_path / f"{g.name}.onnx" for g in spec.compile.graphs]
    _write_contract(spec, tmp_path, tmp_path / "ckpt", tmp_path / "data", ["a", "b"], written)
    shapes = json.loads((tmp_path / CONTRACT_FILE).read_text())["fixed_shapes"]
    assert shapes["image"] == [1, 3, 480, 640]
    assert shapes["stem_input"] == [1, 1, 601, 512]
    assert shapes["normalized_actions"] == [1, 1, 100, 16]


# ------------------------------------------------------------------ result


def test_export_result_requires_a_passing_verification():
    assert not ExportResult("b", graphs=["a"]).ok
    assert not ExportResult("b", graphs=["a"], verification={"ok": False}).ok
    assert not ExportResult("b", graphs=[], verification={"ok": True}).ok
    assert ExportResult("b", graphs=["a"], verification={"ok": True}).ok


def test_fixture_file_is_named_by_the_spec():
    """The generic export driver must carry no policy-specific filename; ACT
    keeps the legacy name because build trees on disk already use it."""
    assert get_policy("act").compile.fixture_file == "act_fixture.npz"
