"""GR00T grouped sin/cos state encoding: reference opcode and checkpoint contract."""

import numpy as np
import pytest

from polima.wire.server_stub import PlanError, StubPlan


def plan(widths, output=128):
    return StubPlan(buffers={"state": 29, "lane": output},
                    steps=[{"op": "sincos_state", "out": "lane",
                            "args": {"src": "state", "group_widths": widths}}],
                    result="lane", wire={})


def test_grouped_state_encoding_preserves_checkpoint_order_and_padding():
    raw = np.linspace(-3, 4, 29, dtype=np.float32)
    groups = np.split(raw, [7, 14, 20, 26])
    expected = np.concatenate([np.concatenate([np.sin(g), np.cos(g)]) for g in groups])
    output = plan([7, 7, 6, 6, 3]).run({"state": raw})
    np.testing.assert_array_equal(output[:58], expected)
    np.testing.assert_array_equal(output[58:], 0)


@pytest.mark.parametrize("widths,output", [([], 128), ([29, 0], 128),
                                         ([28], 128), ([30], 128), ([29], 57)])
def test_grouped_state_encoding_rejects_wrong_geometry(widths, output):
    with pytest.raises(PlanError, match="shape mismatch"):
        plan(widths, output).run({"state": np.zeros(29, np.float32)})


@pytest.mark.parametrize("enabled,keys,expected", [
    (True, ["arm", "hand"], {"kind": "grouped_sincos", "group_widths": [2, 1]}),
    (False, ["arm", "hand"], None),
    (True, ["arm"], "reject"),
])
def test_checkpoint_state_contract_needs_no_image_cache(tmp_path, enabled, keys, expected):
    import json

    from polima.policies.groot.profile import checkpoint_state_encoding

    processor = {"processor_kwargs": {"apply_sincos_state_encoding": enabled,
        "modality_configs": {"gr1": {"state": {
            "modality_keys": ["arm", "hand"], "sin_cos_embedding_keys": keys}}}}}
    (tmp_path / "processor_config.json").write_text(json.dumps(processor))
    (tmp_path / "statistics.json").write_text(json.dumps({"gr1": {"state": {
        "arm": {"mean": [0, 0]}, "hand": {"mean": [0]}}}}))
    if expected == "reject":
        with pytest.raises(ValueError, match="mixed affine/sincos"):
            checkpoint_state_encoding(tmp_path, "gr1")
    else:
        assert checkpoint_state_encoding(tmp_path, "gr1") == expected
