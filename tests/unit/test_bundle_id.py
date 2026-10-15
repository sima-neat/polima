"""Bundle identity.

Content-addressing is what makes `polima deploy` idempotent and rollback an
`ln -sfn`, so determinism and sensitivity both matter.
"""

from __future__ import annotations

import pytest

from polima.bundle.layout import (
    Bundle,
    GraphArtifact,
    compute_bundle_id,
    digest_graphs,
    parse_bundle_id,
    slug,
)

DIGESTS = {
    "vision_backbone": "a" * 64,
    "encoder_layer_00_stem": "b" * 64,
    "decoder_action_tail": "c" * 64,
}
PLAN = {"buffers": {"x": 1}, "steps": [{"op": "run_elf", "out": "x"}], "result": "x"}


def base_id(**overrides):
    kwargs = {"policy": "act", "dataset": "rcwb_f_t", "steps": 100000,
              "graph_digests": DIGESTS, "plan": PLAN}
    kwargs.update(overrides)
    return compute_bundle_id(**kwargs)


def test_deterministic():
    assert base_id() == base_id()


def test_insensitive_to_graph_ordering():
    shuffled = dict(reversed(list(DIGESTS.items())))
    assert base_id() == base_id(graph_digests=shuffled)


def test_sensitive_to_elf_content():
    changed = {**DIGESTS, "vision_backbone": "d" * 64}
    assert base_id() != base_id(graph_digests=changed)


def test_sensitive_to_plan():
    other = {**PLAN, "result": "y"}
    assert base_id() != base_id(plan=other)


def test_sensitive_to_graph_set():
    fewer = {k: v for k, v in DIGESTS.items() if k != "decoder_action_tail"}
    assert base_id() != base_id(graph_digests=fewer)


def test_shape_and_round_trip():
    bundle_id = base_id()
    assert bundle_id.startswith("act-rcwb_f_t-100000-")
    parsed = parse_bundle_id(bundle_id)
    assert parsed["policy"] == "act"
    assert parsed["dataset"] == "rcwb_f_t"
    assert parsed["steps"] == 100000
    assert len(parsed["sha"]) == 8


@pytest.mark.parametrize("bad", ["nope", "act-x", "act-x-notanumber-abcdef12", ""])
def test_parse_rejects_junk(bad):
    with pytest.raises(ValueError):
        parse_bundle_id(bad)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("rcwb f/t", "rcwb_f_t"), ("a..b", "a..b"), ("///", "unknown"), ("Red Cube!", "Red_Cube")],
)
def test_slug(raw, expected):
    assert slug(raw) == expected


def test_digest_graphs_reads_real_files(tmp_path):
    one = tmp_path / "one.elf"
    one.write_bytes(b"hello")
    two = tmp_path / "two.elf"
    two.write_bytes(b"hello")
    three = tmp_path / "three.elf"
    three.write_bytes(b"different")

    digests = digest_graphs({"one": one, "two": two, "three": three})
    assert digests["one"] == digests["two"]        # identical content
    assert digests["one"] != digests["three"]


def test_model_elf_path_is_the_single_convention(tmp_path):
    bundle = Bundle(
        root=tmp_path, policy="act", bundle_id=base_id(),
        graphs=[GraphArtifact(
            name="vision_backbone",
            elf="models/vision_backbone/share/vision_backbone_stage1_mla.elf",
            sha256="a" * 64, elf_bytes=1234,
        )],
    )
    assert bundle.model_elf("vision_backbone") == (
        tmp_path / "models/vision_backbone/share/vision_backbone_stage1_mla.elf"
    )
    # Unknown graphs still resolve to the convention rather than raising, so an
    # incomplete manifest fails at open() with a readable path.
    assert bundle.model_elf("encoder_layer_01").name == "encoder_layer_01_stage1_mla.elf"


def test_manifest_round_trip(tmp_path):
    bundle = Bundle(
        root=tmp_path, policy="act", bundle_id=base_id(), checkpoint="/ckpt/100000",
        graphs=[GraphArtifact("g", "models/g/share/g_stage1_mla.elf", "f" * 64, 10)],
        sidecars=["constants/normalization.json"],
        tool_versions={"afe": "2.1.0"},
    )
    restored = Bundle.from_dict(bundle.to_dict(), tmp_path)
    assert restored.to_dict() == bundle.to_dict()
