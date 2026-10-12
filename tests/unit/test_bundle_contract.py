from __future__ import annotations

import json
from types import SimpleNamespace

from polima.bundle.contract import spec_from_bundle
from polima.policies.act import ACT_SPEC


def bundle_with(tmp_path, smoke):
    (tmp_path / "plan.json").write_text(json.dumps({}))
    return SimpleNamespace(plan_path=tmp_path / "plan.json", smoke=smoke)


def test_the_fixture_check_uses_the_bundles_own_pass_mark(tmp_path):
    spec = spec_from_bundle(ACT_SPEC, bundle_with(tmp_path, {"cosine_min": 0.995, "mean_abs_max": 0.03}))
    assert (spec.smoke.cosine_min, spec.smoke.mean_abs_max) == (0.995, 0.03)


def test_a_bundle_without_a_pass_mark_keeps_the_policys(tmp_path):
    spec = spec_from_bundle(ACT_SPEC, bundle_with(tmp_path, {}))
    assert spec.smoke == ACT_SPEC.smoke
