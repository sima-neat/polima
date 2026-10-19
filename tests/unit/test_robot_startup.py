"""Robot-client startup: import stubs for unused libraries, and the startup timeline."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"


def _run(code: str, env: dict | None = None) -> dict:
    """Run `code` in a fresh interpreter (stubs change sys.meta_path for good)."""
    import os

    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(SRC), **(env or {})}, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_stubbed_libraries_load_as_stand_ins():
    out = _run("""
        import json, sys
        from polima.robot import fast_imports
        fast_imports.STUBBED["zz_heavy_lib"] = "test"
        assert fast_imports.install({})
        import zz_heavy_lib.utils.logging as log
        from zz_heavy_lib import Accelerator, register_to_config
        log.disable_progress_bar()                      # functions: no-ops
        def decorated(): return 7
        assert register_to_config(decorated) is decorated  # decorators pass through
        class Mine(Accelerator):                        # classes: usable as bases and annotations
            pass
        hint = Accelerator | None
        try:
            Accelerator()
            raised = ""
        except RuntimeError as error:
            raised = str(error)
        print(json.dumps({"raised": raised, "active": fast_imports.active()}))
    """)
    assert "not loaded in the PoLiMa robot client" in out["raised"] and "POLIMA_ROBOT_FULL_IMPORTS" in out["raised"]
    assert {"zz_heavy_lib", "zz_heavy_lib.utils.logging"} <= set(out["active"])


def test_deepdiff_stand_in_is_an_equality_check():
    out = _run("""
        import json
        from polima.robot import fast_imports
        fast_imports.install({})
        from deepdiff import DeepDiff
        same = {"Goal_Position": (42, 2)}
        print(json.dumps({"equal": bool(DeepDiff(same, dict(same))),
                          "different": bool(DeepDiff(same, {"Goal_Position": (43, 2)}))}))
    """)
    assert out == {"equal": False, "different": True}


def test_full_imports_opt_out_installs_nothing():
    out = _run("""
        import json, sys
        from polima.robot import fast_imports
        before = list(sys.meta_path)
        active = fast_imports.install()
        print(json.dumps({"active": active, "unchanged": sys.meta_path == before}))
    """, env={"POLIMA_ROBOT_FULL_IMPORTS": "1"})
    assert out == {"active": False, "unchanged": True}


def test_only_the_listed_modules_are_stubbed():
    out = _run("""
        import json
        from polima.robot import fast_imports
        fast_imports.install({})
        import json as real_json, numpy
        print(json.dumps({"numpy_real": hasattr(numpy, "ndarray") and type(numpy).__name__ == "module",
                          "prefix_only": [fast_imports._stubbed(n) for n in
                                          ("lerobot.policies.act", "lerobot.policiesx", "lerobot.robots.so_follower",
                                           "lerobot.datasets.lerobot_dataset", "lerobot.datasets.utils")]}))
    """)
    assert out["numpy_real"]
    assert out["prefix_only"] == [True, False, False, True, False]


def test_startup_marks_each_phase_once_with_elapsed_time(monkeypatch, capsys):
    from polima.robot import so101, startup

    events = []
    monkeypatch.setattr(so101, "record", lambda kind, **fields: events.append((kind, fields)))
    monkeypatch.setattr(startup, "_seen", {})
    first = startup.mark("imports_done")
    assert startup.mark("imports_done") is None
    second = startup.mark("first_action")
    assert first is not None and second >= first >= 0
    assert [e[1]["phase"] for e in events] == ["imports_done", "first_action"]
    out = capsys.readouterr().out
    assert "[startup] imports done at" in out and "[startup] first action at" in out


def test_process_start_reads_proc_on_linux():
    from polima.robot import startup

    if not Path("/proc/self/stat").exists():
        pytest.skip("no /proc")
    start = startup.process_start()
    import time

    assert 0 <= time.time() - start < 600  # this test process started recently


def test_draccus_docstrings_are_skipped_unless_opted_out():
    pytest.importorskip("draccus")
    out = _run("""
        import json
        from polima.robot import fast_imports
        from draccus.wrappers import docstring
        replaced = fast_imports.skip_config_docstrings({})
        doc = docstring.get_attribute_docstring(object, "anything")
        print(json.dumps({"replaced": replaced, "empty": type(doc).__name__ == "AttributeDocString"}))
    """)
    assert out == {"replaced": True, "empty": True}
    out = _run("""
        import json
        from polima.robot import fast_imports
        print(json.dumps({"replaced": fast_imports.skip_config_docstrings()}))
    """, env={"POLIMA_ROBOT_FULL_IMPORTS": "1"})
    assert out == {"replaced": False}
