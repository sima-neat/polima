"""Compiler-process entry point: ``python compile_onnx.py [persistent] ARGS``.

Runs :mod:`polima.compile.tensor` (or its ``persistent`` mode) as a script whose
file name the host-wide compile counters on shared build hosts recognise
(``compile_onnx.py`` / ``compile_persistent.py`` command lines), so GR00T builds
count against the same compiler-process limit as every other build there.
"""
import sys

if __name__ == "__main__":
    sys.path.pop(0)  # this directory: keep its module names from shadowing others
    from polima.compile import tensor

    argv = sys.argv[1:]
    if argv[:1] == ["persistent"]:
        raise SystemExit(tensor.persistent_main(argv[1:]))
    raise SystemExit(tensor.main(argv))
