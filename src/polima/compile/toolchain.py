"""Locating and invoking the SiMa model-compiler venv.

The compiler is a separate interpreter from everything else PoLiMa runs in: it
has `afe`, `onnx` and `onnxsim` but no torch, while the training env has torch
but no afe. Neither can import the other's modules, so the compile stage is
always a subprocess, never an import.

`polima doctor` and `polima compile` need the same discovery, so it lives here
rather than in either.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Searched in order. The `/sdk-extensions` entry is where the SiMa container
#: mounts it, and is also the path baked into `llima-compile`'s shebang.
SEARCH_PATH = (
    "$MODEL_COMPILER_BIN",
    "~/sima-sdk-extensions/model-compiler/bin",
    "~/sdk-extensions/model-compiler/bin",
    "/sdk-extensions/model-compiler/bin",
)


def find_compiler_bin(config=None) -> Path | None:
    """First candidate that actually contains a `python`."""
    candidates: list[str] = []
    explicit = getattr(getattr(config, "paths", None), "model_compiler_bin", None)
    if explicit:
        candidates.append(str(explicit))
    candidates.append(os.environ.get("MODEL_COMPILER_BIN", ""))
    candidates.append(os.path.expanduser("~/sima-sdk-extensions/model-compiler/bin"))
    candidates.append(os.path.expanduser("~/sdk-extensions/model-compiler/bin"))  # sima-cli
    candidates.append("/sdk-extensions/model-compiler/bin")
    for candidate in candidates:
        if candidate and (Path(candidate) / "python").exists():
            return Path(candidate)
    return None


def require_compiler_python(config=None) -> Path:
    compiler_bin = find_compiler_bin(config)
    if compiler_bin is None:
        raise FileNotFoundError(
            "SiMa model compiler not found. Searched: "
            + ", ".join(SEARCH_PATH)
            + ".\nSet MODEL_COMPILER_BIN=/path/to/model-compiler/bin"
        )
    return compiler_bin / "python"


#: Palette ships this in the modelsdk container. Sourcing it is what puts the
#: compiler's own shared libraries on the loader path -- without it `import afe`
#: dies on a missing libLLVM, which reads like a broken install rather than a
#: missing activation step.
ACTIVATION_SCRIPT = "activate-model-compiler"

#: Variables worth taking from the activation. Copying the whole environment
#: would drag in the subshell's own PATH/PWD/SHLVL and quietly undo the caller's.
ACTIVATION_VARS = (
    "LD_LIBRARY_PATH", "PATH", "PYTHONPATH", "VIRTUAL_ENV",
    "SIMA_SDK_ROOT", "MODEL_SDK_ROOT", "LD_PRELOAD",
)


def find_activation(compiler_bin: str | Path | None = None) -> Path | None:
    """Locate `activate-model-compiler`, if this environment has one."""
    from shutil import which

    if compiler_bin:
        candidate = Path(compiler_bin) / ACTIVATION_SCRIPT
        if candidate.is_file():
            return candidate
    found = which(ACTIVATION_SCRIPT)
    if found:
        return Path(found)
    for directory in ("/usr/local/bin", "/opt/bin", "/sdk-extensions"):
        candidate = Path(directory) / ACTIVATION_SCRIPT
        if candidate.is_file():
            return candidate
    return None


def activation_env(script: str | Path) -> dict[str, str]:
    """Source the activation in a subshell and return what it changed.

    Reading the environment back rather than running every compile under
    `bash -lc 'source ... && ...'` keeps the subprocess call sites unchanged --
    and means the preflight, the version probe and the per-graph compiles all
    get the same environment by construction rather than by remembering.
    """
    import os
    import subprocess

    try:
        result = subprocess.run(
            ["bash", "-c", f"source {script} >/dev/null 2>&1 && env -0"],
            capture_output=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}

    changed: dict[str, str] = {}
    for entry in result.stdout.decode("utf-8", "replace").split("\0"):
        name, separator, value = entry.partition("=")
        if not separator or name not in ACTIVATION_VARS:
            continue
        if os.environ.get(name) != value:
            changed[name] = value
    return changed


def compiler_env(compiler_bin: str | Path, source_root: str | Path | None = None) -> dict[str, str]:
    """Environment for a compiler subprocess.

    Two things matter here, both inherited from the legacy scripts:

    * `PATH` is prefixed with the compiler bin, because afe shells out to its own
      helper executables and finds them by name.
    * `CUDA_VISIBLE_DEVICES` is cleared. The compiler will happily grab a GPU and
      then contend with a training run on the same host; the legacy scripts set
      it empty for exactly that reason. `MODEL_COMPILER_CUDA_VISIBLE_DEVICES`
      overrides it when the GPU is genuinely wanted.
    """
    env = dict(os.environ)

    # Apply the Palette activation first, so the explicit settings below win.
    # Without it `import afe` fails on a missing shared library -- the compiler
    # venv links against libraries only that activation puts on the loader path.
    script = find_activation(compiler_bin)
    if script is not None:
        env.update(activation_env(script))

    env["PATH"] = f"{compiler_bin}:{env.get('PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = env.get("MODEL_COMPILER_CUDA_VISIBLE_DEVICES", "")
    if source_root:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{source_root}:{existing}" if existing else str(source_root)
    return env
