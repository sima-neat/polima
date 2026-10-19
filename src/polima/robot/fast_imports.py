"""Keep the SO-101 robot client from importing libraries it never uses.

The client drives a serial arm and two USB cameras and asks the Modalix model
server for actions over TCP. LeRobot's robot, motor, processor and async-client
modules nonetheless import training and model code at module level: every
LeRobot policy (GR00T's diffusers, Pi0's real-time chunking, which loads
torch._dynamo and sympy), torchvision, Hugging Face datasets, accelerate and
deepdiff. On a Modalix DevKit that took 16 s of the client's 20 s of imports
(LeRobot 0.4.4); with these stubs the same imports take 8 s, most of it torch.

`install()` puts a finder in front of the import system that answers for the
modules below with stand-ins:

- a name that reads like a function (lowercase) is a no-op that returns None,
  or its argument when used as a decorator, which covers the registrations and
  progress-bar toggles LeRobot runs at import time;
- a class (capitalised) can be subclassed and used in annotations, but
  instantiating it raises, so code that really needs one fails loudly instead
  of running on a stand-in;
- deepdiff.DeepDiff is a real equality check (motors_bus compares the control
  tables of mixed motor models with it; an SO-101 has one model).

POLIMA_ROBOT_FULL_IMPORTS=1 turns this off.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import types

ENV = "POLIMA_ROBOT_FULL_IMPORTS"

#: Module (prefix) -> why the robot client does not need it.
STUBBED = {
    "lerobot.policies": "policy code; async_inference.helpers imports every policy config for "
                        "LeRobot's own policy server, which PoLiMa's client does not use",
    "lerobot.datasets.lerobot_dataset": "dataset loading (pandas, video decoding); imported by the "
                                        "normalizer processor",
    "torchvision": "image transforms of processor steps the client does not run",
    "datasets": "Hugging Face datasets: lerobot.utils.utils imports it to toggle progress bars",
    "accelerate": "lerobot.utils.utils imports Accelerator for a type annotation",
    "diffusers": "GR00T's action head (via lerobot.policies)",
    "deepdiff": "motors_bus compares mixed motor models with it; replaced by an equality check",
}

_installed: _StubFinder | None = None


def _stubbed(fullname: str) -> bool:
    return any(fullname == name or fullname.startswith(name + ".") for name in STUBBED)


class _StubClassMeta(type):
    """Stand-in classes work in annotations (X | None, X[...]) and as base classes."""

    def __or__(cls, other):
        return cls

    def __ror__(cls, other):
        return cls

    def __getitem__(cls, item):
        return cls

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stand_in(f"{cls.__qualname__}.{name}")


def _stand_in(qualified: str):
    leaf = qualified.rsplit(".", 1)[-1]
    if leaf[:1].islower():
        def function(*args, **kwargs):
            # A decorator stand-in passes the decorated function through.
            return args[0] if len(args) == 1 and not kwargs and callable(args[0]) else None
        function.__qualname__ = qualified
        return function

    def unavailable(*_args, **_kwargs):
        raise RuntimeError(f"{qualified} is not loaded in the PoLiMa robot client (fast imports); "
                           f"set {ENV}=1 to load every library")
    return _StubClassMeta(leaf, (), {"__init__": unavailable, "__qualname__": qualified,
                                     "__module__": qualified.rsplit(".", 1)[0]})


class DeepDiff(dict):
    """deepdiff.DeepDiff for motors_bus: empty (falsy) when the two values are equal."""

    def __init__(self, first, second, **_kwargs):
        super().__init__({} if first == second else {"values_changed": {"root": {"old_value": first,
                                                                                  "new_value": second}}})


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = DeepDiff if (self.__name__, name) == ("deepdiff", "DeepDiff") else \
            _stand_in(f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if not _stubbed(fullname):
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        module.__path__ = []


def install(environ=os.environ) -> bool:
    """Stub the modules above for this process. Returns whether stubs are active.

    Must run before LeRobot is first imported; a module that is already loaded
    stays as it is."""
    global _installed
    if environ.get(ENV, "").strip() not in ("", "0"):
        return False
    if _installed is None:
        _installed = _StubFinder()
        sys.meta_path.insert(0, _installed)
    return True


def skip_config_docstrings(environ=os.environ) -> bool:
    """Stop draccus reading attribute docstrings while it builds LeRobot's argument parser.

    For help text, draccus re-reads and parses (ast) the source of every
    configuration dataclass, including every robot choice, on each parse: 7.4 s
    on a Modalix DevKit (draccus 0.10.0). The client's arguments come from
    polima.robot.so101, so the only loss is field descriptions under --help.
    Returns whether the hook was replaced."""
    if environ.get(ENV, "").strip() not in ("", "0"):
        return False
    try:
        from draccus.wrappers import docstring
    except ImportError:
        return False
    empty = getattr(docstring, "AttributeDocString", None)
    if empty is None or not callable(getattr(docstring, "get_attribute_docstring", None)):
        return False
    docstring.get_attribute_docstring = lambda some_dataclass, field_name: empty()
    return True


def active() -> list[str]:
    """The stubbed modules this process has imported (for the run log)."""
    return sorted(name for name, module in sys.modules.items() if isinstance(module, _StubModule))
