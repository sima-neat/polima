"""Policy registry.

`get_policy("act")` lazily imports polima.policies.act, which registers itself on
import. Lazy because importing a policy module may pull in torch-side helpers
that do not exist in the model-compiler venv or on the board.

Third-party policies can register through the "polima.policies" entry-point
group; the first-party specs are imported directly so there is no
discovery cost and no import-ordering surprise.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

from polima.policies.base import PolicySpec, SpecError

#: Policies shipped with PoLiMa.
BUILTIN = ("act", "groot", "smolvla")

_REGISTRY: dict[str, PolicySpec] = {}


def register_policy(spec: PolicySpec, *, validate: bool = True) -> PolicySpec:
    """Register a spec. Validation runs here so a malformed spec fails at import
    time -- during `polima doctor` -- rather than midway through a deploy."""
    if validate:
        spec.validate()
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing is not spec:
        raise SpecError(f"policy {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get_policy(name: str) -> PolicySpec:
    if name not in _REGISTRY:
        _import_policy(name)
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown policy {name!r}; available: {sorted(available())}"
        ) from None


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def registered() -> list[str]:
    return sorted(_REGISTRY)


def available() -> list[str]:
    """Every policy that could be loaded -- built-ins plus anything already
    registered. A built-in whose module cannot import in this interpreter is
    still listed, but will raise a clear error when actually requested."""
    return sorted(set(BUILTIN) | set(_REGISTRY))


def load_all(*, strict: bool = False) -> dict[str, PolicySpec]:
    """Import every built-in. `polima doctor` and `polima list` use this; with
    strict=False a policy that is not implemented yet is simply skipped."""
    for name in BUILTIN:
        if name in _REGISTRY:
            continue
        try:
            _import_policy(name)
        except (ImportError, SpecError):
            if strict:
                raise
    return dict(_REGISTRY)


def iter_specs() -> Iterator[PolicySpec]:
    yield from (_REGISTRY[name] for name in sorted(_REGISTRY))


def _import_policy(name: str) -> None:
    try:
        importlib.import_module(f"polima.policies.{name}")
        return
    except ImportError as first_error:
        for spec in _entry_points():
            if spec.name == name:
                registered_spec = spec.load()
                if isinstance(registered_spec, PolicySpec):
                    register_policy(registered_spec)
                return
        if name in BUILTIN:
            raise ImportError(
                f"policy {name!r} is a built-in but is not implemented yet "
                f"(ported policies: {registered()})"
            ) from first_error
        if getattr(first_error, "name", None) == f"polima.policies.{name}":
            raise ImportError(
                f"unknown policy {name!r}; available: {available()}"
            ) from first_error
        raise


def _entry_points():
    try:
        from importlib.metadata import entry_points

        return entry_points(group="polima.policies")
    except Exception:  # noqa: BLE001 - entry-point discovery must never be fatal
        return ()
