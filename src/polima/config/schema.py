"""Dataclass <-> plain-dict conversion.

This is what makes a config object crossable between the three interpreters: the
host conda env builds it, serializes to JSON, the model-compiler venv or the
Modalix board reads it back. No pydantic (absent from every env that matters),
no pickle (version-fragile across py3.11/3.12).

Handles nested dataclasses, tuples/lists/dicts of them, Path, Enum and Optional.
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a dict cannot be coerced into the target dataclass."""


def to_dict(value: Any) -> Any:
    """Recursively convert to JSON-safe primitives. Tuples become lists."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_dict(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_dict(v) for v in value]
    return value


def from_dict(cls: type[T], data: Any, *, path: str = "") -> T:
    """Coerce a plain dict into `cls`, honouring field type annotations."""
    if not dataclasses.is_dataclass(cls):
        return _coerce(cls, data, path)
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: expected an object, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(
            f"{path or cls.__name__}: unknown key(s) {sorted(unknown)}; "
            f"known: {sorted(fields)}"
        )

    kwargs: dict[str, Any] = {}
    for name in fields:
        if name not in data:
            continue
        kwargs[name] = _coerce(hints.get(name, Any), data[name], f"{path}.{name}" if path else name)
    try:
        return cls(**kwargs)  # type: ignore[return-value]
    except TypeError as exc:
        raise ConfigError(f"{path or cls.__name__}: {exc}") from exc


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    if annotation is Any or annotation is None:
        return value

    origin = get_origin(annotation)

    # Optional[X] / X | None / unions
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        if len(args) == 1:
            return _coerce(args[0], value, path)
        for candidate in args:  # best effort across a real union
            try:
                return _coerce(candidate, value, path)
            except (ConfigError, TypeError, ValueError):
                continue
        raise ConfigError(f"{path}: no union member matched {value!r}")

    if origin in (list, tuple, set, frozenset):
        args = get_args(annotation)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            item_type = args[0]
        elif origin is tuple and args:
            return tuple(_coerce(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
        else:
            item_type = args[0] if args else Any
        items = [_coerce(item_type, v, f"{path}[{i}]") for i, v in enumerate(value)]
        return origin(items)

    if origin in (dict,) or annotation is dict:
        args = get_args(annotation)
        val_type = args[1] if len(args) == 2 else Any
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected an object, got {type(value).__name__}")
        return {str(k): _coerce(val_type, v, f"{path}.{k}") for k, v in value.items()}

    # typing.Mapping / Sequence and friends fall through to their origin above;
    # anything left that is a plain class:
    if isinstance(annotation, type):
        if dataclasses.is_dataclass(annotation):
            return from_dict(annotation, value, path=path)
        if issubclass(annotation, enum.Enum):
            return annotation(value)
        if annotation is Path:
            return Path(value)
        if annotation is bool:
            return _as_bool(value, path)
        if annotation in (int, float, str):
            try:
                return annotation(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{path}: cannot read {value!r} as {annotation.__name__}") from exc

    return value


def _as_bool(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off", ""):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ConfigError(f"{path}: cannot read {value!r} as a boolean")


def merge(base: dict, overlay: dict) -> dict:
    """Deep-merge `overlay` onto `base`. Lists replace, they do not concatenate."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        elif value is not None:
            out[key] = value
    return out
