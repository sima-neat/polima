"""Episode range specs: "0-9", "0,2,5-8", "3".

Ranges are validated: "9-0" is an error, not an empty selection.
"""

from __future__ import annotations

import json


class EpisodeSpecError(ValueError):
    """The spec is not a valid episode selection."""


def parse_episode_spec(spec: str) -> list[int]:
    """Expand a comma-separated list of indices and inclusive ranges.

    Returns a sorted, de-duplicated list. Raises EpisodeSpecError with a message
    naming the offending token rather than failing silently.

    >>> parse_episode_spec("0-3")
    [0, 1, 2, 3]
    >>> parse_episode_spec("0,2,5-8")
    [0, 2, 5, 6, 7, 8]
    """
    if spec is None:
        raise EpisodeSpecError("episode spec is required")
    text = spec.strip()
    if not text:
        raise EpisodeSpecError("episode spec is empty")

    selected: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            raise EpisodeSpecError(f"empty item in episode spec {spec!r}")
        if "-" in token.lstrip("-"):
            start_text, _, end_text = token.partition("-")
            start, end = _index(start_text, spec), _index(end_text, spec)
            if start > end:
                raise EpisodeSpecError(
                    f"range {token!r} in {spec!r} runs backwards ({start} > {end})"
                )
            selected.update(range(start, end + 1))
        else:
            selected.add(_index(token, spec))
    return sorted(selected)


def episodes_json(spec: str) -> str:
    """The JSON array form `lerobot-train --dataset.episodes` expects."""
    return json.dumps(parse_episode_spec(spec))


def format_episode_spec(indices: list[int]) -> str:
    """Inverse of parse_episode_spec -- collapse runs back into ranges."""
    if not indices:
        return ""
    ordered = sorted(set(indices))
    parts: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(_run(start, previous))
        start = previous = value
    parts.append(_run(start, previous))
    return ",".join(parts)


def _run(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def _index(text: str, spec: str) -> int:
    text = text.strip()
    if not text.isdigit():
        raise EpisodeSpecError(f"{text!r} in episode spec {spec!r} is not a non-negative integer")
    return int(text)
