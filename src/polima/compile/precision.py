"""Named compile-time precision profiles.

Policy specs describe the conservative, known-working build. Experiments must
not mutate that default: doing so would make an ordinary ``--policy groot``
silently produce a different model and could also reuse artifacts under the
wrong human-readable label. A named profile returns a new immutable spec.
"""

from __future__ import annotations

import re
from dataclasses import replace

PRECISION_PROFILES = ("policy", "groot-w8a16")

_GROOT_W8A16 = re.compile(
    r"^(?:eagle_vision_\d\d_\d\d|eagle_qwen_\d\d_\d\d|dit_blocks_\d\d_\d\d)$"
)


def apply_precision_profile(spec, profile: str):
    """Return ``spec`` with the requested graph-level precision overrides.

    ``groot-w8a16`` keeps every graph boundary and activation in BF16 while
    using per-channel symmetric INT8 weights in the weight-heavy transformer
    block pairs. Patch/projection/connector/norm/tail graphs remain fully BF16
    for the first controlled mixed-precision experiment.
    """
    if profile == "policy":
        return spec
    if profile not in PRECISION_PROFILES:
        raise ValueError(
            f"unknown precision profile {profile!r}; have {list(PRECISION_PROFILES)}"
        )
    if profile != "groot-w8a16" or spec.name != "groot":
        raise ValueError(f"precision profile {profile!r} requires --policy groot")

    graphs = tuple(
        replace(
            graph,
            activation_precision="bf16",
            weight_precision="int8",
        )
        if _GROOT_W8A16.fullmatch(graph.name)
        else graph
        for graph in spec.compile.graphs
    )
    configured = replace(spec, compile=replace(spec.compile, graphs=graphs))
    configured.validate()
    return configured


def effective_precision_label(
    precision: str,
    activation_precision: str | None,
    weight_precision: str | None,
) -> str:
    """Canonical bundle label for the precision the compiler actually used."""
    activation = activation_precision or precision
    weights = weight_precision or precision
    if activation == weights == precision:
        return precision
    return f"A_{activation.upper()}_W_{weights.upper()}"
