"""Episode spec grammar.

A reversed range such as "9-0" is an error, not an empty selection.
"""

from __future__ import annotations

import pytest

from polima.data.episodes import (
    EpisodeSpecError,
    episodes_json,
    format_episode_spec,
    parse_episode_spec,
)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("0", [0]),
        ("7", [7]),
        ("0-3", [0, 1, 2, 3]),
        ("5-5", [5]),
        ("0,2,5-8", [0, 2, 5, 6, 7, 8]),
        ("3,1,2", [1, 2, 3]),                  # sorted
        ("1,1,1", [1]),                        # de-duplicated
        ("0-2,1-3", [0, 1, 2, 3]),             # overlapping ranges merge
        (" 0 , 2 ", [0, 2]),                   # whitespace tolerated
        ("10-12", [10, 11, 12]),
        ("0-9", list(range(10))),
        ("0,10-12,20", [0, 10, 11, 12, 20]),
    ],
)
def test_parse_valid(spec, expected):
    assert parse_episode_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "   ", "a", "1-a", "1,,2", "-1", "1-", "1.5", "0--2"])
def test_parse_invalid(spec):
    with pytest.raises(EpisodeSpecError):
        parse_episode_spec(spec)


def test_backwards_range_is_an_error_not_an_empty_selection():
    with pytest.raises(EpisodeSpecError, match="runs backwards"):
        parse_episode_spec("9-0")


def test_episodes_json_is_what_lerobot_train_expects():
    assert episodes_json("0,2,5-8") == "[0, 2, 5, 6, 7, 8]"


@pytest.mark.parametrize(
    ("indices", "expected"),
    [([0, 1, 2, 3], "0-3"), ([0, 2], "0,2"), ([5], "5"), ([], ""),
     ([0, 1, 2, 5, 7, 8, 9], "0-2,5,7-9")],
)
def test_format_round_trip(indices, expected):
    assert format_episode_spec(indices) == expected
    if indices:
        assert parse_episode_spec(expected) == sorted(set(indices))
