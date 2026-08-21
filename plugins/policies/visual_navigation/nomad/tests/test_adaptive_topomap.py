"""Tests for model-aware adaptive topomap selection."""

from dataclasses import replace

import pytest

from nomad_runtime.adaptive_topomap import (
    AdaptiveTopomapConfig,
    CandidateScore,
    select_candidate,
)


def _candidate(position: int, distance: float) -> CandidateScore:
    return CandidateScore(
        position=position,
        source_node=position,
        timestamp_s=float(position),
        time_delta_s=float(position),
        distance=distance,
        minimum_distance=distance,
        maximum_distance=distance,
    )


def test_selects_farthest_candidate_in_preferred_band() -> None:
    candidates = [
        _candidate(1, 6.5),
        _candidate(2, 9.0),
        _candidate(3, 11.5),
        _candidate(4, 12.5),
    ]

    decision = select_candidate(candidates, AdaptiveTopomapConfig())

    assert decision.candidate.position == 3
    assert decision.reason == "farthest_preferred"


def test_falls_back_to_hard_candidate_closest_to_target() -> None:
    candidates = [
        _candidate(1, 3.5),
        _candidate(2, 15.0),
        _candidate(3, 16.0),
    ]

    decision = select_candidate(candidates, AdaptiveTopomapConfig())

    assert decision.candidate.position == 1
    assert decision.reason == "closest_to_target"


def test_marks_selection_outside_hard_range() -> None:
    candidates = [_candidate(1, 1.0), _candidate(2, 18.0)]

    decision = select_candidate(candidates, AdaptiveTopomapConfig())

    assert decision.candidate.position == 1
    assert decision.reason == "outside_hard_range"


def test_rejects_invalid_threshold_order() -> None:
    config = replace(
        AdaptiveTopomapConfig(), preferred_min_distance=10.0,
        target_distance=9.0,
    )

    with pytest.raises(ValueError, match="monotonically nested"):
        config.validate()


def test_requires_candidate_scores() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_candidate([], AdaptiveTopomapConfig())
