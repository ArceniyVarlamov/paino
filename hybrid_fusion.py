from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.append(vendor_path)

import numpy as np

from compat import compat_zip
from hsmm_follower import ScoreFollowerHSMM
from oltw_follower import ScoreFollowerOLTW

HYBRID_PROFILE_TUNING_KEYS = frozenset(
    {
        "confidence_threshold",
        "resync_gap",
        "nudge_target_mass",
        "sigma",
        "outlier_pitch_clip",
        "max_local_cost",
        "max_forward_match_gap",
        "max_forward_match_lead_over_oltw",
        "max_forward_step",
        "recovery_confirmation_events",
        "anchor_window_lengths",
        "anchor_pitch_clip",
        "anchor_total_cost_threshold",
        "anchor_margin_threshold",
        "anchor_time_weight",
        "anchor_min_tempo_scale",
        "anchor_max_tempo_scale",
        "anchor_local_improvement_threshold",
        "anchor_search_max_events",
        "anchor_confirmation_events",
        "anchor_stability_tolerance",
        "anchor_min_jump",
        "anchor_min_supporting_windows",
        "anchor_local_preference_margin",
        "output_confirmation_events",
        "output_high_confidence",
    }
)
HYBRID_PROFILE_FORMAT_VERSION = 2


def hybrid_profile_path(score_json_path: str | Path) -> Path:
    score_path = Path(score_json_path)
    return score_path.with_suffix(".hybrid_profile.json")


def load_hybrid_profile(
    score_json: str | Path | dict[str, Any] | list[dict[str, Any]],
) -> tuple[dict[str, Any], Path | None]:
    if not isinstance(score_json, (str, Path)):
        return {}, None

    score_path = Path(score_json)
    if score_path.suffix.lower() != ".json":
        return {}, None

    profile_path = hybrid_profile_path(score_path)
    if not profile_path.exists():
        return {}, profile_path

    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, profile_path

    format_version = payload.get("format_version")
    if format_version != HYBRID_PROFILE_FORMAT_VERSION:
        return {}, profile_path

    tuning = payload.get("tuning", payload)
    if not isinstance(tuning, dict):
        return {}, profile_path

    overrides = {
        key: value for key, value in tuning.items() if key in HYBRID_PROFILE_TUNING_KEYS
    }
    if "anchor_window_lengths" in overrides:
        try:
            overrides["anchor_window_lengths"] = tuple(
                int(length) for length in overrides["anchor_window_lengths"]
            )
        except (TypeError, ValueError):
            overrides.pop("anchor_window_lengths", None)

    return overrides, profile_path


class HybridScoreFollower:
    """Fuse HSMM confidence with OLTW recovery behavior."""

    _ANCHOR_INTRA_CHORD_GAP = 0.012

    def __init__(
        self,
        score_json: str | Path | dict[str, Any] | list[dict[str, Any]],
        *,
        load_tuning_profile: bool = True,
        confidence_threshold: float = 0.4,
        resync_gap: int = 1,
        nudge_target_mass: float = 0.95,
        sigma: float = 2.5,
        outlier_pitch_clip: float = 6.0,
        max_local_cost: float = 6.0,
        max_forward_match_gap: int = 4,
        max_forward_match_lead_over_oltw: int = 2,
        max_forward_step: int = 3,
        recovery_confirmation_events: int = 1,
        anchor_window_lengths: tuple[int, ...] = (20, 16, 12, 8, 6, 4),
        anchor_pitch_clip: float = 6.0,
        anchor_total_cost_threshold: float = 1.35,
        anchor_margin_threshold: float = 0.05,
        anchor_time_weight: float = 1.25,
        anchor_min_tempo_scale: float = 0.35,
        anchor_max_tempo_scale: float = 3.50,
        anchor_local_improvement_threshold: float = 0.35,
        anchor_search_max_events: int = 100_000,
        anchor_confirmation_events: int = 2,
        anchor_stability_tolerance: int = 2,
        anchor_min_jump: int = 8,
        anchor_min_supporting_windows: int = 2,
        anchor_local_preference_margin: float = 0.05,
        output_confirmation_events: int = 2,
        output_high_confidence: float = 0.4,
        allow_backward_output: bool = True,
    ) -> None:
        self.loaded_profile_path: Path | None = None
        self.loaded_profile_overrides: dict[str, Any] = {}
        if load_tuning_profile:
            profile_overrides, profile_path = load_hybrid_profile(score_json)
            self.loaded_profile_path = profile_path if profile_overrides else None
            self.loaded_profile_overrides = dict(profile_overrides)
            confidence_threshold = float(
                profile_overrides.get("confidence_threshold", confidence_threshold)
            )
            resync_gap = int(profile_overrides.get("resync_gap", resync_gap))
            nudge_target_mass = float(
                profile_overrides.get("nudge_target_mass", nudge_target_mass)
            )
            sigma = float(profile_overrides.get("sigma", sigma))
            outlier_pitch_clip = float(
                profile_overrides.get("outlier_pitch_clip", outlier_pitch_clip)
            )
            max_local_cost = float(profile_overrides.get("max_local_cost", max_local_cost))
            max_forward_match_gap = int(
                profile_overrides.get("max_forward_match_gap", max_forward_match_gap)
            )
            max_forward_match_lead_over_oltw = int(
                profile_overrides.get(
                    "max_forward_match_lead_over_oltw",
                    max_forward_match_lead_over_oltw,
                )
            )
            max_forward_step = int(profile_overrides.get("max_forward_step", max_forward_step))
            recovery_confirmation_events = int(
                profile_overrides.get(
                    "recovery_confirmation_events",
                    recovery_confirmation_events,
                )
            )
            anchor_window_lengths = tuple(
                profile_overrides.get("anchor_window_lengths", anchor_window_lengths)
            )
            anchor_pitch_clip = float(
                profile_overrides.get("anchor_pitch_clip", anchor_pitch_clip)
            )
            anchor_total_cost_threshold = float(
                profile_overrides.get(
                    "anchor_total_cost_threshold",
                    anchor_total_cost_threshold,
                )
            )
            anchor_margin_threshold = float(
                profile_overrides.get("anchor_margin_threshold", anchor_margin_threshold)
            )
            anchor_time_weight = float(
                profile_overrides.get("anchor_time_weight", anchor_time_weight)
            )
            anchor_min_tempo_scale = float(
                profile_overrides.get("anchor_min_tempo_scale", anchor_min_tempo_scale)
            )
            anchor_max_tempo_scale = float(
                profile_overrides.get("anchor_max_tempo_scale", anchor_max_tempo_scale)
            )
            anchor_local_improvement_threshold = float(
                profile_overrides.get(
                    "anchor_local_improvement_threshold",
                    anchor_local_improvement_threshold,
                )
            )
            anchor_search_max_events = int(
                profile_overrides.get("anchor_search_max_events", anchor_search_max_events)
            )
            anchor_confirmation_events = int(
                profile_overrides.get(
                    "anchor_confirmation_events",
                    anchor_confirmation_events,
                )
            )
            anchor_stability_tolerance = int(
                profile_overrides.get(
                    "anchor_stability_tolerance",
                    anchor_stability_tolerance,
                )
            )
            anchor_min_jump = int(profile_overrides.get("anchor_min_jump", anchor_min_jump))
            anchor_min_supporting_windows = int(
                profile_overrides.get(
                    "anchor_min_supporting_windows",
                    anchor_min_supporting_windows,
                )
            )
            anchor_local_preference_margin = float(
                profile_overrides.get(
                    "anchor_local_preference_margin",
                    anchor_local_preference_margin,
                )
            )
            output_confirmation_events = int(
                profile_overrides.get(
                    "output_confirmation_events",
                    output_confirmation_events,
                )
            )
            output_high_confidence = float(
                profile_overrides.get("output_high_confidence", output_high_confidence)
            )

        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in the interval (0, 1]")
        if resync_gap < 1:
            raise ValueError("resync_gap must be at least 1")
        if not 0.5 < nudge_target_mass <= 1.0:
            raise ValueError("nudge_target_mass must be in the interval (0.5, 1]")
        if max_forward_match_gap < 1:
            raise ValueError("max_forward_match_gap must be at least 1")
        if max_forward_match_lead_over_oltw < 0:
            raise ValueError("max_forward_match_lead_over_oltw must be non-negative")
        if max_forward_step < 1:
            raise ValueError("max_forward_step must be at least 1")
        if recovery_confirmation_events < 1:
            raise ValueError("recovery_confirmation_events must be at least 1")
        if not anchor_window_lengths:
            raise ValueError("anchor_window_lengths must not be empty")
        if any(length < 2 for length in anchor_window_lengths):
            raise ValueError("anchor_window_lengths must contain values >= 2")
        if anchor_pitch_clip <= 0.0:
            raise ValueError("anchor_pitch_clip must be positive")
        if anchor_total_cost_threshold <= 0.0:
            raise ValueError("anchor_total_cost_threshold must be positive")
        if anchor_margin_threshold < 0.0:
            raise ValueError("anchor_margin_threshold must be non-negative")
        if anchor_time_weight < 0.0:
            raise ValueError("anchor_time_weight must be non-negative")
        if anchor_min_tempo_scale <= 0.0 or anchor_max_tempo_scale <= 0.0:
            raise ValueError("anchor tempo scale bounds must be positive")
        if anchor_min_tempo_scale > anchor_max_tempo_scale:
            raise ValueError("anchor_min_tempo_scale must be <= anchor_max_tempo_scale")
        if anchor_local_improvement_threshold < 0.0:
            raise ValueError("anchor_local_improvement_threshold must be non-negative")
        if anchor_search_max_events < max(anchor_window_lengths):
            raise ValueError("anchor_search_max_events must cover the largest anchor window")
        if anchor_confirmation_events < 1:
            raise ValueError("anchor_confirmation_events must be at least 1")
        if anchor_stability_tolerance < 0:
            raise ValueError("anchor_stability_tolerance must be non-negative")
        if anchor_min_jump < 1:
            raise ValueError("anchor_min_jump must be at least 1")
        if anchor_min_supporting_windows < 1:
            raise ValueError("anchor_min_supporting_windows must be at least 1")
        if anchor_local_preference_margin < 0.0:
            raise ValueError("anchor_local_preference_margin must be non-negative")
        if output_confirmation_events < 1:
            raise ValueError("output_confirmation_events must be at least 1")
        if not 0.0 < output_high_confidence <= 1.0:
            raise ValueError("output_high_confidence must be in the interval (0, 1]")

        self.hsmm = ScoreFollowerHSMM(
            score_json,
            sigma=sigma,
            outlier_pitch_clip=outlier_pitch_clip,
        )
        self.oltw = ScoreFollowerOLTW(score_json, max_local_cost=max_local_cost)

        if self.hsmm.N != self.oltw.N:
            raise ValueError("HSMM and OLTW must be initialized with the same score length")

        self.confidence_threshold = float(confidence_threshold)
        self.resync_gap = int(resync_gap)
        self.nudge_target_mass = float(nudge_target_mass)
        self.max_forward_match_gap = int(max_forward_match_gap)
        self.max_forward_match_lead_over_oltw = int(max_forward_match_lead_over_oltw)
        self.max_forward_step = int(max_forward_step)
        self.recovery_confirmation_events = int(recovery_confirmation_events)
        self.anchor_window_lengths = tuple(
            sorted({int(length) for length in anchor_window_lengths}, reverse=True)
        )
        self.anchor_pitch_clip = float(anchor_pitch_clip)
        self.anchor_total_cost_threshold = float(anchor_total_cost_threshold)
        self.anchor_margin_threshold = float(anchor_margin_threshold)
        self.anchor_time_weight = float(anchor_time_weight)
        self.anchor_min_tempo_scale = float(anchor_min_tempo_scale)
        self.anchor_max_tempo_scale = float(anchor_max_tempo_scale)
        self.anchor_local_improvement_threshold = float(anchor_local_improvement_threshold)
        self.anchor_search_max_events = int(anchor_search_max_events)
        self.anchor_confirmation_events = int(anchor_confirmation_events)
        self.anchor_stability_tolerance = int(anchor_stability_tolerance)
        self.anchor_min_jump = int(anchor_min_jump)
        self.anchor_min_supporting_windows = int(anchor_min_supporting_windows)
        self.anchor_local_preference_margin = float(anchor_local_preference_margin)
        self.output_confirmation_events = int(output_confirmation_events)
        self.output_high_confidence = float(output_high_confidence)
        self.allow_backward_output = bool(allow_backward_output)
        self.score_data = self.hsmm.score_data
        self.pitches = self.hsmm.pitches
        self.chord_pitch_matrix = self.hsmm.chord_pitch_matrix
        self.N = self.hsmm.N
        (
            self.anchor_event_pitches,
            self.anchor_event_score_positions,
            self.anchor_event_onsets,
            self.anchor_score_end_event_positions,
        ) = self._build_anchor_events()
        self.anchor_event_intervals = np.diff(self.anchor_event_onsets)
        self._interval_window_cache: dict[int, np.ndarray] = {}

        self.last_hsmm_index = int(self.hsmm.current_state_index)
        self.last_oltw_index = int(self.oltw.current_state_index)
        self.last_selected_model = "hsmm"
        self.last_resynced = False
        self.last_recovery_target: int | None = None
        self.last_anchor_target: int | None = None
        self.last_anchor_cost: float | None = None
        self.last_anchor_window: int = 0
        self._current_index = int(self.hsmm.current_state_index)
        self._stable_output_index = int(self.hsmm.current_state_index)
        self._candidate_output_index: int | None = None
        self._candidate_output_streak = 0
        self._recovery_signal_streak = 0
        self._max_anchor_window = int(max(self.anchor_window_lengths))
        self._observed_pitches: list[float] = []
        self._observed_timestamps: list[float] = []
        self._observed_event_count = 0
        self._last_input_timestamp: float | None = None
        self._anchor_search_disabled = False
        self._anchor_candidate_target: int | None = None
        self._anchor_candidate_streak = 0

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def confidence(self) -> float:
        return float(np.max(self.hsmm.alpha))

    @property
    def mode_label(self) -> str:
        if self.last_selected_model == "hsmm":
            return "HMM"
        return "OLTW (Recovery)"

    def process_event(self, pitch: Any, timestamp: float) -> int:
        """Process one observation and return the fused score index."""
        observed_pitches = self._coerce_observed_pitches(pitch)
        event_time = float(timestamp)
        if self._last_input_timestamp is not None and event_time < self._last_input_timestamp:
            event_time = self._last_input_timestamp
        self._last_input_timestamp = event_time
        self._append_observation(observed_pitches, event_time)

        hsmm_index = int(self.hsmm.process_event(observed_pitches, event_time))
        oltw_index = int(self.oltw.process_event(observed_pitches, event_time))

        self.last_hsmm_index = hsmm_index
        self.last_oltw_index = oltw_index
        self.last_resynced = False
        self.last_recovery_target = None
        self.last_anchor_target = None
        self.last_anchor_cost = None
        self.last_anchor_window = 0

        anchor_target = self._sequence_anchor_target()
        self.last_anchor_target = anchor_target

        recovery_target = self._resync_target_position(anchor_target)
        self.last_recovery_target = recovery_target

        should_resync, anchor_resync = self._should_resync(recovery_target, anchor_target)
        if should_resync:
            self._nudge_hsmm_to_position(
                recovery_target,
                event_time,
                allow_large_jump=anchor_resync,
            )
            if anchor_resync:
                self.oltw.seek(recovery_target, event_time)
                self.last_oltw_index = int(self.oltw.current_state_index)
            hsmm_index = int(self.hsmm.current_state_index)
            self.last_hsmm_index = hsmm_index
            self.last_resynced = True

        if self._should_prefer_hsmm():
            selected_index = hsmm_index
            self.last_selected_model = "hsmm"
        elif self.confidence > self.confidence_threshold:
            selected_index = hsmm_index
            self.last_selected_model = "hsmm"
        else:
            selected_index = oltw_index
            self.last_selected_model = "oltw"

        selected_index = self._limit_forward_step(
            selected_index,
            self._current_index,
            allow_large_jump=bool(anchor_resync or self.last_resynced),
        )
        self._current_index = self._debounce_output_index(
            selected_index,
            anchor_resync=bool(anchor_resync),
        )
        return self._current_index

    def seek(self, position: int, timestamp: float | None = None) -> int:
        """Explicitly move the hybrid follower to a chosen score position."""
        event_time = float(self._last_input_timestamp if timestamp is None else timestamp)
        target_position = int(np.clip(position, 0, self.N - 1))

        self.hsmm.seek(target_position, event_time)
        self.oltw.seek(target_position, event_time)

        self.last_hsmm_index = target_position
        self.last_oltw_index = target_position
        self.last_selected_model = "hsmm"
        self.last_resynced = False
        self.last_recovery_target = None
        self.last_anchor_target = None
        self.last_anchor_cost = None
        self.last_anchor_window = 0
        self._current_index = target_position
        self._stable_output_index = target_position
        self._candidate_output_index = None
        self._candidate_output_streak = 0
        self._recovery_signal_streak = 0
        self._observed_pitches.clear()
        self._observed_timestamps.clear()
        self._observed_event_count = 0
        self._last_input_timestamp = event_time
        self._anchor_search_disabled = False
        self._anchor_candidate_target = None
        self._anchor_candidate_streak = 0
        return self._current_index

    @staticmethod
    def _coerce_observed_pitches(pitch: Any) -> np.ndarray:
        if isinstance(pitch, np.ndarray):
            observed_pitches = np.asarray(pitch, dtype=np.float64).reshape(-1)
        elif isinstance(pitch, (list, tuple, set)):
            observed_pitches = np.asarray(list(pitch), dtype=np.float64).reshape(-1)
        else:
            observed_pitches = np.asarray([float(pitch)], dtype=np.float64)

        if observed_pitches.size == 0:
            raise ValueError("observed pitch collection must not be empty")

        return observed_pitches

    def reset_to_start(self) -> int:
        """Reset the fused tracker and all recovery/anchor state to score start."""
        self.hsmm.reset_to_start()
        self.oltw.reset_to_start()

        self.last_hsmm_index = 0
        self.last_oltw_index = 0
        self.last_selected_model = "hsmm"
        self.last_resynced = False
        self.last_recovery_target = None
        self.last_anchor_target = None
        self.last_anchor_cost = None
        self.last_anchor_window = 0
        self._current_index = 0
        self._stable_output_index = 0
        self._candidate_output_index = None
        self._candidate_output_streak = 0
        self._recovery_signal_streak = 0
        self._observed_pitches.clear()
        self._observed_timestamps.clear()
        self._observed_event_count = 0
        self._last_input_timestamp = None
        self._anchor_search_disabled = False
        self._anchor_candidate_target = None
        self._anchor_candidate_streak = 0
        return self._current_index

    def _build_anchor_events(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        event_pitches: list[float] = []
        event_score_positions: list[int] = []
        event_onsets: list[float] = []
        score_end_event_positions = np.zeros(self.N, dtype=np.int64)
        onset_cursor = 0.0
        for position, note in enumerate(self.hsmm.notes):
            onset = float(note.get("nominal_onset", onset_cursor))
            chord = sorted(
                {
                    int(round(float(pitch)))
                    for pitch in self.hsmm.chord_pitches[position]
                    if np.isfinite(pitch)
                }
            )
            for chord_offset, pitch in enumerate(chord):
                event_pitches.append(float(pitch))
                event_score_positions.append(int(position))
                event_onsets.append(onset + (chord_offset * self._ANCHOR_INTRA_CHORD_GAP))
            score_end_event_positions[position] = len(event_pitches) - 1
            onset_cursor = onset + float(note["nominal_duration"])
        return (
            np.asarray(event_pitches, dtype=np.float64),
            np.asarray(event_score_positions, dtype=np.int64),
            np.asarray(event_onsets, dtype=np.float64),
            score_end_event_positions,
        )

    def _append_observation(self, pitches: np.ndarray, timestamp: float) -> None:
        anchor_pitches = np.unique(np.asarray(pitches, dtype=np.float64).reshape(-1))
        if anchor_pitches.size == 0:
            return

        self._observed_event_count += int(anchor_pitches.size)
        for chord_offset, pitch in enumerate(anchor_pitches.tolist()):
            self._observed_pitches.append(float(pitch))
            self._observed_timestamps.append(
                float(timestamp) + (chord_offset * self._ANCHOR_INTRA_CHORD_GAP)
            )

        overflow = len(self._observed_pitches) - self._max_anchor_window
        if overflow > 0:
            del self._observed_pitches[:overflow]
            del self._observed_timestamps[:overflow]

    def _interval_windows(self, window_length: int) -> np.ndarray:
        cached = self._interval_window_cache.get(window_length)
        if cached is None:
            cached = np.lib.stride_tricks.sliding_window_view(
                self.anchor_event_intervals,
                window_length,
            )
            self._interval_window_cache[window_length] = cached
        return cached

    def _pitch_costs_for_observation_window(self, observed_pitches: np.ndarray) -> np.ndarray:
        window_length = int(observed_pitches.size)
        num_windows = self.anchor_event_pitches.size - window_length + 1
        pitch_costs = np.zeros(num_windows, dtype=np.float64)

        for offset, observed_pitch in enumerate(observed_pitches):
            score_pitches = self.anchor_event_pitches[offset : offset + num_windows]
            deltas = np.abs(score_pitches - float(observed_pitch))
            pitch_costs += np.minimum(deltas, self.anchor_pitch_clip)

        return pitch_costs / max(1, window_length)

    def _sequence_anchor_target(self) -> int | None:
        if self._anchor_search_disabled or self._observed_event_count > self.anchor_search_max_events:
            return None

        history_length = len(self._observed_pitches)
        if history_length < self.anchor_window_lengths[-1]:
            return None

        current_position = int(self._current_index)
        lagging = (current_position + 4) < history_length

        candidate_records: list[tuple[int, float, int]] = []
        local_support_costs: list[float] = []

        for window_length in self.anchor_window_lengths:
            if history_length < window_length or window_length > self.N:
                continue

            observed_pitches = np.asarray(
                self._observed_pitches[-window_length:],
                dtype=np.float64,
            )
            observed_timestamps = np.asarray(
                self._observed_timestamps[-window_length:],
                dtype=np.float64,
            )

            pitch_costs = self._pitch_costs_for_observation_window(observed_pitches)

            total_costs = pitch_costs.copy()
            if window_length > 1:
                observed_intervals = np.diff(observed_timestamps)
                if np.any(observed_intervals > 1e-9):
                    interval_windows = self._interval_windows(window_length - 1)
                    denominator = np.sum(interval_windows * interval_windows, axis=1)
                    numerator = np.sum(interval_windows * observed_intervals[None, :], axis=1)
                    tempo_scale = np.ones_like(numerator)
                    valid = denominator > 1e-9
                    tempo_scale[valid] = numerator[valid] / denominator[valid]
                    tempo_scale = np.clip(
                        tempo_scale,
                        self.anchor_min_tempo_scale,
                        self.anchor_max_tempo_scale,
                    )
                    fitted_intervals = interval_windows * tempo_scale[:, None]
                    interval_denominator = np.maximum(0.03, observed_intervals)[None, :]
                    time_costs = np.mean(
                        np.abs(fitted_intervals - observed_intervals[None, :]) / interval_denominator,
                        axis=1,
                    )
                    total_costs += self.anchor_time_weight * time_costs

            candidate_starts = np.arange(total_costs.size, dtype=np.int64)
            best_start = int(np.argmin(total_costs))
            best_cost = float(total_costs[best_start])
            current_event_position = int(self.anchor_score_end_event_positions[current_position])
            local_start = max(0, min(current_event_position - window_length + 1, total_costs.size - 1))
            local_cost = float(total_costs[local_start])
            max_cost = self.anchor_total_cost_threshold
            if window_length <= 12:
                max_cost = min(max_cost, 1.00)
            if np.isfinite(local_cost) and local_cost <= max_cost:
                local_support_costs.append(local_cost)
            if not np.isfinite(best_cost) or best_cost > max_cost:
                continue

            separation = np.abs(candidate_starts - best_start) >= max(2, window_length // 4)
            if np.any(separation):
                second_cost = float(np.min(total_costs[separation]))
            else:
                second_cost = float("inf")
            margin = second_cost - best_cost
            if np.isfinite(second_cost) and margin < self.anchor_margin_threshold:
                continue

            if (not lagging) and ((local_cost - best_cost) < self.anchor_local_improvement_threshold):
                continue

            target_position = int(
                self.anchor_event_score_positions[best_start + window_length - 1]
            )
            if abs(target_position - current_position) < self.anchor_min_jump:
                continue

            candidate_records.append((int(target_position), best_cost, int(window_length)))

        selected_target, selected_cost, selected_window = self._select_anchor_candidate(
            candidate_records,
            local_support_costs,
        )
        if selected_target is None:
            return None

        self.last_anchor_cost = selected_cost
        self.last_anchor_window = selected_window
        return self._confirm_anchor_candidate(selected_target)

    def _select_anchor_candidate(
        self,
        candidate_records: list[tuple[int, float, int]],
        local_support_costs: list[float],
    ) -> tuple[int | None, float | None, int]:
        if not candidate_records:
            return None, None, 0

        clusters: list[dict[str, list[float] | list[int]]] = []
        for target_position, best_cost, window_length in candidate_records:
            target_value = int(target_position)
            assigned_cluster: dict[str, list[float] | list[int]] | None = None
            for cluster in clusters:
                cluster_targets = cluster["targets"]
                representative_target = int(round(float(np.median(cluster_targets))))
                if abs(target_value - representative_target) <= self.anchor_stability_tolerance:
                    assigned_cluster = cluster
                    break

            if assigned_cluster is None:
                assigned_cluster = {
                    "targets": [],
                    "costs": [],
                    "windows": [],
                }
                clusters.append(assigned_cluster)

            assigned_cluster["targets"].append(target_value)
            assigned_cluster["costs"].append(float(best_cost))
            assigned_cluster["windows"].append(int(window_length))

        summarized_clusters: list[tuple[int, float, int, int, float]] = []
        for cluster in clusters:
            targets = [int(value) for value in cluster["targets"]]
            costs = [float(value) for value in cluster["costs"]]
            windows = [int(value) for value in cluster["windows"]]
            representative_target = int(round(float(np.median(targets))))
            mean_cost = float(np.mean(costs))
            support_count = len(windows)
            max_window = max(windows)
            summarized_clusters.append(
                (representative_target, mean_cost, max_window, support_count, min(costs))
            )

        summarized_clusters.sort(
            key=lambda item: (
                -item[3],       # support_count desc
                item[1],        # mean_cost asc
                -item[2],       # max_window desc
                item[4],        # min_cost asc
            ),
        )
        best_target, best_mean_cost, best_window, best_support_count, best_min_cost = summarized_clusters[0]

        if best_support_count < self.anchor_min_supporting_windows:
            return None, None, 0

        local_support_count = len(local_support_costs)
        if local_support_count > best_support_count:
            return None, None, 0
        if local_support_count == best_support_count and local_support_count > 0:
            local_mean_cost = float(np.mean(local_support_costs))
            if local_mean_cost <= (best_mean_cost + self.anchor_local_preference_margin):
                return None, None, 0

        return best_target, best_min_cost, best_window

    def _confirm_anchor_candidate(self, target_position: int | None) -> int | None:
        if target_position is None:
            self._anchor_candidate_target = None
            self._anchor_candidate_streak = 0
            return None

        if (
            self._anchor_candidate_target is not None
            and abs(int(target_position) - self._anchor_candidate_target) <= self.anchor_stability_tolerance
        ):
            self._anchor_candidate_target = int(target_position)
            self._anchor_candidate_streak += 1
        else:
            self._anchor_candidate_target = int(target_position)
            self._anchor_candidate_streak = 1

        if self._anchor_candidate_streak < self.anchor_confirmation_events:
            return None

        confirmed_target = int(self._anchor_candidate_target)
        self._anchor_candidate_target = None
        self._anchor_candidate_streak = 0
        return confirmed_target

    def _should_resync(
        self,
        recovery_target: int | None,
        anchor_target: int | None,
    ) -> tuple[bool, bool]:
        if recovery_target is None:
            self._recovery_signal_streak = 0
            return False, False

        if anchor_target is not None and recovery_target == anchor_target:
            self._recovery_signal_streak = 0
            return True, True

        hsmm_position = self.hsmm.current_state_position
        oltw_position = self.oltw.current_state_position
        gap = oltw_position - hsmm_position

        if gap > 0 and self.oltw.last_forced_advance:
            self._recovery_signal_streak = 0
            return True, False

        self._recovery_signal_streak = 0
        return True, False

    def _forward_match_target_position(self) -> int | None:
        target = int(self.hsmm.last_best_match_position)
        current_position = int(self.hsmm.current_state_position)
        oltw_position = int(self.oltw.current_state_position)
        gap = target - current_position
        if gap < self.resync_gap:
            return None
        if gap > self.max_forward_match_gap:
            return None
        if target > (oltw_position + self.max_forward_match_lead_over_oltw):
            return None
        if float(self.hsmm.last_best_pitch_distance) > 0.75:
            return None
        return target

    def _resync_target_position(self, anchor_target: int | None) -> int | None:
        if anchor_target is not None:
            return int(anchor_target)

        hsmm_position = int(self.hsmm.current_state_position)
        oltw_position = int(self.oltw.current_state_position)
        candidates: list[int] = []

        if (oltw_position - hsmm_position) >= self.resync_gap:
            candidates.append(oltw_position)

        forward_match_target = self._forward_match_target_position()
        if forward_match_target is not None:
            candidates.append(int(forward_match_target))

        if not candidates:
            return None

        target_position = max(candidates)
        return min(target_position, int(self._current_index) + self.max_forward_step)

    def _should_prefer_hsmm(self) -> bool:
        if self.last_resynced:
            return True

        if self.hsmm.current_state_position <= self.oltw.current_state_position:
            return False

        if (self.hsmm.current_state_position - int(self._current_index)) > self.max_forward_step:
            return False

        if self.hsmm.last_best_match_position != self.hsmm.current_state_position:
            return False

        return float(self.hsmm.last_best_pitch_distance) <= 0.75

    def _nudge_hsmm_to_position(
        self,
        target_position: int,
        timestamp: float,
        *,
        allow_large_jump: bool = False,
    ) -> None:
        capped_target = int(target_position)
        if not allow_large_jump:
            capped_target = min(capped_target, int(self._current_index) + self.max_forward_step)
        target_position = int(np.clip(capped_target, 0, self.hsmm.N - 1))

        alpha = np.zeros_like(self.hsmm.alpha)
        alpha[target_position] = self.nudge_target_mass

        residual_mass = 1.0 - self.nudge_target_mass
        if residual_mass > 0.0:
            neighbor_offsets = (-1, 1)
            neighbor_weights = np.asarray([0.65, 0.35], dtype=np.float64)
            neighbor_weights /= neighbor_weights.sum()

            for offset, weight in compat_zip(neighbor_offsets, neighbor_weights, strict=True):
                position = target_position + offset
                if 0 <= position < self.hsmm.N:
                    alpha[position] += residual_mass * float(weight)
                else:
                    alpha[target_position] += residual_mass * float(weight)

        alpha_sum = float(alpha.sum())
        if not np.isfinite(alpha_sum) or alpha_sum <= 0.0:
            alpha.fill(0.0)
            alpha[target_position] = 1.0
        else:
            alpha /= alpha_sum

        self.hsmm.alpha = alpha
        self.hsmm.current_state_position = target_position
        self.hsmm.current_state_index = int(self.hsmm.state_indices[target_position])
        self.hsmm.current_state_start_time = float(timestamp)
        self.hsmm.last_timestamp = float(timestamp)
        self.hsmm.last_elapsed_time = 0.0
        self.hsmm.last_scale = 1.0
        self.hsmm.last_transition_probabilities = {
            "stay": 1.0,
            "advance": 0.0,
            "skip": 0.0,
        }
        self.hsmm._has_seen_event = True

    def _limit_forward_step(
        self,
        selected_index: int,
        previous_output_index: int,
        *,
        allow_large_jump: bool = False,
    ) -> int:
        if allow_large_jump:
            return int(selected_index)
        selected_index = int(selected_index)
        previous_output_index = int(previous_output_index)
        if selected_index < previous_output_index:
            if not self.allow_backward_output:
                return previous_output_index
            return selected_index
        capped_index = min(selected_index, previous_output_index + self.max_forward_step)
        return capped_index

    def _debounce_output_index(
        self,
        proposed_index: int,
        *,
        anchor_resync: bool,
    ) -> int:
        proposed_index = int(proposed_index)
        stable_index = int(self._stable_output_index)

        if proposed_index == stable_index:
            self._candidate_output_index = None
            self._candidate_output_streak = 0
            return stable_index

        immediate_commit = (
            anchor_resync
            or self.last_resynced
            or self.confidence >= self.output_high_confidence
        )
        if immediate_commit:
            self._stable_output_index = proposed_index
            self._candidate_output_index = None
            self._candidate_output_streak = 0
            return proposed_index

        if proposed_index < stable_index:
            if (
                self._candidate_output_index is not None
                and self._candidate_output_index < stable_index
                and proposed_index <= self._candidate_output_index
            ):
                self._candidate_output_index = proposed_index
                self._candidate_output_streak += 1
            else:
                self._candidate_output_index = proposed_index
                self._candidate_output_streak = 1

            if self._candidate_output_streak >= self.output_confirmation_events:
                committed_index = int(self._candidate_output_index)
                self._stable_output_index = committed_index
                self._candidate_output_index = None
                self._candidate_output_streak = 0
                return committed_index

            return stable_index

        if self._candidate_output_index == proposed_index:
            self._candidate_output_streak += 1
        else:
            self._candidate_output_index = proposed_index
            self._candidate_output_streak = 1

        if self._candidate_output_streak >= self.output_confirmation_events:
            self._stable_output_index = proposed_index
            self._candidate_output_index = None
            self._candidate_output_streak = 0
            return proposed_index

        return stable_index


if __name__ == "__main__":
    score_path = Path(__file__).resolve().parent / "generated_dataset" / "ideal.json"
    follower = HybridScoreFollower(score_path)

    repeated_pitch = int(follower.hsmm.chord_pitches[0][0])
    timestamps = [0.05 * step for step in range(10)]

    print(f"score={score_path}")
    print(
        "event  pitch  hsmm_idx  conf   oltw_idx  forced  resync  selected  hybrid_idx"
    )

    for event_number, timestamp in enumerate(timestamps, start=1):
        hybrid_index = follower.process_event(repeated_pitch, timestamp)
        print(
            f"{event_number:>5}  "
            f"{repeated_pitch:>5}  "
            f"{follower.last_hsmm_index:>8}  "
            f"{follower.confidence:>5.3f}  "
            f"{follower.last_oltw_index:>8}  "
            f"{str(follower.oltw.last_forced_advance):>6}  "
            f"{str(follower.last_resynced):>6}  "
            f"{follower.last_selected_model:>8}  "
            f"{hybrid_index:>10}"
        )
