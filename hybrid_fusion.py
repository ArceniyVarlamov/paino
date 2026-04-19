from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import numpy as np

from hsmm_follower import ScoreFollowerHSMM
from oltw_follower import ScoreFollowerOLTW


class HybridScoreFollower:
    """Fuse HSMM confidence with OLTW recovery behavior."""

    def __init__(
        self,
        score_json: str | Path | dict[str, Any] | list[dict[str, Any]],
        *,
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
        anchor_window_lengths: tuple[int, ...] = (20, 16, 12),
        anchor_pitch_clip: float = 6.0,
        anchor_total_cost_threshold: float = 1.35,
        anchor_margin_threshold: float = 0.05,
        anchor_time_weight: float = 1.25,
        anchor_min_tempo_scale: float = 0.35,
        anchor_max_tempo_scale: float = 3.50,
        anchor_local_improvement_threshold: float = 0.35,
        anchor_search_max_events: int = 100_000,
        anchor_confirmation_events: int = 1,
        anchor_stability_tolerance: int = 2,
        anchor_min_jump: int = 8,
        output_confirmation_events: int = 1,
        output_high_confidence: float = 0.4,
    ) -> None:
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
        self.output_confirmation_events = int(output_confirmation_events)
        self.output_high_confidence = float(output_high_confidence)
        self.score_data = self.hsmm.score_data
        self.pitches = self.hsmm.pitches
        self.chord_pitch_matrix = self.hsmm.chord_pitch_matrix
        self.N = self.hsmm.N
        self.nominal_onsets = self._extract_nominal_onsets()
        self.nominal_intervals = np.diff(self.nominal_onsets)
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

    def process_event(self, pitch: int | float, timestamp: float) -> int:
        """Process one observation and return the fused score index."""
        event_pitch = float(pitch)
        event_time = float(timestamp)
        if self._last_input_timestamp is not None and event_time < self._last_input_timestamp:
            event_time = self._last_input_timestamp
        self._last_input_timestamp = event_time
        self._append_observation(event_pitch, event_time)

        hsmm_index = int(self.hsmm.process_event(event_pitch, event_time))
        oltw_index = int(self.oltw.process_event(event_pitch, event_time))

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

        self._current_index = int(selected_index)
        self._stable_output_index = int(selected_index)
        self._candidate_output_index = None
        self._candidate_output_streak = 0
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

    def _extract_nominal_onsets(self) -> np.ndarray:
        onsets = np.zeros(self.N, dtype=np.float64)
        onset_cursor = 0.0
        for position, note in enumerate(self.hsmm.notes):
            onset = float(note.get("nominal_onset", onset_cursor))
            onsets[position] = onset
            onset_cursor = onset + float(note["nominal_duration"])
        return onsets

    def _append_observation(self, pitch: float, timestamp: float) -> None:
        self._observed_event_count += 1
        self._observed_pitches.append(float(pitch))
        self._observed_timestamps.append(float(timestamp))

        if len(self._observed_pitches) > self._max_anchor_window:
            self._observed_pitches.pop(0)
            self._observed_timestamps.pop(0)

    def _interval_windows(self, window_length: int) -> np.ndarray:
        cached = self._interval_window_cache.get(window_length)
        if cached is None:
            cached = np.lib.stride_tricks.sliding_window_view(
                self.nominal_intervals,
                window_length,
            )
            self._interval_window_cache[window_length] = cached
        return cached

    def _pitch_costs_for_observation_window(self, observed_pitches: np.ndarray) -> np.ndarray:
        window_length = int(observed_pitches.size)
        num_windows = self.N - window_length + 1
        pitch_costs = np.zeros(num_windows, dtype=np.float64)

        for offset, observed_pitch in enumerate(observed_pitches):
            score_rows = self.chord_pitch_matrix[offset : offset + num_windows]
            deltas = np.abs(score_rows - float(observed_pitch))
            deltas = np.where(np.isnan(score_rows), np.inf, deltas)
            pitch_costs += np.minimum(np.min(deltas, axis=1), self.anchor_pitch_clip)

        return pitch_costs / max(1, window_length)

    def _sequence_anchor_target(self) -> int | None:
        if self._anchor_search_disabled or self._observed_event_count > self.anchor_search_max_events:
            return None

        history_length = len(self._observed_pitches)
        if history_length < self.anchor_window_lengths[-1]:
            return None

        current_position = int(self._current_index)
        lagging = (current_position + 4) < history_length

        candidate_target: int | None = None

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
            max_cost = self.anchor_total_cost_threshold
            if window_length <= 12:
                max_cost = min(max_cost, 1.00)
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

            local_start = max(0, min(current_position - window_length + 1, total_costs.size - 1))
            local_cost = float(total_costs[local_start])
            if (not lagging) and ((local_cost - best_cost) < self.anchor_local_improvement_threshold):
                continue

            target_position = best_start + window_length - 1
            if abs(target_position - current_position) < self.anchor_min_jump:
                continue

            self.last_anchor_cost = best_cost
            self.last_anchor_window = window_length
            candidate_target = int(target_position)
            break

        return self._confirm_anchor_candidate(candidate_target)

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

            for offset, weight in zip(neighbor_offsets, neighbor_weights, strict=True):
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
