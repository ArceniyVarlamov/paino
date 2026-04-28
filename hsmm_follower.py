from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import numpy as np


class ScoreFollowerHSMM:
    """Realtime score-following with duration-dependent transitions.

    The model keeps one hidden state per score note and updates a scaled Forward
    distribution in realtime. Transition probabilities are recomputed for each
    incoming MIDI event from the elapsed time spent in the current best state.
    """

    _MIN_DURATION = 1e-6
    _TINY = np.finfo(np.float64).tiny

    def __init__(
        self,
        score_json: str | Path | dict[str, Any] | list[dict[str, Any]],
        sigma: float = 2.5,
        outlier_pitch_clip: float = 6.0,
    ) -> None:
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if outlier_pitch_clip <= 0.0:
            raise ValueError("outlier_pitch_clip must be positive")

        score_data, notes = self._load_score(score_json)
        if not notes:
            raise ValueError("score_json must contain at least one note state")

        self.score_data = score_data
        self.notes = notes
        self.num_states = len(notes)
        self.N = self.num_states
        self.sigma = float(sigma)
        self.outlier_pitch_clip = float(outlier_pitch_clip)

        self.state_indices = np.asarray(
            [int(note.get("index", position)) for position, note in enumerate(notes)],
            dtype=np.int64,
        )
        self.chord_pitches = tuple(
            np.asarray(self._note_pitches(note), dtype=np.float64) for note in notes
        )
        self.max_chord_size = max(chord.size for chord in self.chord_pitches)
        self.chord_pitch_matrix = np.full(
            (self.num_states, self.max_chord_size),
            np.nan,
            dtype=np.float64,
        )
        for position, chord in enumerate(self.chord_pitches):
            self.chord_pitch_matrix[position, : chord.size] = chord
        self.pitches = np.nanmax(self.chord_pitch_matrix, axis=1)
        self.nominal_durations = np.maximum(
            np.asarray(
                [float(note["nominal_duration"]) for note in notes],
                dtype=np.float64,
            ),
            self._MIN_DURATION,
        )

        self.alpha = np.zeros(self.num_states, dtype=np.float64)
        self.alpha[0] = 1.0

        self.current_state_position = 0
        self.current_state_index = int(self.state_indices[0])
        self.current_state_start_time: float | None = None
        self.last_timestamp: float | None = None
        self.last_elapsed_time = 0.0
        self.last_scale = 1.0
        self.last_transition_probabilities = {
            "stay": 1.0,
            "advance": 0.0,
            "skip": 0.0,
            "leap": 0.0,
        }
        self.last_best_match_position = 0
        self.last_best_pitch_distance = 0.0
        self._has_seen_event = False

    def process_event(self, pitch: Any, timestamp: float) -> int:
        """Consume one MIDI note event and return the most likely score index."""
        observed_pitches = self._coerce_observed_pitches(pitch)
        event_time = float(timestamp)

        if self.last_timestamp is not None and event_time < self.last_timestamp:
            event_time = self.last_timestamp

        if self.current_state_start_time is None:
            self.current_state_start_time = event_time

        emission, pitch_distance = self._emission_probabilities(observed_pitches)
        best_matching_position = int(np.argmax(emission))
        best_pitch_distance = float(pitch_distance[best_matching_position])
        self.last_best_match_position = best_matching_position
        self.last_best_pitch_distance = best_pitch_distance

        if self._has_seen_event:
            elapsed_time = max(0.0, event_time - self.current_state_start_time)
            prior = self._predict_prior(
                elapsed_time,
                best_matching_position=best_matching_position,
                best_pitch_distance=best_pitch_distance,
            )
            self.last_elapsed_time = elapsed_time
        else:
            prior = self.alpha.copy()
            self.last_elapsed_time = 0.0
            self.last_transition_probabilities = {
                "stay": 1.0,
                "advance": 0.0,
                "skip": 0.0,
                "leap": 0.0,
            }

        updated_alpha = prior * emission
        scale = float(updated_alpha.sum())

        if not np.isfinite(scale) or scale <= self._TINY:
            prior_sum = float(prior.sum())
            if np.isfinite(prior_sum) and prior_sum > self._TINY:
                updated_alpha = prior / prior_sum
                scale = prior_sum
            else:
                updated_alpha = np.zeros_like(self.alpha)
                updated_alpha[self.current_state_position] = 1.0
                scale = 1.0
        else:
            updated_alpha /= scale

        previous_position = self.current_state_position

        self.alpha = updated_alpha
        self.last_scale = scale
        self.last_timestamp = event_time
        self._has_seen_event = True

        predicted_position = int(np.argmax(self.alpha))
        self.current_state_position = predicted_position
        self.current_state_index = int(self.state_indices[predicted_position])

        if predicted_position != previous_position or self.current_state_start_time is None:
            self.current_state_start_time = event_time

        return self.current_state_index

    def seek(self, position: int, timestamp: float | None = None) -> int:
        """Force the follower to a specific score position."""
        target_position = int(np.clip(position, 0, self.N - 1))
        event_time = float(self.last_timestamp if timestamp is None else timestamp)

        self.alpha.fill(0.0)
        self.alpha[target_position] = 1.0
        self.current_state_position = target_position
        self.current_state_index = int(self.state_indices[target_position])
        self.current_state_start_time = event_time
        self.last_timestamp = event_time
        self.last_elapsed_time = 0.0
        self.last_scale = 1.0
        self.last_transition_probabilities = {
            "stay": 1.0,
            "advance": 0.0,
            "skip": 0.0,
            "leap": 0.0,
        }
        self.last_best_match_position = target_position
        self.last_best_pitch_distance = 0.0
        self._has_seen_event = True
        return self.current_state_index

    def reset_to_start(self) -> int:
        """Reset the HSMM back to its initial start-of-score state."""
        self.alpha.fill(0.0)
        self.alpha[0] = 1.0
        self.current_state_position = 0
        self.current_state_index = int(self.state_indices[0])
        self.current_state_start_time = None
        self.last_timestamp = None
        self.last_elapsed_time = 0.0
        self.last_scale = 1.0
        self.last_transition_probabilities = {
            "stay": 1.0,
            "advance": 0.0,
            "skip": 0.0,
            "leap": 0.0,
        }
        self.last_best_match_position = 0
        self.last_best_pitch_distance = 0.0
        self._has_seen_event = False
        return self.current_state_index

    @classmethod
    def _load_score(
        cls,
        score_json: str | Path | dict[str, Any] | list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if isinstance(score_json, (str, Path)):
            score_path = Path(score_json)
            if score_path.suffix.lower() in {".mid", ".midi"}:
                raise ValueError(
                    "ScoreFollowerHSMM expects a score JSON file, not a MIDI file."
                )

            try:
                payload = json.loads(score_path.read_text(encoding="utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError(f"Could not decode score JSON: {score_path}") from exc
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid score JSON: {score_path}") from exc
        else:
            payload = score_json

        if isinstance(payload, list):
            notes = payload
            score_data = {"notes": notes}
        elif isinstance(payload, dict):
            notes = payload.get("notes")
            score_data = payload
        else:
            raise TypeError("score_json must be a path, a score dict, or a list of notes")

        if not isinstance(notes, list):
            raise ValueError("score_json must contain a top-level list of notes")

        for position, note in enumerate(notes):
            if not isinstance(note, dict):
                raise ValueError(f"score note #{position} must be a JSON object")
            if "pitch" not in note and "pitches" not in note:
                raise ValueError(f"score note #{position} is missing 'pitch'/'pitches'")
            if "nominal_duration" not in note:
                raise ValueError(f"score note #{position} is missing 'nominal_duration'")

        return score_data, notes

    @staticmethod
    def _note_pitches(note: dict[str, Any]) -> list[float]:
        raw_pitches = note.get("pitches")
        if raw_pitches is None:
            raw_pitch = note.get("pitch")
            if raw_pitch is None:
                raise ValueError("score note is missing 'pitch'/'pitches'")
            return [float(raw_pitch)]

        if not isinstance(raw_pitches, list) or not raw_pitches:
            raise ValueError("score note 'pitches' must be a non-empty list")
        return [float(pitch) for pitch in raw_pitches]

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

    def _emission_probabilities(self, observed_pitches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pitch_delta = np.abs(
            self.chord_pitch_matrix[:, :, None] - observed_pitches[None, None, :]
        )
        pitch_delta = np.where(np.isnan(self.chord_pitch_matrix[:, :, None]), np.inf, pitch_delta)
        observed_to_state = np.min(pitch_delta, axis=1)
        pitch_distance = np.min(observed_to_state, axis=1)
        pitch_delta = np.minimum(pitch_distance, self.outlier_pitch_clip)
        emission = np.exp(-(pitch_delta * pitch_delta) / (2.0 * self.sigma * self.sigma))
        return np.maximum(emission, self._TINY), pitch_distance

    def _predict_prior(
        self,
        elapsed_time: float,
        *,
        best_matching_position: int,
        best_pitch_distance: float,
    ) -> np.ndarray:
        expected_duration = float(self.nominal_durations[self.current_state_position])
        (
            stay_probability,
            advance_probability,
            skip_probability,
            leap_probability,
        ) = self._transition_probabilities(
            elapsed_time=elapsed_time,
            expected_duration=expected_duration,
            current_position=self.current_state_position,
            best_matching_position=best_matching_position,
            best_pitch_distance=best_pitch_distance,
        )

        self.last_transition_probabilities = {
            "stay": stay_probability,
            "advance": advance_probability,
            "skip": skip_probability,
            "leap": leap_probability,
        }

        prior = self.alpha * stay_probability

        if self.num_states == 1:
            prior[0] += self.alpha[0] * (advance_probability + skip_probability + leap_probability)
            return prior

        prior[1:] += self.alpha[:-1] * advance_probability

        if self.num_states > 2:
            prior[2:] += self.alpha[:-2] * skip_probability

        if self.num_states > 3:
            prior[3:] += self.alpha[:-3] * leap_probability

        # Fold overflow mass into the final note so total probability stays conserved.
        prior[-1] += self.alpha[-1] * (advance_probability + skip_probability + leap_probability)
        prior[-1] += self.alpha[-2] * skip_probability
        if self.num_states > 2:
            prior[-1] += self.alpha[-3] * leap_probability
            prior[-1] += self.alpha[-2] * leap_probability
        return prior

    @staticmethod
    def _transition_probabilities(
        elapsed_time: float,
        expected_duration: float,
        current_position: int,
        best_matching_position: int,
        best_pitch_distance: float,
    ) -> tuple[float, float, float, float]:
        safe_duration = max(float(expected_duration), ScoreFollowerHSMM._MIN_DURATION)
        ratio = max(0.0, float(elapsed_time)) / safe_duration
        forward_gap = max(0, int(best_matching_position) - int(current_position))
        strong_forward_match = forward_gap >= 2 and best_pitch_distance <= 0.75

        if ratio < 1.0:
            stay_probability = 0.82
            advance_probability = 0.18
            skip_probability = 0.0
            leap_probability = 0.0
        elif ratio <= 1.5:
            phase = (ratio - 1.0) / 0.5
            stay_probability = 0.35 + (0.18 - 0.35) * phase
            advance_probability = 0.60 + (0.70 - 0.60) * phase
            skip_probability = 0.03 + (0.06 - 0.03) * phase
            leap_probability = 0.02 + (0.06 - 0.02) * phase
        else:
            stay_probability = 0.12
            advance_probability = 0.68
            skip_probability = 0.08
            leap_probability = 0.12

        if strong_forward_match:
            if forward_gap >= 3:
                stay_probability = 0.03 if ratio >= 1.0 else 0.08
                advance_probability = 0.17 if ratio >= 1.0 else 0.22
                skip_probability = 0.25 if ratio >= 1.0 else 0.20
                leap_probability = 0.55 if ratio >= 1.0 else 0.50
            else:
                stay_probability = min(stay_probability, 0.10 if ratio >= 1.0 else 0.18)
                advance_probability = max(advance_probability, 0.20 if ratio >= 1.0 else 0.25)
                skip_probability = max(skip_probability, 0.55 if ratio >= 1.0 else 0.45)
                leap_probability = max(leap_probability, 0.15 if ratio >= 1.0 else 0.12)

        total = stay_probability + advance_probability + skip_probability + leap_probability
        if not np.isfinite(total) or total <= 0.0:
            return 0.0, 1.0, 0.0, 0.0

        return (
            stay_probability / total,
            advance_probability / total,
            skip_probability / total,
            leap_probability / total,
        )
