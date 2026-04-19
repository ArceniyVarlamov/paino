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


class ScoreFollowerOLTW:
    """Incremental DTW score follower with a RunCount fail-safe.

    The follower updates one DTW column per observation and keeps only the
    previous and current cost columns in memory. A monotonic position tracker is
    derived from the current column minimum, while RunCount forces progress if
    the tracker remains on the same score note for too many input events.
    """

    _HORIZONTAL = "horizontal"
    _DIAGONAL = "diagonal"
    _VERTICAL = "vertical"

    def __init__(
        self,
        score_json: str | Path | dict[str, Any] | list[dict[str, Any]],
        max_local_cost: float = 6.0,
    ) -> None:
        notes = self._load_notes(score_json)
        if not notes:
            raise ValueError("score_json must contain at least one note state")
        if max_local_cost <= 0.0:
            raise ValueError("max_local_cost must be positive")

        self.num_states = len(notes)
        self.N = self.num_states
        self.max_run = 3
        self.max_local_cost = float(max_local_cost)

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

        self.prev_col = np.full(self.num_states + 1, np.inf, dtype=np.float64)
        self.prev_col[0] = 0.0
        self.curr_col = np.full(self.num_states + 1, np.inf, dtype=np.float64)

        self.current_state_position = 0
        self.current_state_index = int(self.state_indices[0])
        self.run_count = 0
        self.last_direction = self._DIAGONAL
        self.last_forced_advance = False
        self.last_timestamp: float | None = None
        self.event_count = 0
        self._has_seen_event = False

    def process_event(self, pitch: int | float, timestamp: float) -> int:
        """Consume one MIDI note event and return the predicted score index."""
        observed_pitch = float(pitch)
        event_time = float(timestamp)

        if self.last_timestamp is not None and event_time < self.last_timestamp:
            event_time = self.last_timestamp

        pitch_delta = np.abs(self.chord_pitch_matrix - observed_pitch)
        pitch_delta = np.where(np.isnan(self.chord_pitch_matrix), np.inf, pitch_delta)
        local_costs = np.minimum(np.min(pitch_delta, axis=1), self.max_local_cost)
        curr_col = np.full(self.num_states + 1, np.inf, dtype=np.float64)

        for row in range(1, self.num_states + 1):
            diagonal_cost = self.prev_col[row - 1]
            horizontal_cost = self.prev_col[row]
            vertical_cost = curr_col[row - 1]

            if diagonal_cost <= horizontal_cost and diagonal_cost <= vertical_cost:
                best_predecessor = diagonal_cost
            elif horizontal_cost <= vertical_cost:
                best_predecessor = horizontal_cost
            else:
                best_predecessor = vertical_cost

            curr_col[row] = local_costs[row - 1] + best_predecessor

        raw_best_position = int(np.argmin(curr_col[1:]))
        previous_position = self.current_state_position
        predicted_position = max(previous_position, raw_best_position)

        if not self._has_seen_event:
            direction = self._DIAGONAL
            self.run_count = 0
        elif predicted_position == previous_position:
            direction = self._HORIZONTAL
            self.run_count += 1
        elif predicted_position == previous_position + 1:
            direction = self._DIAGONAL
            self.run_count = 0
        else:
            direction = self._VERTICAL
            self.run_count = 0

        forced_advance = False
        if self.run_count > self.max_run and previous_position < self.num_states - 1:
            predicted_position = previous_position + 1
            direction = self._DIAGONAL
            self.run_count = 0
            forced_advance = True

        self.curr_col = curr_col
        self.prev_col = curr_col
        self.current_state_position = predicted_position
        self.current_state_index = int(self.state_indices[predicted_position])
        self.last_direction = direction
        self.last_forced_advance = forced_advance
        self.last_timestamp = event_time
        self.event_count += 1
        self._has_seen_event = True

        return self.current_state_index

    def seek(self, position: int, timestamp: float | None = None) -> int:
        """Force the DTW tracker to resume from a specific score position."""
        target_position = int(np.clip(position, 0, self.N - 1))
        event_time = float(self.last_timestamp if timestamp is None else timestamp)

        self.prev_col.fill(np.inf)
        self.curr_col.fill(np.inf)
        self.prev_col[target_position + 1] = 0.0

        self.current_state_position = target_position
        self.current_state_index = int(self.state_indices[target_position])
        self.run_count = 0
        self.last_direction = self._DIAGONAL
        self.last_forced_advance = False
        self.last_timestamp = event_time
        self.event_count = 0
        self._has_seen_event = True
        return self.current_state_index

    def reset_to_start(self) -> int:
        """Reset the OLTW tracker back to the beginning of the score."""
        self.prev_col.fill(np.inf)
        self.curr_col.fill(np.inf)
        self.prev_col[0] = 0.0

        self.current_state_position = 0
        self.current_state_index = int(self.state_indices[0])
        self.run_count = 0
        self.last_direction = self._DIAGONAL
        self.last_forced_advance = False
        self.last_timestamp = None
        self.event_count = 0
        self._has_seen_event = False
        return self.current_state_index

    @classmethod
    def _load_notes(
        cls,
        score_json: str | Path | dict[str, Any] | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if isinstance(score_json, (str, Path)):
            score_path = Path(score_json)
            if score_path.suffix.lower() in {".mid", ".midi"}:
                raise ValueError(
                    "ScoreFollowerOLTW expects a score JSON file, not a MIDI file."
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
        elif isinstance(payload, dict):
            notes = payload.get("notes")
        else:
            raise TypeError("score_json must be a path, a score dict, or a list of notes")

        if not isinstance(notes, list):
            raise ValueError("score_json must contain a top-level list of notes")

        for position, note in enumerate(notes):
            if not isinstance(note, dict):
                raise ValueError(f"score note #{position} must be a JSON object")
            if "pitch" not in note and "pitches" not in note:
                raise ValueError(f"score note #{position} is missing 'pitch'/'pitches'")

        return notes

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
