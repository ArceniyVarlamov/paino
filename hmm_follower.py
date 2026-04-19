from __future__ import annotations

import json
import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import numpy as np


class ScoreFollowerHMM:
    """Realtime score-following with a duration-aware Forward update."""

    BASE_P_STAY = 0.3
    BASE_P_ADVANCE = 0.6
    BASE_P_SKIP = 0.1

    def __init__(self, score_json_path: str | Path, sigma: float = 2.0) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        score_path = Path(score_json_path)
        if score_path.suffix.lower() in {".mid", ".midi"}:
            raise ValueError(
                "ScoreFollowerHMM expects a score JSON file, not a MIDI file. "
                "Convert the MIDI with `midi_to_score.py` and pass the resulting `.json`."
            )

        try:
            score_data = json.loads(score_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Could not read score file as UTF-8 JSON: {score_path}. "
                "Pass a `.json` score file generated for the HMM."
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid score JSON: {score_path}. "
                "Expected the project score format with a top-level `notes` list."
            ) from exc
        notes = score_data.get("notes")

        if not isinstance(notes, list) or not notes:
            raise ValueError("score JSON must contain a non-empty 'notes' list")

        self.score_path = score_path
        self.score_data = score_data
        self.N = len(notes)
        self.sigma = float(sigma)

        self.pitches = np.asarray(
            [float(note["pitch"]) for note in notes],
            dtype=np.float64,
        )
        self.nominal_durations = np.maximum(
            np.asarray(
                [float(note["nominal_duration"]) for note in notes],
                dtype=np.float64,
            ),
            1e-6,
        )

        self.alpha = np.zeros(self.N, dtype=np.float64)
        self.alpha[0] = 1.0

        self.current_index = 0
        self.current_state_started_at: float | None = None
        self.last_timestamp: float | None = None
        self.seen_event = False
        self.last_transition_probabilities = (
            self.BASE_P_STAY,
            self.BASE_P_ADVANCE,
            self.BASE_P_SKIP,
        )

        self._gaussian_norm = 1.0 / (self.sigma * np.sqrt(2.0 * np.pi))
        self._tiny = np.finfo(np.float64).tiny

    def process_event(self, event: dict) -> int:
        """Update the state distribution from one realtime MIDI note event."""
        pitch = float(event["pitch"])
        timestamp = float(event["timestamp"])

        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            timestamp = self.last_timestamp

        if self.current_state_started_at is None:
            self.current_state_started_at = timestamp

        emission = self._emission_probabilities(pitch)

        if self.seen_event:
            elapsed = max(0.0, timestamp - self.current_state_started_at)
            expected = float(self.nominal_durations[self.current_index])
            transition_probabilities = self._transition_probabilities(elapsed, expected)
            prior = self._apply_banded_transition(self.alpha, transition_probabilities)
            self.last_transition_probabilities = transition_probabilities
        else:
            # The first observation is the HMM initialization step alpha_1(i) = pi_i * b_i(o_1).
            prior = self.alpha.copy()
            self.last_transition_probabilities = (1.0, 0.0, 0.0)

        new_alpha = prior * emission
        normalizer = float(new_alpha.sum())

        if not np.isfinite(normalizer) or normalizer <= 0.0:
            prior_sum = float(prior.sum())
            if not np.isfinite(prior_sum) or prior_sum <= 0.0:
                new_alpha = np.zeros_like(self.alpha)
                new_alpha[self.current_index] = 1.0
            else:
                new_alpha = prior / prior_sum
        else:
            new_alpha /= normalizer

        self.alpha = new_alpha

        predicted_index = int(np.argmax(self.alpha))
        if predicted_index != self.current_index:
            self.current_index = predicted_index
            self.current_state_started_at = timestamp

        self.last_timestamp = timestamp
        self.seen_event = True
        return predicted_index

    def _emission_probabilities(self, observed_pitch: float) -> np.ndarray:
        deltas = (observed_pitch - self.pitches) / self.sigma
        emission = self._gaussian_norm * np.exp(-0.5 * np.square(deltas))
        return np.maximum(emission, self._tiny)

    def _transition_probabilities(
        self,
        elapsed_time: float,
        expected_duration: float,
    ) -> tuple[float, float, float]:
        if elapsed_time > 1.5 * expected_duration:
            advance_probability = 0.95
        elif elapsed_time >= expected_duration:
            advance_probability = 0.8
        else:
            advance_probability = self.BASE_P_ADVANCE

        remaining_probability = max(0.0, 1.0 - advance_probability)
        base_remaining = self.BASE_P_STAY + self.BASE_P_SKIP

        stay_probability = remaining_probability * (self.BASE_P_STAY / base_remaining)
        skip_probability = remaining_probability * (self.BASE_P_SKIP / base_remaining)

        return stay_probability, advance_probability, skip_probability

    def _apply_banded_transition(
        self,
        alpha: np.ndarray,
        transition_probabilities: tuple[float, float, float],
    ) -> np.ndarray:
        stay_probability, advance_probability, skip_probability = transition_probabilities
        prior = alpha * stay_probability

        if self.N > 1:
            prior[1:] += alpha[:-1] * advance_probability
            prior[-1] += alpha[-1] * (advance_probability + skip_probability)
            prior[-1] += alpha[-2] * skip_probability
        else:
            prior[0] += alpha[0] * (advance_probability + skip_probability)

        if self.N > 2:
            prior[2:] += alpha[:-2] * skip_probability

        return prior


if __name__ == "__main__":
    from dataset_viewer import load_performance

    dataset_dir = Path(__file__).resolve().parent / "generated_dataset"
    score_path = dataset_dir / "ideal.json"
    midi_path = dataset_dir / "ideal.mid"

    follower = ScoreFollowerHMM(score_path)
    performance = load_performance(midi_path)
    predictions = [follower.process_event(event) for event in performance]
    expected = list(range(follower.N))

    for event, predicted_index in zip(performance, predictions, strict=True):
        score_pitch = int(follower.pitches[predicted_index])
        print(
            f"t={event['timestamp']:.3f}s pitch={int(event['pitch']):>3} "
            f"-> state={predicted_index:>2} score_pitch={score_pitch:>3}"
        )

    print(f"predictions: {predictions}")
    print(f"expected   : {expected}")

    if predictions != expected:
        raise SystemExit("HMM demo failed to track the ideal score correctly.")

    print("HMM demo tracked the ideal score from start to finish.")
