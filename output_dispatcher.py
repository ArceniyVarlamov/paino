from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import numpy as np

try:
    import mido
except ModuleNotFoundError:
    mido = None

try:
    import pygame
    import pygame.midi
except ModuleNotFoundError:
    pygame = None

from hybrid_fusion import HybridScoreFollower

DispatchCallback = Callable[[int, float], None]


@dataclass(frozen=True)
class TempoObservation:
    nominal_elapsed: float
    actual_elapsed: float
    raw_ratio: float


@dataclass(frozen=True)
class DispatchEvent:
    index: int
    timestamp: float
    tempo_update: bool = True


def _load_score(
    score_json: str | Path | dict[str, Any] | list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(score_json, (str, Path)):
        score_path = Path(score_json)
        if score_path.suffix.lower() in {".mid", ".midi"}:
            raise ValueError("Expected a score JSON file, not a MIDI file.")

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

    if not isinstance(notes, list) or not notes:
        raise ValueError("score_json must contain a non-empty top-level list of notes")

    for position, note in enumerate(notes):
        if not isinstance(note, dict):
            raise ValueError(f"score note #{position} must be a JSON object")
        if "pitch" not in note and "pitches" not in note:
            raise ValueError(f"score note #{position} is missing 'pitch'/'pitches'")
        if "nominal_duration" not in note:
            raise ValueError(f"score note #{position} is missing 'nominal_duration'")

    return score_data, notes


def _note_pitches(note: dict[str, Any]) -> list[int]:
    raw_pitches = note.get("pitches")
    if raw_pitches is None:
        raw_pitch = note.get("pitch")
        if raw_pitch is None:
            raise ValueError("score note is missing 'pitch'/'pitches'")
        return [int(raw_pitch)]

    if not isinstance(raw_pitches, list) or not raw_pitches:
        raise ValueError("score note 'pitches' must be a non-empty list")
    return [int(pitch) for pitch in raw_pitches]


def _representative_pitch(note: dict[str, Any]) -> int:
    return max(_note_pitches(note))


def _require_mido() -> Any:
    if mido is None:
        raise RuntimeError(
            "mido is not installed. Install it or place it in the local .vendor directory."
        )
    return mido


def iter_midi_note_events(midi_path: str | Path) -> list[dict[str, float | int]]:
    midi_lib = _require_mido()
    midi_file = midi_lib.MidiFile(Path(midi_path))
    elapsed = 0.0
    events: list[dict[str, float | int]] = []

    for message in midi_file:
        elapsed += float(getattr(message, "time", 0.0))
        if getattr(message, "type", None) == "note_on" and int(getattr(message, "velocity", 0)) > 0:
            events.append(
                {
                    "pitch": int(message.note),
                    "timestamp": elapsed,
                }
            )

    return events


class TempoTracker:
    """Estimate performance tempo from score progress and event timestamps."""

    _MIN_ELAPSED = 1e-6

    def __init__(
        self,
        score_json: str | Path | dict[str, Any] | list[dict[str, Any]],
        *,
        history_size: int = 5,
        smoothing_factor: float = 1.0,
        initial_tempo_ratio: float = 1.0,
        min_tempo_ratio: float = 0.25,
        max_tempo_ratio: float = 4.0,
        deadzone_ratio: float = 0.02,
        min_nominal_window: float = 0.18,
        variance_warn_threshold: float = 0.0,
        variance_log_interval: int = 1,
        idle_reset_seconds: float = 1.5,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        if initial_tempo_ratio <= 0.0:
            raise ValueError("initial_tempo_ratio must be positive")
        if min_tempo_ratio <= 0.0 or max_tempo_ratio <= 0.0:
            raise ValueError("tempo ratio bounds must be positive")
        if min_tempo_ratio > max_tempo_ratio:
            raise ValueError("min_tempo_ratio must be <= max_tempo_ratio")
        if min_nominal_window <= 0.0:
            raise ValueError("min_nominal_window must be positive")
        if idle_reset_seconds <= 0.0:
            raise ValueError("idle_reset_seconds must be positive")

        _, notes = _load_score(score_json)
        self.state_indices = np.asarray(
            [int(note.get("index", position)) for position, note in enumerate(notes)],
            dtype=np.int64,
        )
        self.index_to_position = {
            int(score_index): position for position, score_index in enumerate(self.state_indices)
        }
        self.nominal_durations = np.maximum(
            np.asarray([float(note["nominal_duration"]) for note in notes], dtype=np.float64),
            self._MIN_ELAPSED,
        )
        self.cumulative_nominal_time = np.concatenate(
            (
                np.zeros(1, dtype=np.float64),
                np.cumsum(self.nominal_durations, dtype=np.float64),
            )
        )
        self.nominal_onsets = np.asarray(
            [
                float(note.get("nominal_onset", self.cumulative_nominal_time[position]))
                for position, note in enumerate(notes)
            ],
            dtype=np.float64,
        )

        self.history_size = int(history_size)
        self.smoothing_factor = float(smoothing_factor)
        self.min_tempo_ratio = float(min_tempo_ratio)
        self.max_tempo_ratio = float(max_tempo_ratio)
        self.deadzone_ratio = float(deadzone_ratio)
        self.min_nominal_window = float(min_nominal_window)
        self.variance_warn_threshold = float(variance_warn_threshold)
        self.variance_log_interval = int(variance_log_interval)
        self.idle_reset_seconds = float(idle_reset_seconds)
        self._initial_tempo_ratio = float(initial_tempo_ratio)
        self.tempo_ratio = float(initial_tempo_ratio)
        self.recent_observations: Deque[TempoObservation] = deque(maxlen=self.history_size)
        self.recent_tempo_ratios: Deque[float] = deque(
            [self.tempo_ratio],
            maxlen=self.history_size,
        )
        self.recent_control_points: Deque[tuple[int, float]] = deque(
            maxlen=self.history_size + 1,
        )
        self.last_variance = 0.0
        self._update_count = 0
        self._logger = logging.getLogger(self.__class__.__name__)

        self.last_index: int | None = None
        self.last_position: int | None = None
        self.last_change_timestamp: float | None = None

    def update(self, score_index: int, timestamp: float) -> float:
        position = self._position_for_index(score_index)
        event_time = float(timestamp)

        if self.last_change_timestamp is not None and event_time < self.last_change_timestamp:
            event_time = self.last_change_timestamp

        if self.last_position is None or self.last_change_timestamp is None:
            self.last_index = int(score_index)
            self.last_position = position
            self.last_change_timestamp = event_time
            self.recent_control_points.clear()
            self.recent_control_points.append((position, event_time))
            return self.tempo_ratio

        idle_gap = event_time - self.last_change_timestamp
        if idle_gap > self.idle_reset_seconds:
            self.reset()
            self.last_index = int(score_index)
            self.last_position = position
            self.last_change_timestamp = event_time
            self.recent_control_points.append((position, event_time))
            return self.tempo_ratio

        if position == self.last_position:
            return self.tempo_ratio

        self.recent_control_points.append((position, event_time))
        raw_observations: list[TempoObservation] = []

        for anchor_position, anchor_time in self.recent_control_points:
            if anchor_position == position:
                continue

            nominal_elapsed = abs(float(self.nominal_onsets[position] - self.nominal_onsets[anchor_position]))
            if nominal_elapsed <= self._MIN_ELAPSED:
                nominal_elapsed = abs(
                    float(self.cumulative_nominal_time[position] - self.cumulative_nominal_time[anchor_position])
                )
            if nominal_elapsed < self.min_nominal_window:
                continue

            actual_elapsed = max(self._MIN_ELAPSED, abs(event_time - anchor_time))
            raw_ratio = float(
                np.clip(
                    nominal_elapsed / actual_elapsed,
                    self.min_tempo_ratio,
                    self.max_tempo_ratio,
                )
            )
            raw_observations.append(
                TempoObservation(
                    nominal_elapsed=nominal_elapsed,
                    actual_elapsed=actual_elapsed,
                    raw_ratio=raw_ratio,
                )
            )

        if raw_observations:
            representative_observation = max(
                raw_observations,
                key=lambda observation: observation.nominal_elapsed,
            )
            self.recent_observations.append(representative_observation)
            previous_ratio = float(self.tempo_ratio)
            history = np.asarray(
                [sample.raw_ratio for sample in self.recent_observations],
                dtype=np.float64,
            )
            smoothed_ratio = float(np.median(history))
            if self.smoothing_factor < 1.0:
                smoothed_ratio = float(
                    previous_ratio
                    + (np.clip(self.smoothing_factor, 0.0, 1.0) * (smoothed_ratio - previous_ratio))
                )
            baseline = max(abs(previous_ratio), self._MIN_ELAPSED)
            relative_change = abs(smoothed_ratio - previous_ratio) / baseline
            if relative_change >= self.deadzone_ratio:
                self.tempo_ratio = smoothed_ratio
            self.last_variance = float(np.var(history, dtype=np.float64))
            self.recent_tempo_ratios.append(float(self.tempo_ratio))
        elif self.recent_observations:
            history = np.asarray(
                [sample.raw_ratio for sample in self.recent_observations],
                dtype=np.float64,
            )
            self.last_variance = float(np.var(history, dtype=np.float64))
        else:
            self.last_variance = 0.0

        self._update_count += 1

        self.last_index = int(score_index)
        self.last_position = position
        self.last_change_timestamp = event_time
        return self.tempo_ratio

    def reset(self) -> None:
        self.tempo_ratio = self._initial_tempo_ratio
        self.recent_observations.clear()
        self.recent_tempo_ratios.clear()
        self.recent_tempo_ratios.append(self.tempo_ratio)
        self.recent_control_points.clear()
        self.last_index = None
        self.last_position = None
        self.last_change_timestamp = None
        self.last_variance = 0.0
        self._update_count = 0

    def maybe_reset_idle(self, current_time: float) -> bool:
        if self.last_change_timestamp is None:
            return False
        if (float(current_time) - self.last_change_timestamp) <= self.idle_reset_seconds:
            return False
        if abs(self.tempo_ratio - self._initial_tempo_ratio) <= 1e-6:
            return False
        self.reset()
        return True

    def _position_for_index(self, score_index: int) -> int:
        try:
            return int(self.index_to_position[int(score_index)])
        except KeyError as exc:
            raise ValueError(f"Unknown score index: {score_index}") from exc

    def _log_variance_if_needed(self, raw_std: float, effective_smoothing: float) -> None:
        del raw_std, effective_smoothing
        return


class ScoreEventDispatcher:
    """Fan out follower predictions to playback subscribers on a worker thread."""

    _SENTINEL = object()

    def __init__(
        self,
        score_json: str | Path | dict[str, Any] | list[dict[str, Any]],
        *,
        tempo_tracker: TempoTracker | None = None,
        queue_maxsize: int = 0,
        autostart: bool = True,
    ) -> None:
        self.tempo_tracker = tempo_tracker or TempoTracker(score_json)
        self._queue: queue.Queue[DispatchEvent | object] = queue.Queue(maxsize=queue_maxsize)
        self._callbacks: list[DispatchCallback] = []
        self._callbacks_lock = threading.RLock()
        self._worker_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        self.current_index: int | None = None
        self.current_tempo_ratio = float(self.tempo_tracker.tempo_ratio)
        self.current_event_timestamp: float | None = None
        self.last_broadcast_wall_time: float | None = None

        if autostart:
            self.start()

    def start(self) -> None:
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return

            self._stop_event.clear()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="ScoreEventDispatcher",
                daemon=True,
            )
            self._worker.start()

    def subscribe(self, callback: DispatchCallback) -> None:
        with self._callbacks_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unsubscribe(self, callback: DispatchCallback) -> None:
        with self._callbacks_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def broadcast(self, current_index: int, timestamp: float, *, tempo_update: bool = True) -> None:
        self.start()
        event = DispatchEvent(
            index=int(current_index),
            timestamp=float(timestamp),
            tempo_update=bool(tempo_update),
        )
        self.last_broadcast_wall_time = time.monotonic()

        try:
            self._queue.put_nowait(event)
            return
        except queue.Full:
            pass

        try:
            dropped = self._queue.get_nowait()
            self._queue.task_done()
            if dropped is not self._SENTINEL:
                logging.warning("ScoreEventDispatcher queue overflow, dropping stale event")
        except queue.Empty:
            pass

        self._queue.put_nowait(event)

    def flush(self, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def clear_pending(self) -> int:
        cleared = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break

            self._queue.task_done()
            if item is self._SENTINEL:
                self._queue.put_nowait(item)
                break

            cleared += 1
        return cleared

    def close(self, timeout: float = 1.0) -> None:
        thread: threading.Thread | None
        with self._worker_lock:
            thread = self._worker
            self._stop_event.set()

        if thread is None:
            return

        while True:
            try:
                self._queue.put(self._SENTINEL, timeout=0.05)
                break
            except queue.Full:
                if not thread.is_alive():
                    break

        if thread.is_alive():
            thread.join(timeout=timeout)

        with self._worker_lock:
            if self._worker is thread and not thread.is_alive():
                self._worker = None

    def __enter__(self) -> "ScoreEventDispatcher":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _worker_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                if self.tempo_tracker.maybe_reset_idle(time.monotonic()):
                    self.current_tempo_ratio = float(self.tempo_tracker.tempo_ratio)
                if self._stop_event.is_set():
                    return
                continue

            try:
                if item is self._SENTINEL:
                    return

                assert isinstance(item, DispatchEvent)
                if item.tempo_update:
                    tempo_ratio = float(self.tempo_tracker.update(item.index, item.timestamp))
                else:
                    tempo_ratio = float(self.current_tempo_ratio)
                self.current_index = item.index
                self.current_tempo_ratio = tempo_ratio
                self.current_event_timestamp = item.timestamp

                with self._callbacks_lock:
                    callbacks = tuple(self._callbacks)

                for callback in callbacks:
                    try:
                        callback(item.index, tempo_ratio)
                    except Exception:
                        logging.exception("ScoreEventDispatcher callback failed")
            finally:
                self._queue.task_done()


class MockOrchestraPlayer:
    """Console-only playback stub for dispatcher integration tests."""

    def __init__(
        self,
        dispatcher: ScoreEventDispatcher,
        *,
        tempo_change_threshold: float = 0.05,
        logger: logging.Logger | None = None,
    ) -> None:
        if tempo_change_threshold < 0.0:
            raise ValueError("tempo_change_threshold must be non-negative")

        self.dispatcher = dispatcher
        self.tempo_change_threshold = float(tempo_change_threshold)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._last_logged_index: int | None = None
        self._last_logged_tempo: float | None = None

        self.dispatcher.subscribe(self.handle_dispatch)

    def close(self) -> None:
        self.dispatcher.unsubscribe(self.handle_dispatch)

    def handle_dispatch(self, index: int, tempo_ratio: float) -> None:
        if self._last_logged_index != index:
            self.logger.info("Orchestra jumping to measure/index %d", index)
            self._last_logged_index = int(index)

        if self._should_log_tempo(tempo_ratio):
            self.logger.info("Orchestra adjusting playback speed to %.2fx", tempo_ratio)
            self._last_logged_tempo = float(tempo_ratio)

    def _should_log_tempo(self, tempo_ratio: float) -> bool:
        if self._last_logged_tempo is None:
            return True

        baseline = max(abs(self._last_logged_tempo), TempoTracker._MIN_ELAPSED)
        relative_change = abs(tempo_ratio - self._last_logged_tempo) / baseline
        return relative_change > self.tempo_change_threshold


class PygameMidiOrchestra:
    """Play short MIDI piano accompaniment chords from dispatcher updates."""

    def __init__(
        self,
        dispatcher: ScoreEventDispatcher,
        score_json: str | Path | dict[str, Any] | list[dict[str, Any]],
        *,
        instrument_program: int = 0,
        midi_channel: int = 0,
        velocity: int = 76,
        base_chord_duration: float = 0.42,
        logger: logging.Logger | None = None,
    ) -> None:
        if not 0 <= instrument_program <= 127:
            raise ValueError("instrument_program must be in the range [0, 127]")
        if not 0 <= midi_channel <= 15:
            raise ValueError("midi_channel must be in the range [0, 15]")
        if not 0 <= velocity <= 127:
            raise ValueError("velocity must be in the range [0, 127]")
        if base_chord_duration <= 0.0:
            raise ValueError("base_chord_duration must be positive")

        _, notes = _load_score(score_json)
        self.state_indices = np.asarray(
            [int(note.get("index", position)) for position, note in enumerate(notes)],
            dtype=np.int64,
        )
        self.index_to_position = {
            int(score_index): position for position, score_index in enumerate(self.state_indices)
        }
        self.score_chords = [
            sorted({int(np.clip(pitch, 0, 127)) for pitch in _note_pitches(note)})
            for note in notes
        ]
        self.score_pitches = np.asarray(
            [max(chord) for chord in self.score_chords],
            dtype=np.int64,
        )

        self.dispatcher = dispatcher
        self.instrument_program = int(instrument_program)
        self.midi_channel = int(midi_channel)
        self.velocity = int(velocity)
        self.base_chord_duration = float(base_chord_duration)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._lock = threading.RLock()
        self._output: pygame.midi.Output | None = None
        self._initialized_midi = False
        self._active_notes: list[int] = []
        self._last_index: int | None = None
        self._release_timer: threading.Timer | None = None
        self._playback_token = 0
        self.is_available = False
        self.status_label = "MIDI orchestra unavailable"

        self.dispatcher.subscribe(self.handle_dispatch)
        self._open_output()

    def close(self) -> None:
        self.dispatcher.unsubscribe(self.handle_dispatch)
        with self._lock:
            self._cancel_release_timer_locked()
            self._stop_active_notes_locked()
            output = self._output
            self._output = None
            initialized_midi = self._initialized_midi
            self._initialized_midi = False

        if output is not None:
            try:
                output.close()
            except Exception:
                self.logger.exception("Failed to close MIDI output cleanly")

        if initialized_midi and pygame is not None and pygame.midi.get_init():
            try:
                pygame.midi.quit()
            except Exception:
                self.logger.exception("Failed to quit pygame.midi cleanly")

        self.is_available = False

    def panic(self) -> None:
        with self._lock:
            self._cancel_release_timer_locked()
            self._stop_active_notes_locked()
            self._last_index = None

    def handle_dispatch(self, index: int, tempo_ratio: float) -> None:
        if not self.is_available:
            return

        with self._lock:
            if self._output is None or index == self._last_index:
                return

            self._last_index = int(index)
            self._cancel_release_timer_locked()
            self._stop_active_notes_locked()

            chord_notes = self._chord_for_index(index)
            for note in chord_notes:
                self._output.note_on(int(note), self.velocity, self.midi_channel)
            self._active_notes = chord_notes

            release_after = float(
                np.clip(
                    self.base_chord_duration / max(tempo_ratio, 0.35),
                    0.14,
                    0.75,
                )
            )
            self._playback_token += 1
            token = self._playback_token
            timer = threading.Timer(release_after, self._release_if_current, args=(token,))
            timer.daemon = True
            self._release_timer = timer
            timer.start()

    def _open_output(self) -> None:
        if pygame is None:
            self.logger.warning("pygame.midi is not installed; disabling MIDI orchestra")
            self.status_label = "MIDI orchestra unavailable"
            return

        try:
            if not pygame.midi.get_init():
                pygame.midi.init()
                self._initialized_midi = True

            output_id = pygame.midi.get_default_output_id()
            if output_id < 0:
                output_id = self._first_output_device_id()
            if output_id < 0:
                self.logger.warning("No MIDI output device found; disabling MIDI orchestra")
                self.status_label = "MIDI orchestra unavailable"
                return

            self._output = pygame.midi.Output(output_id, latency=0)
            self._output.set_instrument(self.instrument_program, self.midi_channel)
            self.is_available = True
            self.status_label = f"Piano via MIDI (Program {self.instrument_program})"
        except Exception:
            self.logger.exception("Failed to initialize MIDI orchestra")
            self.status_label = "MIDI orchestra unavailable"
            self.is_available = False
            if self._output is not None:
                try:
                    self._output.close()
                except Exception:
                    pass
                self._output = None

    def _first_output_device_id(self) -> int:
        assert pygame is not None
        for device_id in range(pygame.midi.get_count()):
            device_info = pygame.midi.get_device_info(device_id)
            if device_info is None:
                continue
            is_output = bool(device_info[3])
            if is_output:
                return int(device_id)
        return -1

    def _position_for_index(self, score_index: int) -> int:
        try:
            return int(self.index_to_position[int(score_index)])
        except KeyError:
            return int(np.clip(score_index, 0, len(self.score_pitches) - 1))

    def _chord_for_index(self, score_index: int) -> list[int]:
        position = self._position_for_index(score_index)
        return list(self.score_chords[position])

    def _release_if_current(self, token: int) -> None:
        with self._lock:
            if token != self._playback_token:
                return
            self._release_timer = None
            self._stop_active_notes_locked()

    def _cancel_release_timer_locked(self) -> None:
        if self._release_timer is None:
            return
        self._release_timer.cancel()
        self._release_timer = None

    def _stop_active_notes_locked(self) -> None:
        if self._output is None or not self._active_notes:
            self._active_notes = []
            return

        for note in self._active_notes:
            try:
                self._output.note_off(int(note), 0, self.midi_channel)
            except Exception:
                self.logger.exception("Failed to stop MIDI note %s", note)
        self._active_notes = []


def _run_demo(score_path: Path, midi_path: Path) -> None:
    follower = HybridScoreFollower(score_path)
    dispatcher = ScoreEventDispatcher(score_path)
    orchestra = MockOrchestraPlayer(dispatcher)

    try:
        midi_events = iter_midi_note_events(midi_path)
        logging.info(
            "Dispatching %d MIDI note_on events from %s against %s",
            len(midi_events),
            midi_path.name,
            score_path.name,
        )

        for event in midi_events:
            predicted_index = follower.process_event(
                int(event["pitch"]),
                float(event["timestamp"]),
            )
            dispatcher.broadcast(predicted_index, float(event["timestamp"]))

        dispatcher.flush(timeout=2.0)
        logging.info(
            "Final follower index=%d confidence=%.3f tempo=%.2fx",
            follower.current_index,
            follower.confidence,
            dispatcher.current_tempo_ratio,
        )
    finally:
        orchestra.close()
        dispatcher.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    project_root = Path(__file__).resolve().parent
    score_path = project_root / "generated_dataset" / "noisy.json"
    midi_path = project_root / "generated_dataset" / "noisy.mid"

    if not score_path.exists():
        raise SystemExit(f"Missing score file: {score_path}")
    if not midi_path.exists():
        raise SystemExit(f"Missing MIDI file: {midi_path}")

    _run_demo(score_path, midi_path)
