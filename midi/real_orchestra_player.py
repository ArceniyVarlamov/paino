from __future__ import annotations

import argparse
import heapq
import json
import logging
import queue
import re
import sys
import threading
import time
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"

for candidate in (PROJECT_ROOT, VENDOR_DIR):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    import mido
except ModuleNotFoundError as exc:
    raise SystemExit(
        "mido is not installed. Install it into the local .vendor directory first."
    ) from exc

try:
    import pygame
    import pygame.midi
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pygame.midi is not installed. Install pygame into the local .vendor directory first."
    ) from exc

import numpy as np

from hybrid_fusion import HybridScoreFollower
from midi_to_score import convert_to_score
from output_dispatcher import ScoreEventDispatcher, TempoTracker

MidiEvent = dict[str, float | int]
MidiEventQueue = queue.Queue[MidiEvent]
PHILHARMONIA_STRINGS_URL = (
    "https://philharmonia-assets.s3-eu-west-1.amazonaws.com/uploads/2020/02/12112005/Strings.zip"
)
PHILHARMONIA_STRINGS_PAGE = "https://philharmonia.co.uk/resources/sound-samples/"
DEFAULT_STRINGS_ZIP_PATH = PROJECT_ROOT / "assets" / "orchestra_samples" / "Strings.zip"
DEFAULT_STRINGS_CACHE_DIR = PROJECT_ROOT / "assets" / "orchestra_samples" / "philharmonia_strings"
GM_PIANO_PROGRAMS = set(range(8))
SAMPLE_CHANNEL_START = 64
SAMPLE_CHANNEL_END = 128
ORCHESTRA_EXPRESSION_CC = 11
ORCHESTRA_VOLUME_CC = 7
ORCHESTRA_PAN_CC = 10
ORCHESTRA_REVERB_CC = 91
ORCHESTRA_CHORUS_CC = 93
ORCHESTRA_VOLUME_MIN = 0.05
ORCHESTRA_VOLUME_MAX = 2.0
PHILHARMONIA_NOTE_TO_SEMITONE = {
    "C": 0,
    "Cs": 1,
    "D": 2,
    "Ds": 3,
    "E": 4,
    "F": 5,
    "Fs": 6,
    "G": 7,
    "Gs": 8,
    "A": 9,
    "As": 10,
    "B": 11,
}
_SAMPLE_NAME_RE = re.compile(
    r"^Strings/(?P<family>cello|double bass|viola|violin)/"
    r"[^/]+_(?P<note>[A-G]s?\d)_(?P<length>025|05|1|15)_(?P<dynamic>[a-z-]+)_arco-normal\.mp3$"
)


def available_midi_output_devices() -> list[tuple[int, str]]:
    outputs: list[tuple[int, str]] = []
    for device_id in range(pygame.midi.get_count()):
        info = pygame.midi.get_device_info(device_id)
        if info is None or not bool(info[3]):
            continue
        outputs.append((int(device_id), info[1].decode(errors="ignore")))
    return outputs


def describe_available_midi_output_devices() -> str:
    outputs = available_midi_output_devices()
    if not outputs:
        return "No MIDI output devices found."
    return "\n".join(f"  - {device_id}: {device_name}" for device_id, device_name in outputs)


def clamp_orchestra_mix(value: float) -> float:
    return float(np.clip(float(value), ORCHESTRA_VOLUME_MIN, ORCHESTRA_VOLUME_MAX))


def apply_orchestra_mix_level(base_level: float, mix_scale: float) -> float:
    normalized_level = float(np.clip(float(base_level), 0.0, 1.0))
    normalized_mix = clamp_orchestra_mix(mix_scale)
    if normalized_mix <= 1.0:
        return float(np.clip(normalized_level * normalized_mix, 0.0, 1.0))

    boost_progress = (normalized_mix - 1.0) / max(1e-6, ORCHESTRA_VOLUME_MAX - 1.0)
    boosted_level = normalized_level + ((1.0 - normalized_level) * boost_progress)
    return float(np.clip(boosted_level, 0.0, 1.0))


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the hybrid follower on solo.mid while dynamically time-scaling orchestra.mid.",
    )
    parser.add_argument(
        "--solo-midi",
        type=Path,
        default=script_dir / "solo.mid",
        help="Path to the solo piano MIDI file.",
    )
    parser.add_argument(
        "--solo-json",
        type=Path,
        default=script_dir / "solo.json",
        help="Path to the solo piano score JSON file.",
    )
    parser.add_argument(
        "--orchestra-midi",
        type=Path,
        default=script_dir / "orchestra.mid",
        help="Path to the orchestra MIDI file.",
    )
    parser.add_argument(
        "--human-speed",
        type=float,
        default=1.0,
        help="Relative solo replay speed. 0.8 means 20%% slower, 1.2 means 20%% faster.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian emission sigma passed to HybridScoreFollower.",
    )
    parser.add_argument(
        "--midi-out",
        type=int,
        default=-1,
        help=(
            "MIDI output device ID for orchestra playback. "
            "Use -1 to select the system default output automatically."
        ),
    )
    parser.add_argument(
        "--force-instrument",
        type=int,
        default=None,
        help=(
            "Optional General MIDI program override (0-127). "
            "When set, suppress original program changes and force this instrument."
        ),
    )
    return parser


def ensure_solo_json(solo_midi_path: Path, solo_json_path: Path) -> Path:
    if solo_json_path.exists():
        return solo_json_path

    logging.info("solo.json not found, generating %s from %s", solo_json_path.name, solo_midi_path.name)
    score_payload = convert_to_score(
        solo_midi_path,
        chord_policy="chord",
        chord_epsilon=0.03,
        default_duration=0.5,
        min_duration=0.05,
    )
    solo_json_path.write_text(
        json.dumps(score_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return solo_json_path


def philharmonia_note_to_midi(note_name: str) -> int:
    match = re.fullmatch(r"([A-G]s?)(-?\d+)", note_name)
    if match is None:
        raise ValueError(f"Unsupported Philharmonia note name: {note_name}")

    note, octave_text = match.groups()
    octave = int(octave_text)
    return ((octave + 1) * 12) + PHILHARMONIA_NOTE_TO_SEMITONE[note]


class PhilharmoniaStringBank:
    """Lazy loader for real orchestral string note samples."""

    def __init__(
        self,
        zip_path: Path = DEFAULT_STRINGS_ZIP_PATH,
        *,
        cache_dir: Path = DEFAULT_STRINGS_CACHE_DIR,
        logger: logging.Logger | None = None,
    ) -> None:
        self.zip_path = zip_path
        self.cache_dir = cache_dir
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._zip_file: zipfile.ZipFile | None = None
        self._sample_index: dict[str, dict[int, list[dict[str, Any]]]] = {}
        self._sound_cache: dict[str, pygame.mixer.Sound] = {}
        self._lock = threading.RLock()

        self._ensure_archive()
        self._build_index()

    def get_sound(self, family: str, midi_pitch: int, velocity: int) -> pygame.mixer.Sound:
        family_key = self._normalize_family_name(family)
        target_pitch = int(midi_pitch)
        velocity_value = int(np.clip(velocity, 1, 127))

        with self._lock:
            family_samples = self._sample_index.get(family_key)
            if not family_samples:
                raise RuntimeError(f"No Philharmonia samples indexed for family: {family_key}")

            candidate_pitch = min(
                family_samples.keys(),
                key=lambda pitch: (abs(pitch - target_pitch), pitch),
            )
            candidates = family_samples[candidate_pitch]
            chosen = min(
                candidates,
                key=lambda meta: (
                    self._dynamic_distance(meta["dynamic"], velocity_value),
                    self._length_rank(meta["length"]),
                ),
            )

            archive_name = str(chosen["archive_name"])
            cached = self._sound_cache.get(archive_name)
            if cached is not None:
                return cached

            extracted_path = self._extract_member(archive_name)
            sound = pygame.mixer.Sound(str(extracted_path))
            self._sound_cache[archive_name] = sound
            return sound

    def _ensure_archive(self) -> None:
        if self.zip_path.exists():
            return

        self.zip_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "Downloading Philharmonia string samples from %s",
            PHILHARMONIA_STRINGS_PAGE,
        )
        urllib.request.urlretrieve(PHILHARMONIA_STRINGS_URL, self.zip_path)

    def _build_index(self) -> None:
        self._zip_file = zipfile.ZipFile(self.zip_path)

        for archive_name in self._zip_file.namelist():
            match = _SAMPLE_NAME_RE.match(archive_name)
            if match is None:
                continue

            note_name = match.group("note")
            midi_pitch = philharmonia_note_to_midi(note_name)
            family = self._normalize_family_name(match.group("family"))
            self._sample_index.setdefault(family, {}).setdefault(midi_pitch, []).append(
                {
                    "archive_name": archive_name,
                    "dynamic": match.group("dynamic"),
                    "length": match.group("length"),
                }
            )

        if not self._sample_index:
            raise RuntimeError(f"No usable string samples found in archive: {self.zip_path}")

    def _extract_member(self, archive_name: str) -> Path:
        assert self._zip_file is not None
        destination = self.cache_dir / archive_name
        if destination.exists():
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._zip_file.open(archive_name) as source, destination.open("wb") as target:
            target.write(source.read())
        return destination

    @staticmethod
    def _normalize_family_name(name: str) -> str:
        return name.replace(" ", "-").lower()

    @staticmethod
    def _dynamic_target_bucket(velocity: int) -> str:
        if velocity >= 108:
            return "fortissimo"
        if velocity >= 84:
            return "forte"
        if velocity >= 56:
            return "mezzo-piano"
        if velocity >= 36:
            return "piano"
        return "pianissimo"

    @classmethod
    def _dynamic_distance(cls, dynamic_name: str, velocity: int) -> int:
        target = cls._dynamic_target_bucket(velocity)
        order = {
            "pianissimo": 0,
            "piano": 1,
            "mezzo-piano": 2,
            "forte": 3,
            "fortissimo": 4,
        }
        target_rank = order.get(target, 2)
        sample_rank = order.get(dynamic_name, 2)
        return abs(sample_rank - target_rank)

    @staticmethod
    def _length_rank(length_code: str) -> int:
        order = {
            "15": 0,
            "1": 1,
            "05": 2,
            "025": 3,
        }
        return order.get(length_code, 99)


class ScaledMidiEmulator:
    """Replay MIDI note_on events into a queue at a controllable human speed."""

    def __init__(
        self,
        midi_file_path: str | Path,
        *,
        speed: float = 1.0,
        event_queue: MidiEventQueue | None = None,
    ) -> None:
        if speed <= 0.0:
            raise ValueError("speed must be positive")

        self._midi_file_path = Path(midi_file_path)
        self._speed = float(speed)
        self._events = event_queue if event_queue is not None else queue.Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._play_loop,
                name="ScaledMidiEmulator",
                daemon=True,
            )
            self._thread.start()

    def close(self, timeout: float = 1.0) -> None:
        thread: threading.Thread | None
        with self._lock:
            self._stop_event.set()
            thread = self._thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

    def get_events(self) -> list[MidiEvent]:
        drained: list[MidiEvent] = []
        while True:
            try:
                drained.append(self._events.get_nowait())
            except queue.Empty:
                return drained

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _play_loop(self) -> None:
        try:
            midi_file = mido.MidiFile(self._midi_file_path)
            for message in midi_file:
                delay = max(0.0, float(getattr(message, "time", 0.0))) / self._speed
                if delay and self._stop_event.wait(delay):
                    return

                if getattr(message, "type", None) == "note_on" and int(getattr(message, "velocity", 0)) > 0:
                    self._events.put(
                        {
                            "pitch": int(message.note),
                            "timestamp": time.monotonic(),
                        }
                    )
        finally:
            self._stop_event.set()
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None


@dataclass
class TimedPlaybackEvent:
    source_time: float
    message: mido.Message
    note_duration: float | None = None


@dataclass(order=True)
class ScheduledNoteOff:
    due_time: float
    order: int
    channel: int
    note: int
    generation: int
    note_key: tuple[int, int, int] = field(compare=False)


class DynamicOrchestraPlayer:
    """Slave orchestra transport driven by the dispatcher master clock."""

    _MIN_TEMPO_RATIO = 0.25
    # 2 ms is a reasonable compromise: tighter note scheduling without
    # turning the idle worker into a CPU-hungry busy poller.
    _WAIT_GRANULARITY = 0.002
    _MAX_INTER_EVENT_GAP = 0.050
    _SEEK_TIME_THRESHOLD = 1.0
    _TEMPO_SMOOTHING_WINDOW = 5
    _TEMPO_DEADZONE_RATIO = 0.02
    _MERGED_CHANNEL_FILTERED_CONTROLS = frozenset({7, 10, 91, 93, 121, 123})

    def __init__(
        self,
        orchestra_midi_path: str | Path,
        dispatcher: ScoreEventDispatcher,
        *,
        midi_output_id: int = -1,
        output_channel: int | None = None,
        channel_offset: int = 0,
        force_instrument: int | None = None,
        volume_scale: float = 1.0,
        midi_output: Any | None = None,
        time_source: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
    ) -> None:
        self._midi_path = Path(orchestra_midi_path)
        self._dispatcher = dispatcher
        self._requested_midi_output_id = int(midi_output_id)
        self._forced_output_channel = (
            None if output_channel is None else int(max(0, min(15, output_channel)))
        )
        self._channel_offset = int(max(0, min(15, channel_offset)))
        self._force_instrument = (
            None if force_instrument is None else int(np.clip(force_instrument, 0, 127))
        )
        self._volume_scale = clamp_orchestra_mix(volume_scale)
        self._injected_output = midi_output
        self._clock = time_source or time.time
        self._wall_clock = wall_clock or time.time
        self._logger = logging.getLogger(self.__class__.__name__)
        self._tempo_ratio = 1.0
        self._playback_tempo_ratio = 1.0
        self._tempo_history: deque[float] = deque(
            [self._playback_tempo_ratio],
            maxlen=self._TEMPO_SMOOTHING_WINDOW,
        )
        self._tempo_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._note_off_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._initialized_midi = False
        self._owns_output = midi_output is None
        self._output: pygame.midi.Output | None = None
        self._resolved_midi_output_id = -1
        self._output_lock = threading.RLock()
        self.status_label = "Real MIDI output"
        self._master_index: int | None = None
        self._master_target_time: float | None = self._initial_master_target_time()
        self._master_anchor_clock_time: float | None = (
            self._clock() if self._master_target_time is not None else None
        )
        self._master_next_target_time: float | None = self._initial_master_next_target_time()
        self._transport_paused = False
        self._transport_generation = 0
        self._observed_output_channels: set[int] = set()
        self._program_history_by_output_channel: dict[int, list[tuple[float, int]]] = {}
        self._scheduled_events = self._load_scheduled_events()
        self._source_times = np.asarray(
            [event.source_time for event in self._scheduled_events],
            dtype=np.float64,
        )
        self._event_index = 0
        self._seek_request_time: float | None = None
        self._last_orchestra_time: float | None = None
        self._last_emitted_source_time: float | None = None
        self._last_emit_wall_time: float | None = None
        self._pending_note_offs: list[ScheduledNoteOff] = []
        self._note_off_counter = 0
        self._note_off_lock = threading.Lock()
        self._note_off_condition = threading.Condition(self._note_off_lock)
        self._active_note_generations: dict[tuple[int, int, int], int] = {}
        self._note_generation_counter = 0

        self._dispatcher.subscribe(self.handle_dispatch)
        self._open_output()

    @property
    def playback_tempo_ratio(self) -> float:
        with self._tempo_lock:
            return float(self._playback_tempo_ratio)

    @property
    def midi_output_id(self) -> int:
        return int(self._resolved_midi_output_id)

    def set_volume_scale(self, volume_scale: float) -> None:
        with self._output_lock:
            self._volume_scale = clamp_orchestra_mix(volume_scale)
            self._apply_master_output_level()

    @property
    def shared_output(self) -> pygame.midi.Output | None:
        return self._output

    @property
    def output_lock(self) -> Any:
        return self._output_lock

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._play_loop,
                name="DynamicOrchestraPlayer",
                daemon=True,
            )
            self._note_off_thread = threading.Thread(
                target=self._note_off_loop,
                name="DynamicOrchestraNoteOffs",
                daemon=True,
            )
            self._thread.start()
            self._note_off_thread.start()

    def close(self, timeout: float = 1.0) -> None:
        self._dispatcher.unsubscribe(self.handle_dispatch)
        self.halt()

        transport_thread: threading.Thread | None
        note_off_thread: threading.Thread | None
        with self._lock:
            self._stop_event.set()
            transport_thread = self._thread
            note_off_thread = self._note_off_thread

        with self._note_off_condition:
            self._note_off_condition.notify_all()

        if transport_thread is not None and transport_thread.is_alive():
            transport_thread.join(timeout=timeout)
        if note_off_thread is not None and note_off_thread.is_alive():
            note_off_thread.join(timeout=timeout)

        with self._lock:
            if self._thread is transport_thread and (
                transport_thread is None or not transport_thread.is_alive()
            ):
                self._thread = None
            if self._note_off_thread is note_off_thread and (
                note_off_thread is None or not note_off_thread.is_alive()
            ):
                self._note_off_thread = None

        self.panic()
        output = self._output
        self._output = None
        if output is not None and self._owns_output:
            output.close()

        if self._initialized_midi and pygame.midi.get_init():
            pygame.midi.quit()
            self._initialized_midi = False

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def resume(self) -> None:
        with self._tempo_lock:
            if not self._transport_paused:
                return
            self._transport_paused = False
            self._transport_generation += 1

    def halt(self) -> None:
        with self._tempo_lock:
            self._transport_paused = True
            self._transport_generation += 1
            self._master_index = None
            self._master_target_time = None
            self._master_anchor_clock_time = None
            self._master_next_target_time = None
            self._seek_request_time = None
        self.panic()

    def panic(self) -> None:
        active_notes: list[tuple[int, int]] = []
        with self._note_off_condition:
            active_notes = sorted({(note_key[1], note_key[2]) for note_key in self._active_note_generations})
            self._pending_note_offs.clear()
            self._active_note_generations.clear()
            self._note_off_counter = 0
            self._note_generation_counter = 0
            self._note_off_condition.notify_all()
        for channel, note in active_notes:
            self._send_note_off(channel, note)
        self._send_full_panic()
        self._last_emit_wall_time = None

    def reset_to_start(self) -> None:
        """Immediately rewind orchestra playback to the beginning."""
        self.seek(0.0, log_reset=True)

    def handle_dispatch(self, index: int, tempo_ratio: float) -> None:
        new_index = int(index)
        new_target_time = self._score_index_to_target_time(new_index)
        new_next_target_time = self._score_index_to_next_target_time(new_index)
        new_anchor_clock_time = self._clock()
        with self._tempo_lock:
            previous_index = self._master_index
            previous_target_time = self._master_target_time
            self._tempo_ratio = max(self._MIN_TEMPO_RATIO, float(tempo_ratio))
            self._tempo_history.append(self._tempo_ratio)
            smoothed_ratio = float(np.mean(tuple(self._tempo_history), dtype=np.float64))
            baseline = max(abs(self._playback_tempo_ratio), self._MIN_TEMPO_RATIO)
            relative_change = abs(smoothed_ratio - self._playback_tempo_ratio) / baseline
            if relative_change >= self._TEMPO_DEADZONE_RATIO:
                self._playback_tempo_ratio = smoothed_ratio

            if self._transport_paused:
                return

            if (
                previous_index == new_index
                and previous_target_time is not None
                and abs(new_target_time - previous_target_time) <= 1e-6
            ):
                return

            self._master_index = new_index
            self._master_target_time = new_target_time
            self._master_anchor_clock_time = new_anchor_clock_time
            self._master_next_target_time = new_next_target_time

            if previous_target_time is None:
                self._seek_request_time = new_target_time
                return

            if abs(new_target_time - previous_target_time) > self._SEEK_TIME_THRESHOLD:
                self._seek_request_time = new_target_time

    def _open_output(self) -> None:
        if self._injected_output is not None:
            self._output = self._injected_output
            self.status_label = "Injected MIDI output"
            self._apply_master_output_level()
            self._apply_program_state_at(0.0)
            return

        if not pygame.midi.get_init():
            pygame.midi.init()
            self._initialized_midi = True

        output_id = self._resolve_output_id()
        if output_id < 0:
            raise RuntimeError("No MIDI output device found. Create a virtual MIDI synth/output first.")
        self._output = pygame.midi.Output(output_id, latency=0)
        self._resolved_midi_output_id = int(output_id)
        self.status_label = f"Real MIDI output #{output_id}"
        self._apply_master_output_level()
        self._apply_program_state_at(0.0)

    def _resolve_output_id(self) -> int:
        requested_output_id = self._requested_midi_output_id
        if requested_output_id >= 0:
            info = pygame.midi.get_device_info(requested_output_id)
            if info is None:
                raise RuntimeError(
                    f"MIDI output device {requested_output_id} does not exist.\n"
                    f"Available MIDI output devices:\n{describe_available_midi_output_devices()}"
                )
            if not bool(info[3]):
                raise RuntimeError(
                    f"MIDI device {requested_output_id} is not an output port.\n"
                    f"Available MIDI output devices:\n{describe_available_midi_output_devices()}"
                )
            return requested_output_id

        output_id = pygame.midi.get_default_output_id()
        if output_id < 0:
            output_id = self._first_output_id()
        return output_id

    def _first_output_id(self) -> int:
        for device_id in range(pygame.midi.get_count()):
            info = pygame.midi.get_device_info(device_id)
            if info is None:
                continue
            if bool(info[3]):
                return int(device_id)
        return -1

    def _play_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                with self._tempo_lock:
                    master_target_time = self._master_target_time
                    master_playhead_time = self._projected_master_time_locked()
                    tempo_ratio = max(self._MIN_TEMPO_RATIO, self._playback_tempo_ratio)
                    seek_target_time = self._seek_request_time
                    self._seek_request_time = None
                    transport_paused = self._transport_paused
                    transport_generation = self._transport_generation

                if seek_target_time is not None:
                    self.seek(seek_target_time, tempo_ratio=tempo_ratio, log_reset=True)
                    continue

                if transport_paused:
                    if not self._sleep_until(self._clock() + self._WAIT_GRANULARITY, transport_generation):
                        if self._stop_event.is_set():
                            return
                        continue
                    continue

                if master_target_time is not None and self._should_rewind_for_backward_jump(master_target_time):
                    self.seek(master_target_time, tempo_ratio=tempo_ratio, log_reset=True)
                    continue

                if master_playhead_time is None:
                    if not self._sleep_until(self._clock() + self._WAIT_GRANULARITY, transport_generation):
                        if self._stop_event.is_set():
                            return
                        continue
                    continue

                if self._event_index >= len(self._scheduled_events):
                    if not self._sleep_until(self._clock() + self._WAIT_GRANULARITY, transport_generation):
                        if self._stop_event.is_set():
                            return
                        continue
                    continue

                event = self._scheduled_events[self._event_index]
                if event.source_time > master_playhead_time:
                    if not self._sleep_until(self._clock() + self._WAIT_GRANULARITY, transport_generation):
                        if self._stop_event.is_set():
                            return
                        continue
                    continue

                if not self._wait_inter_event_gap(event, tempo_ratio, transport_generation):
                    if self._stop_event.is_set():
                        return
                    continue
                self._emit_event(event, tempo_ratio)
                self._last_orchestra_time = event.source_time
                self._last_emitted_source_time = event.source_time
                self._last_emit_wall_time = self._clock()
                self._event_index += 1
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._thread = None

    def seek(
        self,
        target_time: float,
        *,
        tempo_ratio: float | None = None,
        log_reset: bool = False,
    ) -> None:
        target_time = max(0.0, float(target_time))
        if tempo_ratio is None:
            with self._tempo_lock:
                tempo_ratio = self._playback_tempo_ratio
        tempo_ratio = max(self._MIN_TEMPO_RATIO, float(tempo_ratio))

        if log_reset:
            self._logger.info("Seeking orchestra to %.3fs", target_time)

        self.panic()
        self._apply_master_output_level()
        self._apply_program_state_at(target_time)
        self._event_index = int(np.searchsorted(self._source_times, target_time, side="left"))
        self._last_orchestra_time = target_time
        self._last_emitted_source_time = None
        self._last_emit_wall_time = None

        now = self._wall_clock()
        for event in self._scheduled_events:
            if event.note_duration is None:
                continue
            note_end_time = event.source_time + float(event.note_duration)
            if event.source_time >= target_time:
                break
            if note_end_time <= target_time:
                continue

            midi_channel = int(getattr(event.message, "channel", 0))
            midi_note = int(getattr(event.message, "note", 0))
            output_channel = self._message_output_channel(midi_channel)
            note_key = self._note_identity_key(midi_channel, output_channel, midi_note)
            note_generation = self._prepare_note_on(note_key)
            self._send_midi_message(event.message)

            remaining_duration = max(0.0, note_end_time - target_time)
            self._schedule_note_off(
                note_key=note_key,
                channel=output_channel,
                note=midi_note,
                generation=note_generation,
                midi_duration=remaining_duration,
                tempo_ratio=tempo_ratio,
                now=now,
            )

    def _wait_inter_event_gap(
        self,
        event: TimedPlaybackEvent,
        tempo_ratio: float,
        transport_generation: int,
    ) -> bool:
        if self._last_emitted_source_time is None or self._last_emit_wall_time is None:
            return True

        source_gap = max(0.0, float(event.source_time - self._last_emitted_source_time))
        if source_gap <= 1e-6:
            return True

        sleep_for = source_gap / tempo_ratio
        if sleep_for <= 1e-4:
            return True
        deadline = self._last_emit_wall_time + sleep_for
        return self._sleep_until(deadline, transport_generation)

    def _emit_event(self, event: TimedPlaybackEvent, tempo_ratio: float) -> None:
        midi_channel = int(getattr(event.message, "channel", 0))
        midi_note = int(getattr(event.message, "note", 0))
        output_channel = self._message_output_channel(midi_channel)
        note_key = self._note_identity_key(midi_channel, output_channel, midi_note)
        note_generation: int | None = None

        if event.message.type == "note_on" and int(getattr(event.message, "velocity", 0)) > 0:
            note_generation = self._prepare_note_on(note_key)

        self._send_midi_message(event.message)
        if event.note_duration is None:
            return

        if note_generation is None:
            return
        self._schedule_note_off(
            note_key=note_key,
            channel=output_channel,
            note=midi_note,
            generation=note_generation,
            midi_duration=max(0.0, float(event.note_duration)),
            tempo_ratio=tempo_ratio,
        )

    def _send_midi_message(self, message: mido.Message) -> bool:
        assert self._output is not None

        if self._should_suppress_merged_channel_message(message):
            return False

        if message.type == "sysex":
            payload = bytes(message.bytes())
            with self._output_lock:
                self._output.write_sys_ex(pygame.midi.time(), payload)
            return True

        data = list(message.bytes())
        while len(data) < 3:
            data.append(0)
        if message.type == "note_on" and int(getattr(message, "velocity", 0)) > 0:
            data[2] = int(
                np.clip(
                    round(127 * apply_orchestra_mix_level(data[2] / 127.0, self._volume_scale)),
                    1,
                    127,
                )
            )
        elif message.type == "control_change" and int(getattr(message, "control", -1)) in {
            ORCHESTRA_VOLUME_CC,
            ORCHESTRA_EXPRESSION_CC,
        }:
            data[2] = int(
                np.clip(
                    round(127 * apply_orchestra_mix_level(data[2] / 127.0, self._volume_scale)),
                    0,
                    127,
                )
            )
        if self._forced_output_channel is not None and 0x80 <= data[0] <= 0xEF:
            data[0] = (data[0] & 0xF0) | self._forced_output_channel
        with self._output_lock:
            self._output.write_short(data[0], data[1], data[2])
        return True

    def _send_note_off(self, channel: int, note: int) -> None:
        assert self._output is not None
        with self._output_lock:
            self._output.write_short(0x80 | int(channel), int(note), 0)

    def _send_program_change(self, channel: int, program: int) -> None:
        assert self._output is not None
        with self._output_lock:
            self._output.write_short(0xC0 | int(channel), int(program), 0)

    def _apply_master_output_level(self) -> None:
        if self._output is None:
            return
        expression_value = int(np.clip(round(127 * min(self._volume_scale, 1.0)), 0, 127))
        channels = self._target_program_channels()
        if not channels:
            return
        with self._output_lock:
            for channel in channels:
                self._output.write_short(
                    0xB0 | int(channel),
                    ORCHESTRA_VOLUME_CC,
                    127,
                )
                self._output.write_short(
                    0xB0 | int(channel),
                    ORCHESTRA_EXPRESSION_CC,
                    expression_value,
                )
                if self._forced_output_channel is None:
                    continue
                # Merged practice playback uses one synth channel, so make
                # its spatial state deterministic instead of inheriting stale
                # pan/reverb/chorus from the external host.
                self._output.write_short(
                    0xB0 | int(channel),
                    ORCHESTRA_PAN_CC,
                    64,
                )
                self._output.write_short(
                    0xB0 | int(channel),
                    ORCHESTRA_REVERB_CC,
                    0,
                )
                self._output.write_short(
                    0xB0 | int(channel),
                    ORCHESTRA_CHORUS_CC,
                    0,
                )

    def _apply_program_state_at(self, target_time: float) -> None:
        if self._output is None:
            return

        channels = self._target_program_channels()
        if not channels:
            return

        if self._force_instrument is not None:
            for channel in channels:
                self._send_program_change(channel, self._force_instrument)
            return

        if self._forced_output_channel is not None:
            return

        for channel in channels:
            program = self._latest_program_for_channel(channel, target_time)
            if program is not None:
                self._send_program_change(channel, program)

    def _send_all_notes_off(self) -> None:
        if self._output is None:
            return
        channels = (
            [self._forced_output_channel]
            if self._forced_output_channel is not None
            else list(range(16))
        )
        for channel in channels:
            with self._output_lock:
                self._output.write_short(0xB0 | channel, 64, 0)
                self._output.write_short(0xB0 | channel, 66, 0)
                self._output.write_short(0xB0 | channel, 67, 0)
                self._output.write_short(0xB0 | channel, 120, 0)
                self._output.write_short(0xB0 | channel, 121, 0)
                self._output.write_short(0xB0 | channel, 123, 0)

    def _send_brute_force_note_offs(self) -> None:
        if self._output is None:
            return
        channels = (
            [self._forced_output_channel]
            if self._forced_output_channel is not None
            else list(range(16))
        )
        for channel in channels:
            for note in range(128):
                self._send_note_off(channel, note)

    def _send_full_panic(self) -> None:
        self._send_all_notes_off()
        self._send_brute_force_note_offs()
        self._send_all_notes_off()

    def _should_rewind_for_backward_jump(self, master_target_time: float) -> bool:
        if self._last_orchestra_time is None:
            return False
        return float(master_target_time) < (self._last_orchestra_time - self._SEEK_TIME_THRESHOLD)

    def _load_scheduled_events(self) -> list[TimedPlaybackEvent]:
        midi_file = mido.MidiFile(self._midi_path)
        absolute_time = 0.0
        scheduled: list[TimedPlaybackEvent] = []
        open_notes: dict[tuple[int, int], list[TimedPlaybackEvent]] = {}
        for message in midi_file:
            absolute_time += float(getattr(message, "time", 0.0))
            if getattr(message, "is_meta", False):
                continue

            source_channel = getattr(message, "channel", None)
            if source_channel is not None:
                output_channel = self._message_output_channel(int(source_channel))
                self._observed_output_channels.add(output_channel)
                if message.type == "program_change" and self._force_instrument is None:
                    self._program_history_by_output_channel.setdefault(output_channel, []).append(
                        (absolute_time, int(getattr(message, "program", 0)))
                    )

            if self._should_suppress_merged_channel_message(message):
                continue

            if message.type == "note_on" and int(getattr(message, "velocity", 0)) > 0:
                event = TimedPlaybackEvent(
                    source_time=absolute_time,
                    message=message.copy(),
                    note_duration=None,
                )
                scheduled.append(event)
                open_notes.setdefault((int(message.channel), int(message.note)), []).append(event)
                continue

            if message.type == "note_off" or (
                message.type == "note_on" and int(getattr(message, "velocity", 0)) == 0
            ):
                note_stack = open_notes.get((int(message.channel), int(message.note)))
                if note_stack:
                    onset_event = note_stack.pop()
                    onset_event.note_duration = max(0.0, absolute_time - onset_event.source_time)
                    if not note_stack:
                        open_notes.pop((int(message.channel), int(message.note)), None)
                continue

            scheduled.append(TimedPlaybackEvent(source_time=absolute_time, message=message.copy()))

        for note_stack in open_notes.values():
            for onset_event in note_stack:
                onset_event.note_duration = max(0.0, absolute_time - onset_event.source_time)
        if self._forced_output_channel is not None:
            return self._merge_forced_channel_note_events(scheduled)
        return scheduled

    def _merge_forced_channel_note_events(
        self,
        scheduled: list[TimedPlaybackEvent],
    ) -> list[TimedPlaybackEvent]:
        note_events: list[TimedPlaybackEvent] = []
        passthrough: list[TimedPlaybackEvent] = []
        for event in scheduled:
            message = event.message
            if (
                message.type == "note_on"
                and int(getattr(message, "velocity", 0)) > 0
                and event.note_duration is not None
            ):
                note_events.append(event)
            else:
                passthrough.append(event)

        by_note: dict[int, list[TimedPlaybackEvent]] = {}
        for event in note_events:
            by_note.setdefault(int(getattr(event.message, "note", 0)), []).append(event)

        merged: list[TimedPlaybackEvent] = list(passthrough)
        for note, events in by_note.items():
            events.sort(key=lambda event: (event.source_time, event.source_time + float(event.note_duration or 0.0)))
            current_start: float | None = None
            current_end: float | None = None
            current_velocity = 0

            for event in events:
                start = float(event.source_time)
                end = start + max(0.0, float(event.note_duration or 0.0))
                velocity = int(getattr(event.message, "velocity", 0))
                if current_start is None:
                    current_start = start
                    current_end = max(start, end)
                    current_velocity = velocity
                    continue

                assert current_end is not None
                if start <= current_end + 1e-6:
                    current_end = max(current_end, end)
                    current_velocity = max(current_velocity, velocity)
                    continue

                merged.append(
                    TimedPlaybackEvent(
                        source_time=current_start,
                        message=mido.Message(
                            "note_on",
                            channel=int(self._forced_output_channel),
                            note=int(note),
                            velocity=int(max(1, min(127, current_velocity))),
                            time=0,
                        ),
                        note_duration=max(0.0, current_end - current_start),
                    )
                )
                current_start = start
                current_end = max(start, end)
                current_velocity = velocity

            if current_start is not None and current_end is not None:
                merged.append(
                    TimedPlaybackEvent(
                        source_time=current_start,
                        message=mido.Message(
                            "note_on",
                            channel=int(self._forced_output_channel),
                            note=int(note),
                            velocity=int(max(1, min(127, current_velocity))),
                            time=0,
                        ),
                        note_duration=max(0.0, current_end - current_start),
                    )
                )

        merged.sort(key=lambda event: event.source_time)
        return merged

    def _initial_master_target_time(self) -> float | None:
        current_index = self._dispatcher.current_index
        if current_index is None:
            return None
        return self._score_index_to_target_time(int(current_index))

    def _initial_master_next_target_time(self) -> float | None:
        current_index = self._dispatcher.current_index
        if current_index is None:
            return None
        return self._score_index_to_next_target_time(int(current_index))

    def _score_index_to_target_time(self, score_index: int) -> float:
        tempo_tracker = self._dispatcher.tempo_tracker
        try:
            position = int(tempo_tracker.index_to_position[int(score_index)])
        except KeyError as exc:
            raise ValueError(f"Unknown score index for orchestra sync: {score_index}") from exc
        return float(tempo_tracker.nominal_onsets[position])

    def _score_index_to_next_target_time(self, score_index: int) -> float | None:
        tempo_tracker = self._dispatcher.tempo_tracker
        try:
            position = int(tempo_tracker.index_to_position[int(score_index)])
        except KeyError as exc:
            raise ValueError(f"Unknown score index for orchestra sync: {score_index}") from exc

        next_position = position + 1
        if next_position >= len(tempo_tracker.nominal_onsets):
            return None
        return float(tempo_tracker.nominal_onsets[next_position])

    def _projected_master_time_locked(self) -> float | None:
        if self._master_target_time is None:
            return None

        target_time = float(self._master_target_time)
        anchor_clock_time = self._master_anchor_clock_time
        if anchor_clock_time is None:
            return target_time

        elapsed = max(0.0, float(self._clock() - anchor_clock_time))
        projected_time = target_time + (elapsed * max(self._MIN_TEMPO_RATIO, self._playback_tempo_ratio))
        # Keep orchestra transport advancing by MIDI time between score updates.
        # This preserves note tails and continuing accompaniment phrases when the
        # soloist pauses, while future score callbacks can still seek/correct.
        return max(target_time, projected_time)

    def _sleep_until(self, deadline: float, transport_generation: int) -> bool:
        while True:
            if self._stop_event.is_set():
                return False

            now = self._clock()
            if now >= deadline:
                return True

            with self._tempo_lock:
                if self._transport_generation != transport_generation:
                    return False
                if self._transport_paused:
                    return False
                if self._seek_request_time is not None:
                    return True
                master_target_time = self._master_target_time
            if master_target_time is not None and self._should_rewind_for_backward_jump(master_target_time):
                return True

            wait_for = max(0.0, min(deadline - now, self._WAIT_GRANULARITY))
            if wait_for <= 1e-4:
                continue
            if self._stop_event.wait(wait_for):
                return False

    def _note_off_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                due_notes: list[ScheduledNoteOff] = []
                with self._note_off_condition:
                    while not self._stop_event.is_set():
                        now = self._wall_clock()
                        while self._pending_note_offs and self._pending_note_offs[0].due_time <= now:
                            scheduled_note_off = heapq.heappop(self._pending_note_offs)
                            active_generation = self._active_note_generations.get(
                                scheduled_note_off.note_key
                            )
                            if active_generation != scheduled_note_off.generation:
                                continue
                            self._active_note_generations.pop(scheduled_note_off.note_key, None)
                            due_notes.append(scheduled_note_off)

                        if due_notes:
                            break

                        if not self._pending_note_offs:
                            self._note_off_condition.wait(timeout=self._WAIT_GRANULARITY)
                            continue

                        next_due_time = float(self._pending_note_offs[0].due_time)
                        wait_for = max(0.0, min(next_due_time - now, self._WAIT_GRANULARITY))
                        self._note_off_condition.wait(timeout=wait_for)

                for scheduled_note_off in due_notes:
                    self._send_note_off(scheduled_note_off.channel, scheduled_note_off.note)
        finally:
            with self._lock:
                if self._note_off_thread is threading.current_thread():
                    self._note_off_thread = None

    def _schedule_note_off(
        self,
        *,
        note_key: tuple[int, int, int],
        channel: int,
        note: int,
        generation: int,
        midi_duration: float,
        tempo_ratio: float,
        now: float | None = None,
    ) -> None:
        with self._note_off_condition:
            note_off = ScheduledNoteOff(
                due_time=(self._wall_clock() if now is None else now)
                + (max(0.0, float(midi_duration)) / max(self._MIN_TEMPO_RATIO, float(tempo_ratio))),
                order=self._note_off_counter,
                channel=int(channel),
                note=int(note),
                generation=int(generation),
                note_key=note_key,
            )
            self._note_off_counter += 1
            heapq.heappush(self._pending_note_offs, note_off)
            self._note_off_condition.notify()

    def _should_suppress_merged_channel_message(self, message: mido.Message) -> bool:
        if self._forced_output_channel is not None and message.type not in {"note_on", "note_off"}:
            return True

        if message.type == "program_change":
            return self._force_instrument is not None or self._forced_output_channel is not None

        if self._forced_output_channel is None:
            return False

        if message.type == "control_change":
            return int(getattr(message, "control", -1)) in self._MERGED_CHANNEL_FILTERED_CONTROLS

        return False

    def _message_output_channel(self, source_channel: int) -> int:
        if self._forced_output_channel is not None:
            return int(self._forced_output_channel)
        return int(max(0, min(15, int(source_channel) + self._channel_offset)))

    def _note_identity_key(
        self,
        source_channel: int,
        output_channel: int,
        note: int,
    ) -> tuple[int, int, int]:
        return (int(source_channel), int(output_channel), int(note))

    def _prepare_note_on(self, note_key: tuple[int, int, int]) -> int:
        _, output_channel, note = note_key
        send_dedup_note_off = False
        with self._note_off_condition:
            if note_key in self._active_note_generations:
                send_dedup_note_off = True
            generation = self._note_generation_counter
            self._note_generation_counter += 1
            self._active_note_generations[note_key] = generation
            self._note_off_condition.notify()

        if send_dedup_note_off:
            self._send_note_off(output_channel, note)

        return generation

    def _target_program_channels(self) -> list[int]:
        if self._forced_output_channel is not None:
            return [int(self._forced_output_channel)]
        return sorted(self._observed_output_channels)

    def _latest_program_for_channel(self, channel: int, target_time: float) -> int | None:
        history = self._program_history_by_output_channel.get(int(channel))
        if not history:
            return None

        latest_program: int | None = None
        for event_time, program in history:
            if event_time > target_time:
                break
            latest_program = int(program)
        return latest_program


def main() -> int:
    args = build_parser().parse_args()
    solo_midi_path = args.solo_midi.expanduser().resolve()
    solo_json_path = args.solo_json.expanduser().resolve()
    orchestra_midi_path = args.orchestra_midi.expanduser().resolve()

    if not solo_midi_path.exists():
        raise SystemExit(f"solo.mid not found: {solo_midi_path}")
    if not orchestra_midi_path.exists():
        raise SystemExit(f"orchestra.mid not found: {orchestra_midi_path}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    solo_json_path = ensure_solo_json(solo_midi_path, solo_json_path)

    follower = HybridScoreFollower(solo_json_path, sigma=args.sigma)
    tempo_tracker = TempoTracker(solo_json_path)
    dispatcher = ScoreEventDispatcher(solo_json_path, tempo_tracker=tempo_tracker)
    orchestra: DynamicOrchestraPlayer | None = None
    emulator: ScaledMidiEmulator | None = None

    try:
        orchestra = DynamicOrchestraPlayer(
            orchestra_midi_path,
            dispatcher,
            midi_output_id=args.midi_out,
            force_instrument=args.force_instrument,
        )
        emulator = ScaledMidiEmulator(solo_midi_path, speed=args.human_speed)
        last_logged_index = -1

        logging.info(
            "Starting demo with solo=%s orchestra=%s human_speed=%.2fx",
            solo_midi_path.name,
            orchestra_midi_path.name,
            args.human_speed,
        )

        orchestra.start()
        emulator.start()

        while True:
            events = emulator.get_events()
            if not events:
                if not emulator.is_running:
                    break
                time.sleep(0.005)
                continue

            for event in events:
                predicted_index = follower.process_event(
                    int(event["pitch"]),
                    float(event["timestamp"]),
                )
                dispatcher.broadcast(predicted_index, float(event["timestamp"]))

                if predicted_index != last_logged_index:
                    last_logged_index = predicted_index
                    logging.info(
                        "Follower index=%d confidence=%.3f tempo=%.2fx mode=%s",
                        predicted_index,
                        follower.confidence,
                        dispatcher.current_tempo_ratio,
                        follower.mode_label,
                    )

        dispatcher.flush(timeout=2.0)
        time.sleep(0.25)
        logging.info(
            "Finished. Final index=%d confidence=%.3f tempo=%.2fx",
            follower.current_index,
            follower.confidence,
            dispatcher.current_tempo_ratio,
        )
        return 0
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        if emulator is not None:
            emulator.close()
        if orchestra is not None:
            orchestra.close()
        dispatcher.close()


if __name__ == "__main__":
    raise SystemExit(main())
