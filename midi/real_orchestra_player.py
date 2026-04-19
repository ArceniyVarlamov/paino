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
from dataclasses import dataclass
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


class DynamicOrchestraPlayer:
    """Slave orchestra transport driven by the dispatcher master clock."""

    _MIN_TEMPO_RATIO = 0.25
    _WAIT_GRANULARITY = 0.010
    _MAX_INTER_EVENT_GAP = 0.050
    _BACKWARD_RESET_THRESHOLD = 1.0
    _FORWARD_SEEK_INDEX_THRESHOLD = 4

    def __init__(
        self,
        orchestra_midi_path: str | Path,
        dispatcher: ScoreEventDispatcher,
        *,
        midi_output_id: int = -1,
        midi_output: Any | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self._midi_path = Path(orchestra_midi_path)
        self._dispatcher = dispatcher
        self._requested_midi_output_id = int(midi_output_id)
        self._injected_output = midi_output
        self._clock = time_source or time.time
        self._logger = logging.getLogger(self.__class__.__name__)
        self._tempo_ratio = 1.0
        self._tempo_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._initialized_midi = False
        self._output: pygame.midi.Output | None = None
        self.status_label = "Real MIDI output"
        self._master_index: int | None = None
        self._master_target_time: float | None = self._initial_master_target_time()
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
        self._pending_note_offs_lock = threading.Lock()
        self._active_note_generations: dict[tuple[int, int], int] = {}
        self._note_generation_counter = 0

        self._dispatcher.subscribe(self.handle_dispatch)
        self._open_output()

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
            self._thread.start()

    def close(self, timeout: float = 1.0) -> None:
        self._dispatcher.unsubscribe(self.handle_dispatch)

        thread: threading.Thread | None
        with self._lock:
            self._stop_event.set()
            thread = self._thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

        with self._lock:
            if self._thread is thread and (thread is None or not thread.is_alive()):
                self._thread = None

        self._panic_all_notes_off()
        output = self._output
        self._output = None
        if output is not None:
            output.close()

        if self._initialized_midi and pygame.midi.get_init():
            pygame.midi.quit()
            self._initialized_midi = False

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def panic(self) -> None:
        self._panic_all_notes_off()
        with self._pending_note_offs_lock:
            self._pending_note_offs.clear()
            self._active_note_generations.clear()
        self._note_off_counter = 0
        self._last_emit_wall_time = None

    def reset_to_start(self) -> None:
        """Immediately rewind orchestra playback to the beginning."""
        self.seek(0.0, log_reset=True)

    def handle_dispatch(self, index: int, tempo_ratio: float) -> None:
        new_index = int(index)
        new_target_time = self._score_index_to_target_time(new_index)
        with self._tempo_lock:
            self._tempo_ratio = max(self._MIN_TEMPO_RATIO, float(tempo_ratio))
            previous_index = self._master_index
            previous_target_time = self._master_target_time
            self._master_index = new_index
            self._master_target_time = new_target_time

            if previous_index is None:
                if new_target_time > self._BACKWARD_RESET_THRESHOLD:
                    self._seek_request_time = new_target_time
                return

            backward_jump = (
                previous_target_time is not None
                and new_target_time < (previous_target_time - 1e-6)
            )
            forward_jump = (new_index - previous_index) >= self._FORWARD_SEEK_INDEX_THRESHOLD
            if backward_jump or forward_jump:
                self._seek_request_time = new_target_time

    def _open_output(self) -> None:
        if self._injected_output is not None:
            self._output = self._injected_output
            self.status_label = "Injected MIDI output"
            return

        if not pygame.midi.get_init():
            pygame.midi.init()
            self._initialized_midi = True

        output_id = self._resolve_output_id()
        if output_id < 0:
            raise RuntimeError("No MIDI output device found. Create a virtual MIDI synth/output first.")
        self._output = pygame.midi.Output(output_id, latency=0)
        self.status_label = f"Real MIDI output #{output_id}"

    def _resolve_output_id(self) -> int:
        requested_output_id = self._requested_midi_output_id
        if requested_output_id >= 0:
            info = pygame.midi.get_device_info(requested_output_id)
            if info is None:
                raise RuntimeError(f"MIDI output device {requested_output_id} does not exist.")
            if not bool(info[3]):
                raise RuntimeError(f"MIDI device {requested_output_id} is not an output port.")
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
                now = self._clock()
                self._flush_due_note_offs(now)

                with self._tempo_lock:
                    master_target_time = self._master_target_time
                    tempo_ratio = max(self._MIN_TEMPO_RATIO, self._tempo_ratio)
                    seek_target_time = self._seek_request_time
                    self._seek_request_time = None

                if seek_target_time is not None:
                    self.seek(seek_target_time, tempo_ratio=tempo_ratio, log_reset=True)
                    continue

                if master_target_time is not None and self._should_rewind_for_backward_jump(master_target_time):
                    self.seek(master_target_time, tempo_ratio=tempo_ratio, log_reset=True)
                    continue

                if master_target_time is None:
                    if self._sleep_with_note_offs(self._sleep_deadline(now)):
                        return
                    continue

                if self._event_index >= len(self._scheduled_events):
                    if self._sleep_with_note_offs(self._sleep_deadline(now)):
                        return
                    continue

                event = self._scheduled_events[self._event_index]
                if event.source_time > master_target_time:
                    if self._sleep_with_note_offs(self._sleep_deadline(now)):
                        return
                    continue

                if not self._wait_inter_event_gap(event, tempo_ratio):
                    return
                self._emit_event(event, tempo_ratio)
                self._last_orchestra_time = event.source_time
                self._last_emitted_source_time = event.source_time
                self._last_emit_wall_time = self._clock()
                self._event_index += 1
        finally:
            self._flush_due_note_offs(float("inf"))
            self._panic_all_notes_off()
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
                tempo_ratio = self._tempo_ratio
        tempo_ratio = max(self._MIN_TEMPO_RATIO, float(tempo_ratio))

        if log_reset:
            self._logger.info("Seeking orchestra to %.3fs", target_time)

        self.panic()
        self._event_index = int(np.searchsorted(self._source_times, target_time, side="left"))
        self._last_orchestra_time = target_time
        self._last_emitted_source_time = None
        self._last_emit_wall_time = None

        now = self._clock()
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
            note_key = (midi_channel, midi_note)
            note_generation = self._prepare_note_on(note_key)
            self._send_midi_message(event.message)

            remaining_duration = max(0.0, note_end_time - target_time)
            note_off = ScheduledNoteOff(
                due_time=now + (remaining_duration / tempo_ratio),
                order=self._note_off_counter,
                channel=midi_channel,
                note=midi_note,
                generation=note_generation,
            )
            self._note_off_counter += 1
            with self._pending_note_offs_lock:
                heapq.heappush(self._pending_note_offs, note_off)

    def _wait_inter_event_gap(
        self,
        event: TimedPlaybackEvent,
        tempo_ratio: float,
    ) -> bool:
        if self._last_emitted_source_time is None or self._last_emit_wall_time is None:
            return True

        source_gap = max(0.0, float(event.source_time - self._last_emitted_source_time))
        if source_gap <= 1e-6:
            return True

        if source_gap > self._MAX_INTER_EVENT_GAP:
            return True

        sleep_for = source_gap / tempo_ratio
        if sleep_for <= 1e-4:
            return True
        deadline = self._last_emit_wall_time + sleep_for
        return not self._sleep_with_note_offs(deadline)

    def _emit_event(self, event: TimedPlaybackEvent, tempo_ratio: float) -> None:
        self._flush_due_note_offs(self._clock())

        midi_channel = int(getattr(event.message, "channel", 0))
        midi_note = int(getattr(event.message, "note", 0))
        note_key = (midi_channel, midi_note)
        note_generation: int | None = None

        if event.message.type == "note_on" and int(getattr(event.message, "velocity", 0)) > 0:
            note_generation = self._prepare_note_on(note_key)

        self._send_midi_message(event.message)
        if event.note_duration is None:
            return

        note_duration = max(0.0, float(event.note_duration))
        due_time = self._clock() + (note_duration / tempo_ratio)
        note_off = ScheduledNoteOff(
            due_time=due_time,
            order=self._note_off_counter,
            channel=midi_channel,
            note=midi_note,
            generation=int(note_generation if note_generation is not None else -1),
        )
        self._note_off_counter += 1
        with self._pending_note_offs_lock:
            heapq.heappush(self._pending_note_offs, note_off)

    def _send_midi_message(self, message: mido.Message) -> None:
        assert self._output is not None

        if message.type == "sysex":
            payload = bytes(message.bytes())
            self._output.write_sys_ex(pygame.midi.time(), payload)
            return

        data = list(message.bytes())
        while len(data) < 3:
            data.append(0)
        self._output.write_short(data[0], data[1], data[2])

    def _send_note_off(self, channel: int, note: int) -> None:
        assert self._output is not None
        self._output.write_short(0x80 | int(channel), int(note), 0)

    def _panic_all_notes_off(self) -> None:
        if self._output is None:
            return
        for channel in range(16):
            self._output.write_short(0xB0 | channel, 121, 0)
            self._output.write_short(0xB0 | channel, 123, 0)
        with self._pending_note_offs_lock:
            self._pending_note_offs.clear()
            self._active_note_generations.clear()

    def _should_rewind_for_backward_jump(self, master_target_time: float) -> bool:
        if self._last_orchestra_time is None:
            return False
        return float(master_target_time) < (self._last_orchestra_time - self._BACKWARD_RESET_THRESHOLD)

    def _load_scheduled_events(self) -> list[TimedPlaybackEvent]:
        midi_file = mido.MidiFile(self._midi_path)
        absolute_time = 0.0
        scheduled: list[TimedPlaybackEvent] = []
        open_notes: dict[tuple[int, int], list[TimedPlaybackEvent]] = {}
        for message in midi_file:
            absolute_time += float(getattr(message, "time", 0.0))
            if getattr(message, "is_meta", False):
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
        return scheduled

    def _initial_master_target_time(self) -> float | None:
        current_index = self._dispatcher.current_index
        if current_index is None:
            return None
        return self._score_index_to_target_time(int(current_index))

    def _score_index_to_target_time(self, score_index: int) -> float:
        tempo_tracker = self._dispatcher.tempo_tracker
        try:
            position = int(tempo_tracker.index_to_position[int(score_index)])
        except KeyError as exc:
            raise ValueError(f"Unknown score index for orchestra sync: {score_index}") from exc
        return float(tempo_tracker.nominal_onsets[position])

    def _flush_due_note_offs(self, now: float) -> None:
        due_notes: list[ScheduledNoteOff] = []
        with self._pending_note_offs_lock:
            while self._pending_note_offs and self._pending_note_offs[0].due_time <= now:
                scheduled_note_off = heapq.heappop(self._pending_note_offs)
                note_key = (scheduled_note_off.channel, scheduled_note_off.note)
                active_generation = self._active_note_generations.get(note_key)
                if active_generation != scheduled_note_off.generation:
                    continue
                self._active_note_generations.pop(note_key, None)
                due_notes.append(scheduled_note_off)

        for scheduled_note_off in due_notes:
            self._send_note_off(scheduled_note_off.channel, scheduled_note_off.note)

    def _has_pending_note_offs(self) -> bool:
        with self._pending_note_offs_lock:
            return bool(self._pending_note_offs)

    def _next_note_off_due_time(self) -> float | None:
        with self._pending_note_offs_lock:
            if not self._pending_note_offs:
                return None
            return float(self._pending_note_offs[0].due_time)

    def _sleep_deadline(self, now: float) -> float:
        next_note_off_due_time = self._next_note_off_due_time()
        if next_note_off_due_time is None:
            return now + self._WAIT_GRANULARITY
        return min(now + self._WAIT_GRANULARITY, next_note_off_due_time)

    def _sleep_with_note_offs(self, deadline: float) -> bool:
        while True:
            if self._stop_event.is_set():
                return True

            now = self._clock()
            self._flush_due_note_offs(now)
            if now >= deadline:
                return False

            next_note_off_due_time = self._next_note_off_due_time()
            wake_time = deadline
            if next_note_off_due_time is not None:
                wake_time = min(wake_time, next_note_off_due_time)
            wait_for = max(0.0, min(wake_time - now, self._WAIT_GRANULARITY))
            if wait_for <= 1e-4:
                continue
            if self._stop_event.wait(wait_for):
                return True

    def _prepare_note_on(self, note_key: tuple[int, int]) -> int:
        channel, note = note_key
        send_dedup_note_off = False
        with self._pending_note_offs_lock:
            if note_key in self._active_note_generations:
                send_dedup_note_off = True
            generation = self._note_generation_counter
            self._note_generation_counter += 1
            self._active_note_generations[note_key] = generation

        if send_dedup_note_off:
            self._send_note_off(channel, note)

        return generation


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
