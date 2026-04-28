from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"

for candidate in (PROJECT_ROOT, VENDOR_DIR):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import mido

from midi.real_orchestra_player import (
    DynamicOrchestraPlayer,
    ORCHESTRA_CHORUS_CC,
    ORCHESTRA_EXPRESSION_CC,
    ORCHESTRA_PAN_CC,
    ORCHESTRA_REVERB_CC,
    ORCHESTRA_VOLUME_CC,
)
from output_dispatcher import ScoreEventDispatcher, TempoTracker

TICKS_PER_BEAT = 480
TEMPO_US_PER_BEAT = 500_000
TICKS_PER_SECOND = int(TICKS_PER_BEAT * (1_000_000 / TEMPO_US_PER_BEAT))
TEMPO_RATIO = 2.0
PAUSE_SECONDS = 5.0
TIMING_TOLERANCE = 0.10


class MockMidiOutput:
    def __init__(self) -> None:
        self.records: list[dict[str, float | int | str]] = []
        self.closed = False

    def write_short(self, status: int, data1: int = 0, data2: int = 0) -> None:
        timestamp = time.monotonic()
        message_type = status & 0xF0
        channel = status & 0x0F

        if message_type == 0x90 and data2 > 0:
            event_type = "note_on"
        elif message_type in {0x80, 0x90}:
            event_type = "note_off"
        else:
            event_type = "other"

        self.records.append(
            {
                "timestamp": timestamp,
                "event_type": event_type,
                "status": int(status),
                "channel": int(channel),
                "note": int(data1),
                "velocity": int(data2),
            }
        )

    def write_sys_ex(self, when: int, payload: bytes) -> None:
        self.records.append(
            {
                "timestamp": time.monotonic(),
                "event_type": "sysex",
                "status": int(when),
                "channel": -1,
                "note": -1,
                "velocity": len(payload),
            }
        )

    def close(self) -> None:
        self.closed = True


def seconds_to_ticks(seconds: float) -> int:
    return int(round(seconds * TICKS_PER_SECOND))


def build_validation_fixture(root: Path) -> tuple[Path, Path]:
    orchestra_path = root / "validator_orchestra.mid"
    score_path = root / "validator_score.json"

    midi_file = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO_US_PER_BEAT, time=0))
    track.append(mido.Message("program_change", channel=0, program=48, time=0))
    track.append(mido.Message("note_on", channel=0, note=60, velocity=90, time=0))
    track.append(mido.Message("note_on", channel=0, note=64, velocity=84, time=seconds_to_ticks(0.10)))
    track.append(mido.Message("note_off", channel=0, note=60, velocity=0, time=seconds_to_ticks(0.30)))
    track.append(mido.Message("note_off", channel=0, note=64, velocity=0, time=seconds_to_ticks(0.10)))
    track.append(mido.Message("note_on", channel=0, note=67, velocity=88, time=seconds_to_ticks(0.10)))
    track.append(mido.Message("note_off", channel=0, note=67, velocity=0, time=seconds_to_ticks(0.40)))
    track.append(mido.Message("note_on", channel=0, note=71, velocity=76, time=seconds_to_ticks(0.10)))
    track.append(mido.Message("note_off", channel=0, note=71, velocity=0, time=seconds_to_ticks(0.20)))
    midi_file.save(orchestra_path)

    score_payload = {
        "notes": [
            {"index": 0, "pitch": 72, "nominal_onset": 0.0, "nominal_duration": 0.10},
            {"index": 1, "pitch": 74, "nominal_onset": 0.10, "nominal_duration": 0.50},
            {"index": 2, "pitch": 76, "nominal_onset": 0.60, "nominal_duration": 0.50},
            {"index": 3, "pitch": 77, "nominal_onset": 1.10, "nominal_duration": 0.20},
        ]
    }
    score_path.write_text(json.dumps(score_payload, indent=2) + "\n", encoding="utf-8")
    return orchestra_path, score_path


def pair_note_events(records: list[dict[str, float | int | str]]) -> dict[tuple[int, int], dict[str, float]]:
    paired: dict[tuple[int, int], dict[str, float]] = {}
    note_on_times: dict[tuple[int, int], list[float]] = {}

    for record in records:
        event_type = str(record["event_type"])
        key = (int(record["channel"]), int(record["note"]))
        timestamp = float(record["timestamp"])

        if event_type == "note_on":
            note_on_times.setdefault(key, []).append(timestamp)
            continue

        if event_type != "note_off":
            continue

        if key not in note_on_times or not note_on_times[key]:
            continue

        onset = note_on_times[key].pop(0)
        paired[key] = {
            "on": onset,
            "off": timestamp,
            "duration": timestamp - onset,
        }

    return paired


def assert_close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > TIMING_TOLERANCE:
        raise AssertionError(
            f"{label} mismatch: actual={actual:.3f}s expected={expected:.3f}s "
            f"tolerance={TIMING_TOLERANCE:.3f}s"
        )


def assert_control_sent(
    records: list[dict[str, float | int | str]],
    *,
    channel: int,
    control: int,
    value: int,
) -> None:
    expected_status = 0xB0 | int(channel)
    for record in records:
        if (
            int(record["status"]) == expected_status
            and int(record["note"]) == int(control)
            and int(record["velocity"]) == int(value)
        ):
            return
    raise AssertionError(f"Missing CC{control}={value} on channel {channel}")


def wait_for_record(
    mock_output: MockMidiOutput,
    *,
    event_type: str,
    note: int,
    timeout: float = 1.0,
) -> None:
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        for record in mock_output.records:
            if str(record["event_type"]) == event_type and int(record["note"]) == note:
                return
        time.sleep(0.005)
    raise AssertionError(f"Timed out waiting for {event_type} for note {note}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="playback-validator-") as tmp_dir:
        fixture_root = Path(tmp_dir)
        orchestra_path, score_path = build_validation_fixture(fixture_root)
        mock_output = MockMidiOutput()

        tempo_tracker = TempoTracker(
            score_path,
            history_size=1,
            smoothing_factor=1.0,
            initial_tempo_ratio=TEMPO_RATIO,
        )
        dispatcher = ScoreEventDispatcher(score_path, tempo_tracker=tempo_tracker)

        with (
            mock.patch("pygame.midi.get_init", return_value=True),
            mock.patch("pygame.midi.init", return_value=None),
            mock.patch("pygame.midi.get_default_output_id", return_value=0),
            mock.patch("pygame.midi.get_device_info", return_value=(b"mock", b"mock", 0, 1, 0)),
            mock.patch("pygame.midi.Output", return_value=mock_output),
        ):
            player = DynamicOrchestraPlayer(
                orchestra_path,
                dispatcher,
                midi_output_id=0,
                output_channel=0,
            )

        final_tempo_ratio = TEMPO_RATIO
        try:
            player.start()
            human_start = time.monotonic()

            dispatcher.broadcast(0, human_start)
            if not dispatcher.flush(timeout=1.0):
                raise AssertionError("Dispatcher failed to process index 0")

            time.sleep(0.05)
            dispatcher.broadcast(1, time.monotonic())
            if not dispatcher.flush(timeout=1.0):
                raise AssertionError("Dispatcher failed to process index 1")

            time.sleep(0.25)
            dispatcher.broadcast(2, time.monotonic())
            if not dispatcher.flush(timeout=1.0):
                raise AssertionError("Dispatcher failed to process index 2")

            wait_for_record(mock_output, event_type="note_on", note=67)
            pause_start = time.monotonic()
            time.sleep(PAUSE_SECONDS)
            pause_end = time.monotonic()

            dispatcher.broadcast(3, time.monotonic())
            if not dispatcher.flush(timeout=1.0):
                raise AssertionError("Dispatcher failed to process index 3")

            final_tempo_ratio = float(dispatcher.current_tempo_ratio)
            time.sleep(1.00)
        finally:
            player.close()
            dispatcher.close()

    expected_expression = int(round(127 * 0.40))
    assert_control_sent(
        mock_output.records,
        channel=0,
        control=ORCHESTRA_VOLUME_CC,
        value=127,
    )
    assert_control_sent(
        mock_output.records,
        channel=0,
        control=ORCHESTRA_EXPRESSION_CC,
        value=expected_expression,
    )
    assert_control_sent(
        mock_output.records,
        channel=0,
        control=ORCHESTRA_PAN_CC,
        value=64,
    )
    assert_control_sent(
        mock_output.records,
        channel=0,
        control=ORCHESTRA_REVERB_CC,
        value=0,
    )
    assert_control_sent(
        mock_output.records,
        channel=0,
        control=ORCHESTRA_CHORUS_CC,
        value=0,
    )

    note_records = [record for record in mock_output.records if record["event_type"] in {"note_on", "note_off"}]
    paired = pair_note_events(note_records)

    for required_note in (60, 64, 67, 71):
        key = (0, required_note)
        if key not in paired:
            raise AssertionError(f"Missing paired note_on/note_off for MIDI note {required_note}")

    active_before_pause = []
    for key, timing in paired.items():
        if timing["on"] <= pause_start < timing["off"]:
            active_before_pause.append((key, timing))

    if not active_before_pause:
        raise AssertionError("Expected active notes at the start of the human pause, found none")

    for key, timing in active_before_pause:
        if timing["off"] > pause_end:
            raise AssertionError(
                f"Note {key[1]} sustained past the 5-second pause: "
                f"off={timing['off'] - pause_start:.3f}s after pause start"
            )

    expected_durations = {
        (0, 60): 0.40 / TEMPO_RATIO,
        (0, 64): 0.40 / TEMPO_RATIO,
        (0, 67): 0.40 / TEMPO_RATIO,
        (0, 71): 0.20 / final_tempo_ratio,
    }
    for key, expected_duration in expected_durations.items():
        assert_close(float(paired[key]["duration"]), expected_duration, f"duration for note {key[1]}")

    overlap_60_64 = min(float(paired[(0, 60)]["off"]), float(paired[(0, 64)]["off"])) - max(
        float(paired[(0, 60)]["on"]),
        float(paired[(0, 64)]["on"]),
    )
    assert_close(overlap_60_64, 0.30 / TEMPO_RATIO, "overlap between notes 60 and 64")

    print("playback_validator: PASS")
    print(f"  note records: {len(note_records)}")
    print(f"  active notes released during pause: {len(active_before_pause)}")
    print(f"  measured overlap(60,64): {overlap_60_64:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
