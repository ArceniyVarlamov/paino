from __future__ import annotations

import queue
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import mido

import midi_generator
import live_midi_receiver as input_module


def wait_until(predicate, timeout: float = 1.0, interval: float = 0.01) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def create_empty_midi(path: Path) -> None:
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi_file.save(path)


def create_dense_midi(path: Path, note_count: int, duration: float = 0.0) -> None:
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

    ticks = max(0, int(round(mido.second2tick(duration, midi_file.ticks_per_beat, 500000))))
    for index in range(note_count):
        pitch = 60 + (index % 12)
        track.append(mido.Message("note_on", note=pitch, velocity=64, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=ticks))

    track.append(mido.MetaMessage("end_of_track", time=0))
    midi_file.save(path)


def count_note_ons(midi_path: Path) -> int:
    midi_file = mido.MidiFile(midi_path)
    return sum(
        1
        for msg in midi_file
        if getattr(msg, "type", None) == "note_on" and getattr(msg, "velocity", 0) > 0
    )


class FakePort:
    def __init__(
        self,
        message_batches: list[list[SimpleNamespace]] | None = None,
        *,
        raise_after_batches: bool = False,
    ) -> None:
        self._message_batches = list(message_batches or [])
        self._raise_after_batches = raise_after_batches
        self._closed = False
        self.close_calls = 0
        self.iter_calls = 0

    def iter_pending(self) -> list[SimpleNamespace]:
        self.iter_calls += 1
        if self._closed:
            raise OSError("port already closed")

        if self._message_batches:
            return self._message_batches.pop(0)

        if self._raise_after_batches:
            raise OSError("simulated disconnection")

        return []

    def close(self) -> None:
        self._closed = True
        self.close_calls += 1


def case_empty_file(work_dir: Path) -> str:
    empty_path = work_dir / "empty.mid"
    create_empty_midi(empty_path)
    emulator = input_module.MidiEmulator(empty_path)

    emulator.start()
    assert wait_until(lambda: not emulator.is_running, timeout=0.5), "emulator did not stop"
    events = emulator.get_events()
    emulator.close()

    assert events == [], f"expected no events, got {events}"
    return "empty file produced no note events"


def case_overflow() -> str:
    events = queue.Queue(maxsize=128)
    for index in range(10_000):
        input_module._push_event(events, index % 128, timestamp=float(index))

    assert events.qsize() == 128, f"expected bounded queue, got {events.qsize()}"
    drained = []
    while not events.empty():
        drained.append(events.get_nowait())

    assert drained[-1]["timestamp"] == 9999.0, "latest event was not retained"
    assert drained[0]["timestamp"] == 9872.0, "oldest retained event did not roll forward"
    return "queue stayed bounded and retained the newest events"


def case_disconnection() -> str:
    port = FakePort(
        [
            [SimpleNamespace(type="note_on", note=60, velocity=90)],
            [],
        ],
        raise_after_batches=True,
    )

    with mock.patch.object(input_module.mido, "open_input", return_value=port):
        receiver = input_module.LiveMidiReceiver(
            "fake-port",
            poll_interval=0.001,
            open_immediately=False,
        )
        receiver.start()
        assert wait_until(lambda: not receiver.is_running, timeout=0.5), "receiver thread hung"
        events = receiver.get_events()
        receiver.close()

    assert [event["pitch"] for event in events] == [60], f"unexpected events: {events}"
    assert port.close_calls >= 1, "port.close() was not called"
    return "receiver exited cleanly after simulated disconnection"


def case_concurrency(work_dir: Path) -> str:
    shared_queue: input_module.MidiEventQueue = queue.Queue(maxsize=256)
    generated_dir = work_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    midi_generator.generate_dataset(generated_dir)

    emulator = input_module.MidiEmulator(
        generated_dir / "ideal.mid",
        event_queue=shared_queue,
    )
    receiver_port = FakePort(
        [
            [SimpleNamespace(type="note_on", note=72, velocity=64)],
            [SimpleNamespace(type="note_on", note=74, velocity=64)],
            [SimpleNamespace(type="note_on", note=76, velocity=64)],
            [],
        ]
    )

    with mock.patch.object(input_module.mido, "open_input", return_value=receiver_port):
        receiver = input_module.LiveMidiReceiver(
            "fake-port",
            event_queue=shared_queue,
            poll_interval=0.001,
            open_immediately=False,
        )
        receiver.start()
        emulator.start()

        assert wait_until(lambda: not emulator.is_running, timeout=6.0), "emulator did not finish"
        time.sleep(0.05)
        receiver.close()

    events = input_module._drain_queue(shared_queue)
    observed_pitches = {event["pitch"] for event in events}
    expected_scale = {60, 62, 64, 65, 67, 69, 71, 72}
    expected_live = {72, 74, 76}

    assert expected_scale.issubset(observed_pitches), f"missing emulator notes: {expected_scale - observed_pitches}"
    assert expected_live.issubset(observed_pitches), f"missing receiver notes: {expected_live - observed_pitches}"
    return f"captured {len(events)} events from both producers without errors"


def case_reopening(work_dir: Path) -> str:
    midi_path = work_dir / "reopen.mid"
    create_dense_midi(midi_path, note_count=3, duration=0.01)
    emulator_queue: input_module.MidiEventQueue = queue.Queue()
    emulator = input_module.MidiEmulator(midi_path, event_queue=emulator_queue)

    for _ in range(3):
        emulator.start()
        assert wait_until(lambda: not emulator.is_running, timeout=1.0), "emulator cycle hung"
        emulator.close()

    emulator_events = input_module._drain_queue(emulator_queue)
    assert len(emulator_events) == 9, f"expected 9 emulator events, got {len(emulator_events)}"

    receiver_queue: input_module.MidiEventQueue = queue.Queue()
    ports = [
        FakePort([[SimpleNamespace(type="note_on", note=60 + index, velocity=64)]])
        for index in range(3)
    ]

    def open_input(_port_name: str | None = None):
        if not ports:
            raise RuntimeError("no more fake ports available")
        return ports.pop(0)

    with mock.patch.object(input_module.mido, "open_input", side_effect=open_input):
        receiver = input_module.LiveMidiReceiver(
            "fake-port",
            event_queue=receiver_queue,
            poll_interval=0.001,
            open_immediately=False,
        )
        for _ in range(3):
            receiver.start()
            time.sleep(0.02)
            receiver.close()

    receiver_events = input_module._drain_queue(receiver_queue)
    assert len(receiver_events) == 3, f"expected 3 receiver events, got {len(receiver_events)}"
    return "start/close cycles worked repeatedly for both classes"


def case_existing_midi(path: Path) -> str:
    assert path.exists(), "file is missing"
    note_count = count_note_ons(path)
    assert note_count > 0, "no note_on events found"

    emulator = input_module.MidiEmulator(path)
    emulator.start()
    time.sleep(0.05)
    emulator.close()
    return f"parsed successfully with {note_count} note_on events"


def run_case(name: str, func) -> bool:
    try:
        detail = func()
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        return False

    print(f"[PASS] {name}: {detail}")
    return True


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    generated_dir = project_dir / "generated_dataset"
    midi_generator.generate_dataset(generated_dir)

    results: list[bool] = []
    with tempfile.TemporaryDirectory(prefix="input-module-tests-") as temp_dir:
        work_dir = Path(temp_dir)

        results.append(run_case("EMPTY FILE", lambda: case_empty_file(work_dir)))
        results.append(run_case("OVERFLOW", case_overflow))
        results.append(run_case("DISCONNECTION", case_disconnection))
        results.append(run_case("CONCURRENCY", lambda: case_concurrency(work_dir)))
        results.append(run_case("RE-OPENING", lambda: case_reopening(work_dir)))
        results.append(
            run_case(
                "EXISTING MIDI 89876_In-the-Pool.mid",
                lambda: case_existing_midi(project_dir / "89876_In-the-Pool.mid"),
            )
        )
        results.append(
            run_case(
                "EXISTING MIDI 破旧世界 (完整版) (Remake).mid",
                lambda: case_existing_midi(
                    project_dir / "破旧世界 (完整版) (Remake).mid"
                ),
            )
        )

    passed = sum(1 for result in results if result)
    total = len(results)
    status = "PASS" if passed == total else "FAIL"
    print(f"\n[{status}] Summary: {passed}/{total} cases passed")


if __name__ == "__main__":
    main()
