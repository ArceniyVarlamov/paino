from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Iterable

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import mido

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated_dataset"
DEFAULT_TEMPO = 500000
DEFAULT_TICKS_PER_BEAT = 480

ScaleEvent = dict[str, list[int] | float]


def seconds_to_ticks(
    seconds: float,
    *,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    tempo: int = DEFAULT_TEMPO,
) -> int:
    return max(0, int(round(mido.second2tick(seconds, ticks_per_beat, tempo))))


def build_score(piece_name: str, notes: Iterable[tuple[int, float]]) -> dict[str, object]:
    return {
        "piece_name": piece_name,
        "notes": [
            {
                "index": index,
                "pitch": pitch,
                "nominal_duration": nominal_duration,
            }
            for index, (pitch, nominal_duration) in enumerate(notes)
        ],
    }


def write_midi(
    path: Path,
    events: list[ScaleEvent],
    *,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    tempo: int = DEFAULT_TEMPO,
) -> None:
    midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    pending_delay_ticks = 0
    for event in events:
        pitches = [int(pitch) for pitch in event["pitches"]]
        duration_seconds = float(event["duration"])
        delay_seconds = float(event.get("delay", 0.0))
        pending_delay_ticks += seconds_to_ticks(
            delay_seconds,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )

        for index, pitch in enumerate(pitches):
            track.append(
                mido.Message(
                    "note_on",
                    note=pitch,
                    velocity=72,
                    time=pending_delay_ticks if index == 0 else 0,
                )
            )
            if index == 0:
                pending_delay_ticks = 0

        note_off_ticks = seconds_to_ticks(
            duration_seconds,
            ticks_per_beat=ticks_per_beat,
            tempo=tempo,
        )
        for index, pitch in enumerate(pitches):
            track.append(
                mido.Message(
                    "note_off",
                    note=pitch,
                    velocity=0,
                    time=note_off_ticks if index == 0 else 0,
                )
            )
            if index == 0:
                note_off_ticks = 0

    track.append(mido.MetaMessage("end_of_track", time=0))
    midi_file.save(path)


def save_pair(
    output_dir: Path,
    piece_name: str,
    performance_events: list[ScaleEvent],
    score_notes: list[tuple[int, float]],
) -> tuple[Path, Path]:
    midi_path = output_dir / f"{piece_name}.mid"
    json_path = output_dir / f"{piece_name}.json"
    write_midi(midi_path, performance_events)
    json_path.write_text(
        json.dumps(build_score(piece_name, score_notes), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return midi_path, json_path


def ideal_case() -> tuple[list[ScaleEvent], list[tuple[int, float]]]:
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    duration = 0.5
    performance = [{"pitches": [pitch], "duration": duration} for pitch in scale]
    score = [(pitch, duration) for pitch in scale]
    return performance, score


def rubato_case() -> tuple[list[ScaleEvent], list[tuple[int, float]]]:
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    nominal_duration = 0.5
    expressive_durations = [0.7, 0.62, 0.48, 0.35, 0.28, 0.42, 0.56, 0.78]
    performance = [
        {"pitches": [pitch], "duration": duration}
        for pitch, duration in zip(scale, expressive_durations, strict=True)
    ]
    score = [(pitch, nominal_duration) for pitch in scale]
    return performance, score


def noisy_case() -> tuple[list[ScaleEvent], list[tuple[int, float]]]:
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    nominal_duration = 0.5
    rng = random.Random(20260418)
    missing_indices = {2, 6}
    performance: list[ScaleEvent] = []

    for index, pitch in enumerate(scale):
        if index in missing_indices:
            if rng.random() < 0.6:
                performance.append(
                    {"pitches": [rng.randint(58, 74)], "duration": 0.16}
                )
            continue

        noisy_pitch = pitch
        if rng.random() < 0.45:
            noisy_pitch += rng.choice([-1, 1])

        if rng.random() < 0.5:
            performance.append(
                {"pitches": [rng.randint(58, 74)], "duration": 0.12}
            )

        performance.append({"pitches": [noisy_pitch], "duration": nominal_duration})

        if rng.random() < 0.35:
            performance.append(
                {"pitches": [rng.randint(58, 74)], "duration": 0.1}
            )

    score = [(pitch, nominal_duration) for pitch in scale]
    return performance, score


def polyphonic_case() -> tuple[list[ScaleEvent], list[tuple[int, float]]]:
    chords = [
        [60, 64, 67],
        [62, 65, 69],
        [55, 59, 62],
        [60, 64, 67, 72],
    ]
    duration = 0.75
    performance = [{"pitches": chord, "duration": duration} for chord in chords]
    score = [(pitch, duration) for chord in chords for pitch in chord]
    return performance, score


def generate_dataset(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[tuple[Path, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_pairs: list[tuple[Path, Path]] = []
    for piece_name, builder in (
        ("ideal", ideal_case),
        ("rubato", rubato_case),
        ("noisy", noisy_case),
        ("polyphonic", polyphonic_case),
    ):
        performance, score = builder()
        generated_pairs.append(save_pair(output_dir, piece_name, performance, score))

    return generated_pairs


def main() -> None:
    generated_pairs = generate_dataset()
    for midi_path, json_path in generated_pairs:
        print(f"generated {midi_path.name} + {json_path.name}")


if __name__ == "__main__":
    main()
