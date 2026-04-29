from __future__ import annotations

import argparse
import json
import sys
from itertools import zip_longest
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.append(vendor_path)

import mido

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "generated_dataset"


def load_score(score_path: Path) -> dict[str, object]:
    return json.loads(score_path.read_text(encoding="utf-8"))


def load_performance(midi_path: Path) -> list[dict[str, float | int]]:
    midi_file = mido.MidiFile(midi_path)
    events: list[dict[str, float | int]] = []
    elapsed = 0.0

    for msg in midi_file:
        elapsed += float(getattr(msg, "time", 0.0))
        if getattr(msg, "type", None) == "note_on" and getattr(msg, "velocity", 0) > 0:
            events.append(
                {
                    "index": len(events),
                    "pitch": int(msg.note),
                    "timestamp": round(elapsed, 3),
                }
            )

    return events


def discover_pairs(paths: list[str]) -> list[tuple[Path, Path]]:
    if not paths:
        dataset_dir = DEFAULT_DATASET_DIR
        return sorted(
            (json_path, json_path.with_suffix(".mid"))
            for json_path in dataset_dir.glob("*.json")
            if json_path.with_suffix(".mid").exists()
        )

    if len(paths) == 1:
        candidate = Path(paths[0])
        if candidate.is_dir():
            return sorted(
                (json_path, json_path.with_suffix(".mid"))
                for json_path in candidate.glob("*.json")
                if json_path.with_suffix(".mid").exists()
            )
        raise SystemExit("Provide either a directory or both a score JSON path and a MIDI path.")

    if len(paths) == 2:
        return [(Path(paths[0]), Path(paths[1]))]

    raise SystemExit("Usage: dataset_viewer.py [dataset_dir] or dataset_viewer.py score.json performance.mid")


def render_pair(score_path: Path, midi_path: Path) -> None:
    score = load_score(score_path)
    performance = load_performance(midi_path)
    piece_name = score.get("piece_name", score_path.stem)
    score_notes = score.get("notes", [])

    print(f"=== {piece_name} ===")
    print(f"score: {score_path.name}")
    print(f"midi : {midi_path.name}")
    print(
        f"{'SCORE':<38} | {'PERFORMANCE':<38}"
    )
    print(f"{'-' * 38}-+-{'-' * 38}")

    for score_note, perf_note in zip_longest(score_notes, performance):
        score_text = ""
        perf_text = ""

        if score_note is not None:
            score_text = (
                f"#{score_note['index']:>2} pitch={score_note['pitch']:>3} "
                f"nominal_duration={score_note['nominal_duration']:.3f}"
            )
        if perf_note is not None:
            perf_text = (
                f"#{perf_note['index']:>2} pitch={perf_note['pitch']:>3} "
                f"timestamp={perf_note['timestamp']:.3f}"
            )

        print(f"{score_text:<38} | {perf_text:<38}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print paired score/performance data side-by-side."
    )
    parser.add_argument("paths", nargs="*", help="Optional dataset directory or score/midi pair.")
    args = parser.parse_args()

    pairs = discover_pairs(args.paths)
    if not pairs:
        raise SystemExit("No score/MIDI pairs found.")

    for score_path, midi_path in pairs:
        render_pair(score_path, midi_path)


if __name__ == "__main__":
    main()
