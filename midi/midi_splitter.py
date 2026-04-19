from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively split a concerto MIDI into solo and orchestra parts.",
    )
    parser.add_argument(
        "midi_file",
        type=Path,
        help="Input MIDI file to split.",
    )
    parser.add_argument(
        "--solo-out",
        type=Path,
        default=None,
        help="Optional output path for the extracted solo MIDI.",
    )
    parser.add_argument(
        "--orchestra-out",
        type=Path,
        default=None,
        help="Optional output path for the extracted orchestra MIDI.",
    )
    return parser


def clone_track(track: mido.MidiTrack) -> mido.MidiTrack:
    cloned = mido.MidiTrack()
    for message in track:
        cloned.append(message.copy(time=message.time))
    return cloned


def track_display_name(track: mido.MidiTrack, index: int) -> str:
    name = getattr(track, "name", "") or ""
    return name if name.strip() else f"<unnamed track {index}>"


def print_track_summary(midi_file: mido.MidiFile) -> None:
    print(f"\nLoaded: {midi_file.filename or '<in-memory>'}")
    print(f"Format type: {midi_file.type}    Ticks/beat: {midi_file.ticks_per_beat}")
    print("\nTracks:")
    print("  idx | name                           | messages")
    print("  ----+--------------------------------+---------")
    for index, track in enumerate(midi_file.tracks):
        name = track_display_name(track, index)
        print(f"  {index:>3} | {name[:30]:<30} | {len(track):>8}")


def prompt_solo_track(midi_file: mido.MidiFile) -> int:
    if len(midi_file.tracks) < 2:
        raise SystemExit(
            "This MIDI file has fewer than 2 tracks, so there is nothing meaningful to split."
        )

    print(
        "\nTrack 0 is copied automatically because it usually contains tempo/meta data."
    )
    prompt = "Enter the track number for the Solo Piano: "

    while True:
        raw = input(prompt).strip()
        try:
            track_index = int(raw)
        except ValueError:
            print("Please enter an integer track number.")
            continue

        if track_index == 0:
            print("Track 0 is reserved. Choose a musical track from 1 and above.")
            continue

        if 1 <= track_index < len(midi_file.tracks):
            return track_index

        print(f"Track number must be between 1 and {len(midi_file.tracks) - 1}.")


def build_split_files(
    midi_file: mido.MidiFile,
    solo_track_index: int,
) -> tuple[mido.MidiFile, mido.MidiFile]:
    solo_file = mido.MidiFile(type=midi_file.type, ticks_per_beat=midi_file.ticks_per_beat)
    orchestra_file = mido.MidiFile(type=midi_file.type, ticks_per_beat=midi_file.ticks_per_beat)

    solo_file.tracks.append(clone_track(midi_file.tracks[0]))
    orchestra_file.tracks.append(clone_track(midi_file.tracks[0]))

    solo_file.tracks.append(clone_track(midi_file.tracks[solo_track_index]))

    for track_index, track in enumerate(midi_file.tracks[1:], start=1):
        if track_index == solo_track_index:
            continue
        orchestra_file.tracks.append(clone_track(track))

    return solo_file, orchestra_file


def save_split_files(
    source_path: Path,
    solo_file: mido.MidiFile,
    orchestra_file: mido.MidiFile,
    *,
    solo_out: Path | None = None,
    orchestra_out: Path | None = None,
) -> tuple[Path, Path]:
    output_dir = source_path.resolve().parent
    solo_path = (solo_out or (output_dir / "solo.mid")).expanduser().resolve()
    orchestra_path = (orchestra_out or (output_dir / "orchestra.mid")).expanduser().resolve()

    solo_path.parent.mkdir(parents=True, exist_ok=True)
    orchestra_path.parent.mkdir(parents=True, exist_ok=True)
    solo_file.save(solo_path)
    orchestra_file.save(orchestra_path)
    return solo_path, orchestra_path


def main() -> int:
    args = build_parser().parse_args()
    midi_path = args.midi_file.expanduser().resolve()
    if not midi_path.exists():
        raise SystemExit(f"Input MIDI file not found: {midi_path}")

    try:
        midi_file = mido.MidiFile(midi_path)
    except Exception as exc:
        raise SystemExit(f"Failed to read MIDI file {midi_path}: {exc}") from exc

    print_track_summary(midi_file)
    solo_track_index = prompt_solo_track(midi_file)
    solo_track_name = track_display_name(midi_file.tracks[solo_track_index], solo_track_index)

    solo_file, orchestra_file = build_split_files(midi_file, solo_track_index)
    solo_path, orchestra_path = save_split_files(
        midi_path,
        solo_file,
        orchestra_file,
        solo_out=args.solo_out,
        orchestra_out=args.orchestra_out,
    )

    print("\nSplit complete.")
    print(f"Solo track:      {solo_track_index} ({solo_track_name})")
    print(f"Saved solo.mid:       {solo_path}")
    print(f"Saved orchestra.mid:  {orchestra_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
