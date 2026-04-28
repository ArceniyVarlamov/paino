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


DEFAULT_SPLIT_POINT = 60


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactively split a MIDI into tracks or left/right hands by pitch.",
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
    parser.add_argument(
        "--left-out",
        type=Path,
        default=None,
        help="Optional output path for the extracted left-hand MIDI in pitch split mode.",
    )
    parser.add_argument(
        "--right-out",
        type=Path,
        default=None,
        help="Optional output path for the extracted right-hand MIDI in pitch split mode.",
    )
    return parser


def clone_track(track: mido.MidiTrack) -> mido.MidiTrack:
    cloned = mido.MidiTrack()
    for message in track:
        cloned.append(message.copy(time=message.time))
    return cloned


def track_has_notes(track: mido.MidiTrack) -> bool:
    return any(_is_note_message(message) for message in track)


def clone_meta_track(track: mido.MidiTrack) -> mido.MidiTrack:
    cloned = mido.MidiTrack()
    carried_time = 0
    for message in track:
        if getattr(message, "is_meta", False):
            cloned.append(message.copy(time=message.time + carried_time))
            carried_time = 0
        else:
            carried_time += int(message.time)
    return cloned


def track_display_name(track: mido.MidiTrack, index: int) -> str:
    name = getattr(track, "name", "") or ""
    return name if name.strip() else f"<unnamed track {index}>"


def pitch_display_name(midi_pitch: int) -> str:
    note_names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    octave = (int(midi_pitch) // 12) - 1
    return f"{note_names[int(midi_pitch) % 12]}{octave}"


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

    if track_has_notes(midi_file.tracks[0]):
        print(
            "\nTrack 0 contains notes in this file, so it can be selected normally."
        )
    else:
        print(
            "\nTrack 0 is copied automatically because it contains only tempo/meta data."
        )
    prompt = "Enter the track number for the Solo Piano: "

    while True:
        raw = input(prompt).strip()
        try:
            track_index = int(raw)
        except ValueError:
            print("Please enter an integer track number.")
            continue

        if track_index == 0 and not track_has_notes(midi_file.tracks[0]):
            print("Track 0 is reserved. Choose a musical track from 1 and above.")
            continue

        if 0 <= track_index < len(midi_file.tracks):
            return track_index

        print(f"Track number must be between 0 and {len(midi_file.tracks) - 1}.")


def prompt_split_mode() -> str:
    prompt = "\nSplit by (T)rack or (P)itch? "
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"t", "track", "tracks"}:
            return "tracks"
        if raw in {"p", "pitch"}:
            return "pitch"
        print("Please enter T for tracks or P for pitch.")


def prompt_split_point() -> int:
    prompt = f"Enter split point MIDI note [{DEFAULT_SPLIT_POINT} = Middle C]: "
    while True:
        raw = input(prompt).strip()
        if not raw:
            return DEFAULT_SPLIT_POINT

        try:
            split_point = int(raw)
        except ValueError:
            print("Please enter an integer MIDI pitch from 0 to 127.")
            continue

        if 0 <= split_point <= 127:
            return split_point

        print("Split point must be between 0 and 127.")


def build_split_files(
    midi_file: mido.MidiFile,
    solo_track_index: int,
) -> tuple[mido.MidiFile, mido.MidiFile]:
    solo_file = mido.MidiFile(type=midi_file.type, ticks_per_beat=midi_file.ticks_per_beat)
    orchestra_file = mido.MidiFile(type=midi_file.type, ticks_per_beat=midi_file.ticks_per_beat)

    if not 0 <= solo_track_index < len(midi_file.tracks):
        raise ValueError(f"solo_track_index out of range: {solo_track_index}")

    first_track_has_notes = track_has_notes(midi_file.tracks[0])
    if not first_track_has_notes:
        solo_file.tracks.append(clone_track(midi_file.tracks[0]))
        orchestra_file.tracks.append(clone_track(midi_file.tracks[0]))
    elif solo_track_index != 0:
        meta_track = clone_meta_track(midi_file.tracks[0])
        if len(meta_track) > 0:
            solo_file.tracks.append(meta_track)

    solo_file.tracks.append(clone_track(midi_file.tracks[solo_track_index]))

    for track_index, track in enumerate(midi_file.tracks):
        if track_index == solo_track_index:
            continue
        if track_index == 0 and not first_track_has_notes:
            continue
        orchestra_file.tracks.append(clone_track(track))

    return solo_file, orchestra_file


def _is_note_message(message: mido.Message) -> bool:
    return getattr(message, "type", None) in {"note_on", "note_off"}


def _filtered_track(
    track: mido.MidiTrack,
    *,
    keep_note: Any,
    keep_non_note: bool,
) -> mido.MidiTrack:
    filtered = mido.MidiTrack()
    carried_time = 0

    for message in track:
        keep_message = False
        if getattr(message, "is_meta", False):
            keep_message = True
        elif _is_note_message(message):
            keep_message = bool(keep_note(message))
        else:
            keep_message = keep_non_note

        if keep_message:
            filtered.append(message.copy(time=message.time + carried_time))
            carried_time = 0
            continue

        carried_time += int(message.time)

    return filtered


def build_pitch_split_files(
    midi_file: mido.MidiFile,
    *,
    split_point: int,
) -> tuple[mido.MidiFile, mido.MidiFile]:
    left_file = mido.MidiFile(type=midi_file.type, ticks_per_beat=midi_file.ticks_per_beat)
    right_file = mido.MidiFile(type=midi_file.type, ticks_per_beat=midi_file.ticks_per_beat)

    for track_index, track in enumerate(midi_file.tracks):
        keep_non_note = True
        left_track = _filtered_track(
            track,
            keep_note=lambda message: int(message.note) < split_point,
            keep_non_note=keep_non_note,
        )
        right_track = _filtered_track(
            track,
            keep_note=lambda message: int(message.note) >= split_point,
            keep_non_note=keep_non_note,
        )

        if len(left_track) > 0 or track_index == 0:
            left_file.tracks.append(left_track)
        if len(right_track) > 0 or track_index == 0:
            right_file.tracks.append(right_track)

    return left_file, right_file


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


def save_pitch_split_files(
    source_path: Path,
    left_file: mido.MidiFile,
    right_file: mido.MidiFile,
    *,
    left_out: Path | None = None,
    right_out: Path | None = None,
) -> tuple[Path, Path]:
    output_dir = source_path.resolve().parent
    left_path = (left_out or (output_dir / "left_hand.mid")).expanduser().resolve()
    right_path = (right_out or (output_dir / "right_hand.mid")).expanduser().resolve()

    left_path.parent.mkdir(parents=True, exist_ok=True)
    right_path.parent.mkdir(parents=True, exist_ok=True)
    left_file.save(left_path)
    right_file.save(right_path)
    return left_path, right_path


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
    split_mode = prompt_split_mode()

    if split_mode == "tracks":
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

        print("\nTrack split complete.")
        print(f"Solo track:           {solo_track_index} ({solo_track_name})")
        print(f"Saved solo.mid:       {solo_path}")
        print(f"Saved orchestra.mid:  {orchestra_path}")
        return 0

    split_point = prompt_split_point()
    left_file, right_file = build_pitch_split_files(midi_file, split_point=split_point)
    left_path, right_path = save_pitch_split_files(
        midi_path,
        left_file,
        right_file,
        left_out=args.left_out,
        right_out=args.right_out,
    )

    print("\nPitch split complete.")
    print(f"Split point:          {split_point} ({pitch_display_name(split_point)})")
    print(f"Saved left_hand.mid:  {left_path}")
    print(f"Saved right_hand.mid: {right_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
