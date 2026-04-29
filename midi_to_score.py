from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.append(vendor_path)

import mido


@dataclass
class NoteEvent:
    pitch: int
    start_time: float
    order: int
    end_time: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a MIDI file into the project's score JSON format.",
    )
    parser.add_argument("midi_file", type=Path, help="Input MIDI file.")
    parser.add_argument(
        "output_json",
        nargs="?",
        type=Path,
        help="Optional output JSON path. Defaults to the MIDI stem with .json.",
    )
    parser.add_argument(
        "--chord-policy",
        choices=("chord", "highest", "lowest", "first", "flatten"),
        default="chord",
        help=(
            "How to represent simultaneous note onsets. "
            "`chord` keeps all notes together in one score state."
        ),
    )
    parser.add_argument(
        "--chord-epsilon",
        type=float,
        default=0.03,
        help="Seconds within which note_on events are treated as the same onset group.",
    )
    parser.add_argument(
        "--default-duration",
        type=float,
        default=0.5,
        help="Fallback duration in seconds when a note-off is unavailable.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.05,
        help="Minimum duration assigned to any emitted score note.",
    )
    return parser


def extract_note_events(midi_path: Path) -> list[NoteEvent]:
    midi_file = mido.MidiFile(midi_path)
    active_notes: dict[tuple[int, int], deque[NoteEvent]] = defaultdict(deque)
    note_events: list[NoteEvent] = []
    elapsed = 0.0

    for msg in midi_file:
        elapsed += float(getattr(msg, "time", 0.0))
        msg_type = getattr(msg, "type", None)
        velocity = int(getattr(msg, "velocity", 0))
        channel = int(getattr(msg, "channel", 0))

        if msg_type == "note_on" and velocity > 0:
            event = NoteEvent(
                pitch=int(msg.note),
                start_time=elapsed,
                order=len(note_events),
            )
            active_notes[(channel, event.pitch)].append(event)
            note_events.append(event)
            continue

        if msg_type == "note_off" or (msg_type == "note_on" and velocity == 0):
            pitch = int(msg.note)
            key = (channel, pitch)
            if active_notes[key]:
                active_notes[key].popleft().end_time = elapsed

    return note_events


def cluster_note_events(note_events: list[NoteEvent], epsilon: float) -> list[list[NoteEvent]]:
    if not note_events:
        return []

    sorted_events = sorted(note_events, key=lambda event: (event.start_time, event.order))
    clusters: list[list[NoteEvent]] = []
    current_cluster: list[NoteEvent] = []
    cluster_anchor = 0.0

    for event in sorted_events:
        if not current_cluster:
            current_cluster = [event]
            cluster_anchor = event.start_time
            continue

        if event.start_time - cluster_anchor <= epsilon:
            current_cluster.append(event)
            continue

        clusters.append(current_cluster)
        current_cluster = [event]
        cluster_anchor = event.start_time

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def select_cluster_events(cluster: list[NoteEvent], chord_policy: str) -> list[NoteEvent]:
    if chord_policy in {"chord", "flatten"}:
        return sorted(cluster, key=lambda event: (event.start_time, event.order, event.pitch))

    if chord_policy == "highest":
        return [max(cluster, key=lambda event: (event.pitch, -event.order))]

    if chord_policy == "lowest":
        return [min(cluster, key=lambda event: (event.pitch, event.order))]

    return [min(cluster, key=lambda event: event.order)]


def cluster_onset_time(cluster: list[NoteEvent]) -> float:
    return min(float(event.start_time) for event in cluster)


def cluster_state_duration(
    cluster: list[NoteEvent],
    next_onset_time: float | None,
    *,
    default_duration: float,
    min_duration: float,
) -> float:
    onset_time = cluster_onset_time(cluster)
    if next_onset_time is not None and next_onset_time > onset_time:
        return max(min_duration, float(next_onset_time - onset_time))

    latest_end_time = max(
        (
            float(event.end_time)
            for event in cluster
            if event.end_time is not None and event.end_time > onset_time
        ),
        default=onset_time + default_duration,
    )
    return max(min_duration, float(latest_end_time - onset_time))


def convert_to_score(
    midi_path: Path,
    *,
    chord_policy: str,
    chord_epsilon: float,
    default_duration: float,
    min_duration: float,
) -> dict[str, object]:
    note_events = extract_note_events(midi_path)
    if not note_events:
        raise ValueError(f"No note_on events found in {midi_path}")

    onset_clusters = cluster_note_events(note_events, chord_epsilon)
    notes: list[dict[str, object]] = []

    for index, cluster in enumerate(onset_clusters):
        emitted_events = select_cluster_events(cluster, chord_policy)
        next_onset_time = (
            cluster_onset_time(onset_clusters[index + 1])
            if index + 1 < len(onset_clusters)
            else None
        )
        onset_time = cluster_onset_time(emitted_events)
        pitches = sorted({int(event.pitch) for event in emitted_events})
        duration = cluster_state_duration(
            emitted_events,
            next_onset_time,
            default_duration=default_duration,
            min_duration=min_duration,
        )

        notes.append(
            {
                "index": len(notes),
                "pitches": pitches,
                "nominal_onset": round(onset_time, 6),
                "nominal_duration": round(duration, 6),
            }
        )

    return {
        "piece_name": midi_path.stem,
        "notes": notes,
    }


def main() -> int:
    args = build_parser().parse_args()
    output_json = args.output_json or args.midi_file.with_suffix(".json")

    score = convert_to_score(
        args.midi_file,
        chord_policy=args.chord_policy,
        chord_epsilon=args.chord_epsilon,
        default_duration=args.default_duration,
        min_duration=args.min_duration,
    )

    output_json.write_text(
        json.dumps(score, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Converted {args.midi_file.name} -> {output_json.name} "
        f"({len(score['notes'])} score states, policy={args.chord_policy})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
