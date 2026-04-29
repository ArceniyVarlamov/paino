from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_VENDOR_DIR = Path(__file__).resolve().parents[1] / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.append(vendor_path)

import mido


@dataclass(frozen=True)
class NoteSpan:
    pitch: int
    start_tick: int
    end_tick: int
    velocity: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reduce a dense orchestral MIDI into a single-track arrangement for one synth patch."
    )
    parser.add_argument("input_midi", type=Path, help="Source MIDI file.")
    parser.add_argument("output_midi", type=Path, help="Reduced output MIDI file.")
    parser.add_argument(
        "--track-indices",
        type=int,
        nargs="+",
        default=[10, 11, 12, 13, 14],
        help="Track indices to keep (default: string section tracks 10-14).",
    )
    parser.add_argument(
        "--max-polyphony",
        type=int,
        default=5,
        help="Maximum simultaneous pitches per onset cluster (default: %(default)s).",
    )
    parser.add_argument(
        "--cluster-ticks",
        type=int,
        default=12,
        help="Onset clustering tolerance in ticks (default: %(default)s).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Output MIDI channel in zero-based form (default: %(default)s).",
    )
    parser.add_argument(
        "--min-velocity",
        type=int,
        default=42,
        help="Floor for reduced note velocities (default: %(default)s).",
    )
    parser.add_argument(
        "--max-velocity",
        type=int,
        default=96,
        help="Ceiling for reduced note velocities (default: %(default)s).",
    )
    return parser


def collect_note_spans(mid: mido.MidiFile, track_indices: set[int]) -> list[NoteSpan]:
    spans: list[NoteSpan] = []
    for track_index, track in enumerate(mid.tracks):
        if track_index not in track_indices:
            continue
        absolute_tick = 0
        open_notes: dict[int, list[tuple[int, int]]] = {}
        for message in track:
            absolute_tick += int(message.time)
            if message.type == "note_on" and int(getattr(message, "velocity", 0)) > 0:
                open_notes.setdefault(int(message.note), []).append(
                    (absolute_tick, int(message.velocity))
                )
            elif message.type == "note_off" or (
                message.type == "note_on" and int(getattr(message, "velocity", 0)) == 0
            ):
                note = int(message.note)
                stack = open_notes.get(note)
                if not stack:
                    continue
                start_tick, velocity = stack.pop(0)
                if not stack:
                    open_notes.pop(note, None)
                end_tick = max(start_tick + 1, absolute_tick)
                spans.append(
                    NoteSpan(
                        pitch=note,
                        start_tick=start_tick,
                        end_tick=end_tick,
                        velocity=velocity,
                    )
                )
    spans.sort(key=lambda span: (span.start_tick, span.pitch, span.end_tick))
    return spans


def cluster_spans(spans: list[NoteSpan], cluster_ticks: int) -> list[list[NoteSpan]]:
    if not spans:
        return []
    clusters: list[list[NoteSpan]] = []
    current_cluster: list[NoteSpan] = [spans[0]]
    current_anchor = spans[0].start_tick
    for span in spans[1:]:
        if span.start_tick - current_anchor <= cluster_ticks:
            current_cluster.append(span)
            continue
        clusters.append(current_cluster)
        current_cluster = [span]
        current_anchor = span.start_tick
    clusters.append(current_cluster)
    return clusters


def spread_select(sorted_pitches: list[int], max_polyphony: int) -> set[int]:
    if len(sorted_pitches) <= max_polyphony:
        return set(sorted_pitches)
    if max_polyphony <= 1:
        return {sorted_pitches[-1]}

    selected_indices = {0, len(sorted_pitches) - 1}
    while len(selected_indices) < max_polyphony:
        fraction = len(selected_indices) / (max_polyphony - 1)
        candidate = round((len(sorted_pitches) - 1) * fraction)
        selected_indices.add(int(candidate))
        if len(selected_indices) >= max_polyphony:
            break
        for candidate in range(len(sorted_pitches)):
            if candidate not in selected_indices:
                selected_indices.add(candidate)
                break
    return {sorted_pitches[index] for index in sorted(selected_indices)[:max_polyphony]}


def reduce_clusters(
    clusters: Iterable[list[NoteSpan]],
    *,
    max_polyphony: int,
    min_velocity: int,
    max_velocity: int,
) -> list[NoteSpan]:
    reduced: list[NoteSpan] = []
    for cluster in clusters:
        by_pitch: dict[int, NoteSpan] = {}
        for span in cluster:
            existing = by_pitch.get(span.pitch)
            if existing is None:
                by_pitch[span.pitch] = span
                continue
            by_pitch[span.pitch] = NoteSpan(
                pitch=span.pitch,
                start_tick=min(existing.start_tick, span.start_tick),
                end_tick=max(existing.end_tick, span.end_tick),
                velocity=max(existing.velocity, span.velocity),
            )

        pitches = sorted(by_pitch)
        keep = spread_select(pitches, max_polyphony)
        for pitch in sorted(keep):
            span = by_pitch[pitch]
            reduced.append(
                NoteSpan(
                    pitch=span.pitch,
                    start_tick=span.start_tick,
                    end_tick=span.end_tick,
                    velocity=max(min_velocity, min(max_velocity, span.velocity)),
                )
            )
    reduced.sort(key=lambda span: (span.pitch, span.start_tick, span.end_tick))
    return truncate_overlaps(reduced)


def truncate_overlaps(spans: list[NoteSpan]) -> list[NoteSpan]:
    by_pitch: dict[int, list[NoteSpan]] = {}
    for span in spans:
        by_pitch.setdefault(span.pitch, []).append(span)

    normalized: list[NoteSpan] = []
    for pitch, pitch_spans in by_pitch.items():
        pitch_spans.sort(key=lambda span: (span.start_tick, span.end_tick))
        previous: NoteSpan | None = None
        for span in pitch_spans:
            if previous is None:
                previous = span
                continue
            if span.start_tick < previous.end_tick:
                normalized.append(
                    NoteSpan(
                        pitch=previous.pitch,
                        start_tick=previous.start_tick,
                        end_tick=max(previous.start_tick + 1, span.start_tick),
                        velocity=previous.velocity,
                    )
                )
                previous = span
                continue
            normalized.append(previous)
            previous = span
        if previous is not None:
            normalized.append(previous)
    normalized.sort(key=lambda span: (span.start_tick, span.pitch, span.end_tick))
    return normalized


def build_output_file(
    source_mid: mido.MidiFile,
    spans: list[NoteSpan],
    *,
    output_channel: int,
) -> mido.MidiFile:
    out_mid = mido.MidiFile(type=1, ticks_per_beat=source_mid.ticks_per_beat)

    meta_track = mido.MidiTrack()
    out_mid.tracks.append(meta_track)
    for message in source_mid.tracks[0]:
        if message.is_meta:
            meta_track.append(message.copy())
    if not meta_track or meta_track[-1].type != "end_of_track":
        meta_track.append(mido.MetaMessage("end_of_track", time=0))

    note_track = mido.MidiTrack()
    out_mid.tracks.append(note_track)
    note_track.append(mido.MetaMessage("track_name", name="Single Track Orchestra Reduction", time=0))

    events: list[tuple[int, int, mido.Message]] = []
    for span in spans:
        events.append(
            (
                span.start_tick,
                1,
                mido.Message(
                    "note_on",
                    note=int(span.pitch),
                    velocity=int(span.velocity),
                    channel=int(output_channel),
                    time=0,
                ),
            )
        )
        events.append(
            (
                span.end_tick,
                0,
                mido.Message(
                    "note_off",
                    note=int(span.pitch),
                    velocity=0,
                    channel=int(output_channel),
                    time=0,
                ),
            )
        )
    events.sort(key=lambda item: (item[0], item[1], item[2].note))

    previous_tick = 0
    for absolute_tick, _, message in events:
        delta = max(0, int(absolute_tick) - previous_tick)
        note_track.append(message.copy(time=delta))
        previous_tick = int(absolute_tick)
    note_track.append(mido.MetaMessage("end_of_track", time=0))
    return out_mid


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input_midi.expanduser().resolve()
    output_path = args.output_midi.expanduser().resolve()

    mid = mido.MidiFile(input_path)
    track_indices = {int(index) for index in args.track_indices}
    spans = collect_note_spans(mid, track_indices)
    clusters = cluster_spans(spans, int(args.cluster_ticks))
    reduced = reduce_clusters(
        clusters,
        max_polyphony=max(1, int(args.max_polyphony)),
        min_velocity=max(1, int(args.min_velocity)),
        max_velocity=max(1, int(args.max_velocity)),
    )
    out_mid = build_output_file(mid, reduced, output_channel=int(args.channel))
    out_mid.save(output_path)

    print(f"input_spans={len(spans)}")
    print(f"clusters={len(clusters)}")
    print(f"reduced_spans={len(reduced)}")
    print(f"wrote={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
