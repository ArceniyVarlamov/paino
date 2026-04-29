from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable

from compat import compat_zip

PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"

for candidate in (PROJECT_ROOT, VENDOR_DIR):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "numpy is not installed. Install it first, for example with: pip install numpy"
    ) from exc

try:
    import mido
except ModuleNotFoundError as exc:
    raise SystemExit(
        "mido is not installed. Install it first, for example with: pip install mido"
    ) from exc


LH = 0
RH = 1

DEFAULT_TEMPO = 500000
MASSIVE_COST = 1e9

# Cost model constants.
MIDLINE_PITCH = 60.0
LEFT_HAND_CENTER = 50.0
RIGHT_HAND_CENTER = 70.0
REGISTER_SIGMA = 15.0
REGISTER_WRONG_SIDE_WEIGHT = 0.08

MAX_CHORD_SPAN = 15.0
COMFORTABLE_CHORD_SPAN = 11.0
STRICT_OVERLAP_SEC = 0.18
OVERLAP_DECAY_SEC = 0.70
OVERLAP_WEIGHT = 2.5
CLUSTER_SPAN_WEIGHT = 4.0

BASE_JUMP_ALLOWANCE = 7.0
JUMP_TIME_GAIN = 26.0
JUMP_WEIGHT = 0.35

VOICE_CROSS_MARGIN = 3.0
VOICE_CROSS_WEIGHT = 8.0


@dataclass(frozen=True)
class Note:
    index: int
    pitch: int
    velocity: int
    channel: int
    onset_tick: int
    offset_tick: int
    onset_sec: float
    offset_sec: float


@dataclass(frozen=True)
class TimedMessage:
    abs_tick: int
    abs_sec: float
    order: int
    msg: mido.Message | mido.MetaMessage


@dataclass
class PathContext:
    active_notes: dict[int, list[Note]]
    last_note: dict[int, Note | None]

    @classmethod
    def empty(cls) -> "PathContext":
        return cls(active_notes={LH: [], RH: []}, last_note={LH: None, RH: None})

    def pruned(self, onset_tick: int) -> "PathContext":
        return PathContext(
            active_notes={
                LH: [note for note in self.active_notes[LH] if note.offset_tick > onset_tick],
                RH: [note for note in self.active_notes[RH] if note.offset_tick > onset_tick],
            },
            last_note={LH: self.last_note[LH], RH: self.last_note[RH]},
        )

    def with_assigned_note(self, hand: int, note: Note) -> "PathContext":
        next_context = self.pruned(note.onset_tick)
        next_context.active_notes[hand].append(note)
        next_context.last_note[hand] = note
        return next_context


@dataclass
class DPState:
    cost: float
    context: PathContext


@dataclass(frozen=True)
class HandSplitResult:
    source_midi: Path
    left_out: Path
    right_out: Path
    left_notes: int
    right_notes: int
    left_summary: str
    right_summary: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split a standard piano MIDI into left_hand.mid and right_hand.mid using "
            "a cost-based dynamic programming hand assignment."
        )
    )
    parser.add_argument("midi_path", type=Path, help="Input MIDI file.")
    parser.add_argument(
        "--left-out",
        type=Path,
        default=None,
        help="Optional output path for the left-hand MIDI.",
    )
    parser.add_argument(
        "--right-out",
        type=Path,
        default=None,
        help="Optional output path for the right-hand MIDI.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print split statistics.",
    )
    return parser


def is_note_message(message: mido.Message | mido.MetaMessage) -> bool:
    return getattr(message, "type", None) in {"note_on", "note_off"}


def is_note_off(message: mido.Message | mido.MetaMessage) -> bool:
    return getattr(message, "type", None) == "note_off" or (
        getattr(message, "type", None) == "note_on"
        and int(getattr(message, "velocity", 0)) <= 0
    )


def parse_midi(midi_path: Path) -> tuple[mido.MidiFile, list[Note], list[TimedMessage], int]:
    midi_file = mido.MidiFile(midi_path)
    if midi_file.type == 2:
        raise ValueError("Type-2 MIDI files are not supported by this standalone splitter.")

    merged_track = mido.merge_tracks(midi_file.tracks)
    open_notes: dict[tuple[int, int], Deque[tuple[int, float, int]]] = defaultdict(deque)
    notes: list[Note] = []
    timed_messages: list[TimedMessage] = []

    abs_tick = 0
    abs_sec = 0.0
    tempo = DEFAULT_TEMPO

    for order, message in enumerate(merged_track):
        delta_ticks = int(message.time)
        abs_tick += delta_ticks
        abs_sec += mido.tick2second(delta_ticks, midi_file.ticks_per_beat, tempo)

        timed_messages.append(
            TimedMessage(
                abs_tick=abs_tick,
                abs_sec=abs_sec,
                order=order,
                msg=message.copy(time=0),
            )
        )

        if getattr(message, "type", None) == "set_tempo":
            tempo = int(message.tempo)
            continue

        if getattr(message, "type", None) == "note_on" and int(message.velocity) > 0:
            key = (int(message.channel), int(message.note))
            open_notes[key].append((abs_tick, abs_sec, int(message.velocity)))
            continue

        if is_note_off(message):
            key = (int(message.channel), int(message.note))
            if not open_notes[key]:
                continue
            onset_tick, onset_sec, velocity = open_notes[key].popleft()
            notes.append(
                Note(
                    index=len(notes),
                    pitch=int(message.note),
                    velocity=int(velocity),
                    channel=int(message.channel),
                    onset_tick=int(onset_tick),
                    offset_tick=int(abs_tick),
                    onset_sec=float(onset_sec),
                    offset_sec=float(abs_sec),
                )
            )

    source_end_tick = abs_tick
    source_end_sec = abs_sec
    for (channel, pitch), stack in open_notes.items():
        while stack:
            onset_tick, onset_sec, velocity = stack.popleft()
            notes.append(
                Note(
                    index=len(notes),
                    pitch=int(pitch),
                    velocity=int(velocity),
                    channel=int(channel),
                    onset_tick=int(onset_tick),
                    offset_tick=int(source_end_tick),
                    onset_sec=float(onset_sec),
                    offset_sec=float(source_end_sec),
                )
            )

    notes.sort(key=lambda note: (note.onset_tick, note.pitch, note.offset_tick, note.index))
    notes = [
        Note(
            index=index,
            pitch=note.pitch,
            velocity=note.velocity,
            channel=note.channel,
            onset_tick=note.onset_tick,
            offset_tick=note.offset_tick,
            onset_sec=note.onset_sec,
            offset_sec=note.offset_sec,
        )
        for index, note in enumerate(notes)
    ]
    return midi_file, notes, timed_messages, source_end_tick


def register_penalty(note: Note, hand: int) -> float:
    center = LEFT_HAND_CENTER if hand == LH else RIGHT_HAND_CENTER
    distance = (float(note.pitch) - center) / REGISTER_SIGMA
    penalty = 0.35 * float(distance * distance)

    wrong_side = max(0.0, float(note.pitch) - MIDLINE_PITCH) if hand == LH else max(
        0.0, MIDLINE_PITCH - float(note.pitch)
    )
    penalty += REGISTER_WRONG_SIDE_WEIGHT * ((wrong_side / 6.0) ** 2)
    return float(penalty)


def same_hand_overlap_penalty(note: Note, active_notes: Iterable[Note]) -> float:
    active_notes = list(active_notes)
    if not active_notes:
        return 0.0

    pitches = np.array([active.pitch for active in active_notes], dtype=np.float64)
    onset_gaps = np.array(
        [max(0.0, note.onset_sec - active.onset_sec) for active in active_notes],
        dtype=np.float64,
    )
    intervals = np.abs(pitches - float(note.pitch))

    if np.any((intervals > MAX_CHORD_SPAN) & (onset_gaps <= STRICT_OVERLAP_SEC)):
        return MASSIVE_COST

    overlap_weights = np.exp(-onset_gaps / OVERLAP_DECAY_SEC)
    interval_excess = np.clip(intervals - COMFORTABLE_CHORD_SPAN, 0.0, None)
    penalty = OVERLAP_WEIGHT * float(np.sum(overlap_weights * interval_excess * interval_excess))

    recent_mask = onset_gaps <= STRICT_OVERLAP_SEC
    if np.any(recent_mask):
        recent_pitches = pitches[recent_mask]
        cluster_span = max(float(np.max(recent_pitches)), float(note.pitch)) - min(
            float(np.min(recent_pitches)), float(note.pitch)
        )
        if cluster_span > MAX_CHORD_SPAN:
            return MASSIVE_COST
        cluster_excess = max(0.0, cluster_span - COMFORTABLE_CHORD_SPAN)
        penalty += CLUSTER_SPAN_WEIGHT * (cluster_excess * cluster_excess)

    return float(penalty)


def horizontal_jump_penalty(note: Note, previous_note: Note | None) -> float:
    if previous_note is None:
        return 0.0
    if previous_note.offset_tick > note.onset_tick:
        return 0.0
    if previous_note.onset_tick == note.onset_tick:
        return 0.0

    jump = abs(float(note.pitch) - float(previous_note.pitch))
    travel_time = max(0.0, note.onset_sec - previous_note.offset_sec)
    allowed_jump = BASE_JUMP_ALLOWANCE + (JUMP_TIME_GAIN * travel_time)
    excess = max(0.0, jump - allowed_jump)
    return float(JUMP_WEIGHT * excess * excess)


def voice_crossing_penalty(note: Note, hand: int, opposite_active: Iterable[Note]) -> float:
    opposite_active = list(opposite_active)
    if not opposite_active:
        return 0.0

    pitches = np.array([active.pitch for active in opposite_active], dtype=np.float64)
    if hand == RH:
        boundary = float(np.max(pitches))
        crossing = boundary - float(note.pitch) + VOICE_CROSS_MARGIN
    else:
        boundary = float(np.min(pitches))
        crossing = float(note.pitch) - boundary + VOICE_CROSS_MARGIN

    if crossing <= 0.0:
        return 0.0
    return float(VOICE_CROSS_WEIGHT * ((crossing / 3.0) ** 2))


def assignment_cost(note: Note, hand: int, context: PathContext) -> float:
    same_hand_active = context.active_notes[hand]
    opposite_active = context.active_notes[RH if hand == LH else LH]

    cost = 0.0
    cost += register_penalty(note, hand)
    cost += same_hand_overlap_penalty(note, same_hand_active)
    cost += horizontal_jump_penalty(note, context.last_note[hand])
    cost += voice_crossing_penalty(note, hand, opposite_active)
    return float(cost)


def viterbi_split(notes: list[Note]) -> list[int]:
    if not notes:
        return []

    backpointers = np.full((len(notes), 2), -1, dtype=np.int16)
    current_states: list[DPState | None] = [None, None]

    first_context = PathContext.empty()
    for hand in (LH, RH):
        cost = assignment_cost(notes[0], hand, first_context)
        current_states[hand] = DPState(
            cost=cost,
            context=first_context.with_assigned_note(hand, notes[0]),
        )

    for note_index in range(1, len(notes)):
        note = notes[note_index]
        next_states: list[DPState | None] = [None, None]

        for hand in (LH, RH):
            best_cost = math.inf
            best_prev_hand = -1
            best_context: PathContext | None = None

            for prev_hand in (LH, RH):
                prev_state = current_states[prev_hand]
                if prev_state is None:
                    continue

                pruned_context = prev_state.context.pruned(note.onset_tick)
                transition_cost = assignment_cost(note, hand, pruned_context)
                total_cost = prev_state.cost + transition_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_prev_hand = prev_hand
                    best_context = pruned_context.with_assigned_note(hand, note)

            if best_context is None:
                raise RuntimeError("Dynamic programming failed to produce a valid hand state.")

            backpointers[note_index, hand] = best_prev_hand
            next_states[hand] = DPState(cost=best_cost, context=best_context)

        current_states = next_states

    final_hand = LH
    if current_states[RH] is not None and current_states[LH] is not None:
        final_hand = RH if current_states[RH].cost < current_states[LH].cost else LH

    assignments = [LH] * len(notes)
    hand = final_hand
    for note_index in range(len(notes) - 1, -1, -1):
        assignments[note_index] = hand
        if note_index == 0:
            break
        hand = int(backpointers[note_index, hand])
        if hand not in (LH, RH):
            raise RuntimeError("Backtracking failed due to an invalid predecessor hand.")

    return assignments


def should_copy_message(message: mido.Message | mido.MetaMessage) -> bool:
    if is_note_message(message):
        return False
    message_type = getattr(message, "type", None)
    if message_type in {"track_name", "end_of_track"}:
        return False
    return True


def event_priority(message: mido.Message | mido.MetaMessage) -> int:
    message_type = getattr(message, "type", None)
    if message_type == "note_off" or (message_type == "note_on" and int(getattr(message, "velocity", 0)) <= 0):
        return 2
    if message_type == "note_on":
        return 3
    if getattr(message, "is_meta", False):
        return 0
    return 1


def build_output_midi(
    source_midi: mido.MidiFile,
    timed_messages: list[TimedMessage],
    selected_notes: list[Note],
    hand_name: str,
    source_end_tick: int,
) -> mido.MidiFile:
    output = mido.MidiFile(type=0, ticks_per_beat=source_midi.ticks_per_beat)
    track = mido.MidiTrack()
    output.tracks.append(track)

    absolute_events: list[tuple[int, int, int, mido.Message | mido.MetaMessage]] = []
    absolute_events.append(
        (
            0,
            0,
            -1,
            mido.MetaMessage("track_name", name=hand_name, time=0),
        )
    )

    for timed_message in timed_messages:
        if not should_copy_message(timed_message.msg):
            continue
        absolute_events.append(
            (
                timed_message.abs_tick,
                event_priority(timed_message.msg),
                timed_message.order,
                timed_message.msg.copy(time=0),
            )
        )

    note_order_base = len(timed_messages) + 1
    for note in selected_notes:
        absolute_events.append(
            (
                note.onset_tick,
                3,
                note_order_base + (note.index * 2),
                mido.Message(
                    "note_on",
                    note=note.pitch,
                    velocity=note.velocity,
                    channel=note.channel,
                    time=0,
                ),
            )
        )
        absolute_events.append(
            (
                note.offset_tick,
                2,
                note_order_base + (note.index * 2) + 1,
                mido.Message(
                    "note_off",
                    note=note.pitch,
                    velocity=0,
                    channel=note.channel,
                    time=0,
                ),
            )
        )

    absolute_events.sort(key=lambda event: (event[0], event[1], event[2]))

    last_tick = 0
    for abs_tick, _, _, message in absolute_events:
        delta = int(abs_tick - last_tick)
        if delta < 0:
            raise RuntimeError("Absolute event ordering became invalid during MIDI reconstruction.")
        track.append(message.copy(time=delta))
        last_tick = abs_tick

    trailing_delta = max(0, int(source_end_tick - last_tick))
    track.append(mido.MetaMessage("end_of_track", time=trailing_delta))
    return output


def split_notes_by_assignment(notes: list[Note], assignments: list[int]) -> tuple[list[Note], list[Note]]:
    left_notes = [note for note, hand in compat_zip(notes, assignments, strict=True) if hand == LH]
    right_notes = [note for note, hand in compat_zip(notes, assignments, strict=True) if hand == RH]
    return left_notes, right_notes


def summarize_hand(notes: list[Note]) -> str:
    if not notes:
        return "0 notes"
    pitches = np.array([note.pitch for note in notes], dtype=np.float64)
    return (
        f"{len(notes)} notes | range {int(np.min(pitches))}-{int(np.max(pitches))} "
        f"| mean pitch {float(np.mean(pitches)):.2f}"
    )


def default_output_paths(midi_path: Path) -> tuple[Path, Path]:
    output_dir = midi_path.resolve().parent
    return output_dir / "left_hand.mid", output_dir / "right_hand.mid"


def split_midi_file(
    midi_path: Path,
    *,
    left_out: Path | None = None,
    right_out: Path | None = None,
) -> HandSplitResult:
    midi_path = midi_path.expanduser().resolve()
    if not midi_path.exists():
        raise FileNotFoundError(f"Input MIDI file does not exist: {midi_path}")

    resolved_left_out, resolved_right_out = default_output_paths(midi_path)
    if left_out is not None:
        resolved_left_out = left_out.expanduser().resolve()
    if right_out is not None:
        resolved_right_out = right_out.expanduser().resolve()

    source_midi, notes, timed_messages, source_end_tick = parse_midi(midi_path)
    if not notes:
        raise ValueError(f"No note events found in: {midi_path}")

    assignments = viterbi_split(notes)
    left_notes, right_notes = split_notes_by_assignment(notes, assignments)

    left_midi = build_output_midi(
        source_midi=source_midi,
        timed_messages=timed_messages,
        selected_notes=left_notes,
        hand_name="Left Hand",
        source_end_tick=source_end_tick,
    )
    right_midi = build_output_midi(
        source_midi=source_midi,
        timed_messages=timed_messages,
        selected_notes=right_notes,
        hand_name="Right Hand",
        source_end_tick=source_end_tick,
    )

    resolved_left_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_right_out.parent.mkdir(parents=True, exist_ok=True)
    left_midi.save(resolved_left_out)
    right_midi.save(resolved_right_out)

    return HandSplitResult(
        source_midi=midi_path,
        left_out=resolved_left_out,
        right_out=resolved_right_out,
        left_notes=len(left_notes),
        right_notes=len(right_notes),
        left_summary=summarize_hand(left_notes),
        right_summary=summarize_hand(right_notes),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = split_midi_file(
            args.midi_path,
            left_out=args.left_out,
            right_out=args.right_out,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Input: {result.source_midi}")
    print(f"Left hand:  {result.left_out}")
    print(f"Right hand: {result.right_out}")
    if args.verbose:
        print(f"LH summary: {result.left_summary}")
        print(f"RH summary: {result.right_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
