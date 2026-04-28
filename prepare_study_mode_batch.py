from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
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

from midi.midi_splitter import build_pitch_split_files, pitch_display_name
from midi_to_score import convert_to_score

DEFAULT_PIECES = (
    "Élégie,_Opus_3_No_1_–_Sergei_Rachmaninoff",
    "rachmaninov_57525a_prelude_(nc)smythe",
    "etude_8_12_(c)lefeldt",
    "Chopin_Etude_Op_10_No_12_in_C_Minor__Revolutionary_",
)
DEFAULT_SPLIT_POINTS = tuple(range(55, 67))
DEFAULT_CHORD_EPSILONS = (0.02, 0.025, 0.03, 0.04, 0.05)
CALIBRATION_LEVELS = ("baseline", "fast", "medium", "long")


@dataclass(frozen=True)
class SplitCandidate:
    split_point: int
    note_name: str
    left_notes: int
    right_notes: int
    left_ratio: float
    boundary_density: float
    score: float


@dataclass(frozen=True)
class EpsilonCandidate:
    chord_epsilon: float
    states: int
    average_chord_size: float
    p95_chord_size: int
    max_chord_size: int
    score: float


@dataclass(frozen=True)
class PreparedPiece:
    title: str
    source_midi: str
    output_dir: str
    split_point: int
    split_note: str
    chord_epsilon: float
    left_notes: int
    right_notes: int
    left_score_states: int
    right_score_states: int
    profile_path: str
    calibrate_command: list[str]
    tester_command: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare split left/right-hand Study Mode assets for solo piano MIDI files.",
    )
    parser.add_argument(
        "pieces",
        nargs="*",
        help=(
            "MIDI filenames, stems, or substrings. If omitted, prepares the four configured "
            "Chopin/Scriabin/Rachmaninoff pieces."
        ),
    )
    parser.add_argument(
        "--midi-dir",
        type=Path,
        default=PROJECT_ROOT / "midi",
        help="Directory containing the source MIDI files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "midi" / "study_mode",
        help="Directory where per-piece Study Mode folders will be created.",
    )
    parser.add_argument(
        "--split-points",
        type=int,
        nargs="+",
        default=list(DEFAULT_SPLIT_POINTS),
        help="Candidate MIDI split points. Left hand is < split, right hand is >= split.",
    )
    parser.add_argument(
        "--chord-epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_CHORD_EPSILONS),
        help="Candidate onset grouping windows in seconds.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibrate_hybrid_profile.py after writing each left_hand.json.",
    )
    parser.add_argument(
        "--calibration-level",
        choices=CALIBRATION_LEVELS,
        default="fast",
        help="Depth preset passed to calibrate_hybrid_profile.py when --calibrate is used.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=("clean", "mistakes", "both"),
        default=None,
        help="Optional mode override passed to calibrate_hybrid_profile.py when --calibrate is used.",
    )
    parser.add_argument(
        "--calibration-passes",
        type=int,
        default=None,
        help="Optional passes override passed to calibrate_hybrid_profile.py when --calibrate is used.",
    )
    parser.add_argument(
        "--calibration-search-preset",
        choices=("full", "quick", "none"),
        default=None,
        help="Optional search preset override passed to calibrate_hybrid_profile.py when --calibrate is used.",
    )
    parser.add_argument(
        "--calibration-offset-states",
        type=int,
        default=None,
        help="Optional offset states override passed to calibrate_hybrid_profile.py when --calibrate is used.",
    )
    parser.add_argument(
        "--calibration-max-starts",
        type=int,
        default=None,
        help="Optional max starts override passed to calibrate_hybrid_profile.py when --calibrate is used.",
    )
    parser.add_argument(
        "--jump-states",
        type=int,
        default=None,
        help="Optional jump states override passed to calibrate_hybrid_profile.py when --calibrate is used.",
    )
    parser.add_argument(
        "--jump-prime-states",
        type=int,
        default=None,
        help="Optional jump prime states override passed to calibrate_hybrid_profile.py when --calibrate is used.",
    )
    parser.add_argument(
        "--max-jump-scenarios",
        type=int,
        default=None,
        help="Optional max jump scenarios override passed to calibrate_hybrid_profile.py when --calibrate is used.",
    )
    return parser


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_value).strip("_")
    return slug or "piece"


def resolve_piece(piece: str, midi_dir: Path) -> Path:
    raw_path = Path(piece).expanduser()
    if raw_path.exists():
        return raw_path.resolve()

    candidates = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.midi"))
    normalized_piece = unicodedata.normalize("NFC", piece).lower()
    exact_matches: list[Path] = []
    partial_matches: list[Path] = []
    for candidate in candidates:
        stem = unicodedata.normalize("NFC", candidate.stem).lower()
        name = unicodedata.normalize("NFC", candidate.name).lower()
        if normalized_piece in {stem, name}:
            exact_matches.append(candidate)
        elif normalized_piece in stem or normalized_piece in name:
            partial_matches.append(candidate)

    matches = exact_matches or partial_matches
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(f"Could not resolve MIDI piece: {piece}")

    formatted = "\n".join(f"  - {path}" for path in matches)
    raise ValueError(f"Ambiguous MIDI piece '{piece}':\n{formatted}")


def pitch_counts(midi_path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    midi_file = mido.MidiFile(midi_path)
    for message in midi_file:
        if getattr(message, "type", None) != "note_on":
            continue
        if int(getattr(message, "velocity", 0)) <= 0:
            continue
        pitch = int(message.note)
        counts[pitch] = counts.get(pitch, 0) + 1
    return counts


def choose_split_point(
    counts: dict[int, int],
    split_points: list[int],
) -> tuple[SplitCandidate, list[SplitCandidate]]:
    total_notes = sum(counts.values())
    if total_notes <= 0:
        raise ValueError("MIDI file contains no note_on events.")

    candidates: list[SplitCandidate] = []
    for split_point in split_points:
        left_notes = sum(count for pitch, count in counts.items() if pitch < split_point)
        right_notes = total_notes - left_notes
        left_ratio = left_notes / total_notes
        boundary_notes = sum(
            count
            for pitch, count in counts.items()
            if (split_point - 2) <= pitch <= (split_point + 2)
        )
        boundary_density = boundary_notes / total_notes
        ratio_penalty = abs(left_ratio - 0.45)
        imbalance_penalty = 0.0
        if left_ratio < 0.22:
            imbalance_penalty += (0.22 - left_ratio) * 2.5
        if left_ratio > 0.68:
            imbalance_penalty += (left_ratio - 0.68) * 2.5
        middle_c_bias = abs(split_point - 60) * 0.006
        score = (2.2 * boundary_density) + (1.4 * ratio_penalty) + imbalance_penalty + middle_c_bias
        candidates.append(
            SplitCandidate(
                split_point=int(split_point),
                note_name=pitch_display_name(split_point),
                left_notes=int(left_notes),
                right_notes=int(right_notes),
                left_ratio=round(float(left_ratio), 6),
                boundary_density=round(float(boundary_density), 6),
                score=round(float(score), 6),
            )
        )

    best = min(candidates, key=lambda candidate: (candidate.score, abs(candidate.split_point - 60)))
    return best, sorted(candidates, key=lambda candidate: candidate.score)


def write_score_json(midi_path: Path, output_path: Path, chord_epsilon: float) -> dict[str, Any]:
    score = convert_to_score(
        midi_path,
        chord_policy="chord",
        chord_epsilon=float(chord_epsilon),
        default_duration=0.5,
        min_duration=0.05,
    )
    output_path.write_text(
        json.dumps(score, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return score


def chord_sizes(score: dict[str, Any]) -> list[int]:
    notes = score.get("notes")
    if not isinstance(notes, list):
        return []
    sizes: list[int] = []
    for note in notes:
        if not isinstance(note, dict):
            continue
        pitches = note.get("pitches")
        if isinstance(pitches, list):
            sizes.append(len(pitches))
    return sizes


def percentile(values: list[int], percent: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * percent))
    return int(sorted_values[max(0, min(len(sorted_values) - 1, index))])


def choose_chord_epsilon(
    left_midi_path: Path,
    output_dir: Path,
    epsilons: list[float],
) -> tuple[float, list[EpsilonCandidate]]:
    candidates: list[EpsilonCandidate] = []
    scratch_json = output_dir / ".epsilon_probe.json"
    for epsilon in epsilons:
        score = write_score_json(left_midi_path, scratch_json, epsilon)
        sizes = chord_sizes(score)
        states = len(sizes)
        average_chord_size = (sum(sizes) / states) if states else 0.0
        p95 = percentile(sizes, 0.95)
        max_size = max(sizes, default=0)
        score_value = (
            abs(float(epsilon) - 0.03) * 4.0
            + max(0, p95 - 5) * 0.55
            + max(0, max_size - 9) * 0.75
            + max(0.0, average_chord_size - 3.6) * 0.35
        )
        candidates.append(
            EpsilonCandidate(
                chord_epsilon=float(epsilon),
                states=int(states),
                average_chord_size=round(float(average_chord_size), 6),
                p95_chord_size=int(p95),
                max_chord_size=int(max_size),
                score=round(float(score_value), 6),
            )
        )
    if scratch_json.exists():
        scratch_json.unlink()

    best = min(candidates, key=lambda candidate: (candidate.score, abs(candidate.chord_epsilon - 0.03)))
    return float(best.chord_epsilon), sorted(candidates, key=lambda candidate: candidate.score)


def note_on_count(midi_path: Path) -> int:
    midi_file = mido.MidiFile(midi_path)
    return sum(
        1
        for message in midi_file
        if getattr(message, "type", None) == "note_on"
        and int(getattr(message, "velocity", 0)) > 0
    )


def calibrate_command(left_json_path: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "calibrate_hybrid_profile.py"),
        str(left_json_path),
        "--level",
        str(args.calibration_level),
    ]
    optional_flags = (
        ("--mode", args.calibration_mode),
        ("--passes", args.calibration_passes),
        ("--search-preset", args.calibration_search_preset),
        ("--offset-states", args.calibration_offset_states),
        ("--max-starts", args.calibration_max_starts),
        ("--jump-states", args.jump_states),
        ("--jump-prime-states", args.jump_prime_states),
        ("--max-jump-scenarios", args.max_jump_scenarios),
    )
    for flag, value in optional_flags:
        if value is None:
            continue
        command.extend([flag, str(value)])
    return command


def tester_command(left_json_path: Path, right_midi_path: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "interactive_tester.py"),
        str(left_json_path),
        "--orchestra-midi",
        str(right_midi_path),
        "--midi-out",
        "3",
    ]


def prepare_piece(midi_path: Path, args: argparse.Namespace) -> PreparedPiece:
    output_dir = args.output_root.resolve() / slugify(midi_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = pitch_counts(midi_path)
    split_choice, split_candidates = choose_split_point(counts, list(args.split_points))
    midi_file = mido.MidiFile(midi_path)
    left_file, right_file = build_pitch_split_files(
        midi_file,
        split_point=split_choice.split_point,
    )

    left_midi_path = output_dir / "left_hand.mid"
    right_midi_path = output_dir / "right_hand.mid"
    left_json_path = output_dir / "left_hand.json"
    right_json_path = output_dir / "right_hand.json"
    report_path = output_dir / "prep_report.json"
    left_file.save(left_midi_path)
    right_file.save(right_midi_path)

    chord_epsilon, epsilon_candidates = choose_chord_epsilon(
        left_midi_path,
        output_dir,
        list(args.chord_epsilons),
    )
    left_score = write_score_json(left_midi_path, left_json_path, chord_epsilon)
    right_score = write_score_json(right_midi_path, right_json_path, chord_epsilon)

    command = calibrate_command(left_json_path, args)
    if args.calibrate:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)

    profile_path = left_json_path.with_suffix(".hybrid_profile.json")
    prepared = PreparedPiece(
        title=midi_path.stem,
        source_midi=str(midi_path),
        output_dir=str(output_dir),
        split_point=int(split_choice.split_point),
        split_note=split_choice.note_name,
        chord_epsilon=float(chord_epsilon),
        left_notes=note_on_count(left_midi_path),
        right_notes=note_on_count(right_midi_path),
        left_score_states=len(left_score.get("notes", [])),
        right_score_states=len(right_score.get("notes", [])),
        profile_path=str(profile_path),
        calibrate_command=command,
        tester_command=tester_command(left_json_path, right_midi_path),
    )

    report = {
        "prepared": asdict(prepared),
        "split_candidates": [asdict(candidate) for candidate in split_candidates],
        "epsilon_candidates": [asdict(candidate) for candidate in epsilon_candidates],
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return prepared


def main() -> int:
    args = build_parser().parse_args()
    pieces = args.pieces or list(DEFAULT_PIECES)
    args.midi_dir = args.midi_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    prepared_pieces: list[PreparedPiece] = []
    for piece in pieces:
        midi_path = resolve_piece(piece, args.midi_dir)
        print(f"Preparing {midi_path.name}")
        prepared = prepare_piece(midi_path, args)
        prepared_pieces.append(prepared)
        print(
            f"  split={prepared.split_point} ({prepared.split_note}), "
            f"epsilon={prepared.chord_epsilon:.3f}, "
            f"left_states={prepared.left_score_states}, right_notes={prepared.right_notes}"
        )

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"pieces": [asdict(piece) for piece in prepared_pieces]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
