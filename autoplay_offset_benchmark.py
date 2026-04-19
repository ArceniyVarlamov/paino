from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import numpy as np

from hybrid_fusion import HybridScoreFollower

DEFAULT_SCORE_PATH = Path(__file__).resolve().parent / "broken.json"
DEFAULT_STARTS = (0, 25, 50, 100, 150, 200, 250, 300, 400, 500)
MIN_AUTOPLAY_GAP = 0.012
SOUND_START = 0
SOUND_END = 127


@dataclass(frozen=True)
class SweepResult:
    mode: str
    start_index: int
    final_error: int
    average_error: float
    min_error: int
    first_within_10: int | None
    first_within_5: int | None
    first_within_3: int | None
    anchor_event: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark HybridScoreFollower on autoplay offsets for a score JSON.",
    )
    parser.add_argument(
        "score_json",
        nargs="?",
        type=Path,
        default=DEFAULT_SCORE_PATH,
        help=f"Score JSON to benchmark (default: {DEFAULT_SCORE_PATH}).",
    )
    parser.add_argument(
        "--starts",
        type=int,
        nargs="+",
        default=list(DEFAULT_STARTS),
        help="Score indices to use as autoplay start offsets.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=80,
        help="How many autoplay notes to simulate from each start offset.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.5,
        help="HSMM sigma passed through to HybridScoreFollower.",
    )
    return parser


def load_score(score_path: Path) -> list[dict[str, object]]:
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    notes = payload.get("notes", payload)
    if not isinstance(notes, list):
        raise ValueError(f"score_json must contain a note list: {score_path}")
    return notes


def build_autoplay_events(score_notes: list[dict[str, object]]) -> list[dict[str, float | int]]:
    events: list[dict[str, float | int]] = []
    previous_onset: float | None = None
    onset_cursor = 0.0

    for note in score_notes:
        duration = max(0.0, float(note.get("nominal_duration", 0.25)))
        onset = float(note.get("nominal_onset", onset_cursor))
        if previous_onset is None:
            delay = max(0.14, onset)
        else:
            delay = max(MIN_AUTOPLAY_GAP, onset - previous_onset)

        events.append(
            {
                "pitch": int(note["pitch"]),
                "delay": delay,
            }
        )
        previous_onset = onset
        onset_cursor = onset + duration

    return events


def choose_autoplay_pitch(target_pitch: int, mode: str, rng: np.random.Generator) -> int:
    if mode != "mistakes":
        return int(target_pitch)

    roll = float(rng.random())
    played_pitch = int(target_pitch)

    if roll < 0.11:
        played_pitch += int(rng.choice([-2, -1, 1, 2]))
    elif roll < 0.155:
        played_pitch += int(rng.choice([-12, 12]))
    elif roll < 0.19:
        played_pitch += int(rng.choice([-5, 5]))

    return max(SOUND_START, min(SOUND_END, played_pitch))


def first_within_threshold(errors: list[int], threshold: int) -> int | None:
    for index, error in enumerate(errors, start=1):
        if error <= threshold:
            return index
    return None


def run_sweep(
    score_path: Path,
    autoplay_events: list[dict[str, float | int]],
    *,
    start_index: int,
    mode: str,
    max_events: int,
    sigma: float,
) -> SweepResult:
    follower = HybridScoreFollower(score_path, sigma=sigma)
    rng = np.random.default_rng(20260419)
    event_time = 0.0
    errors: list[int] = []
    anchor_event: int | None = None

    for local_index, score_index in enumerate(
        range(start_index, min(len(autoplay_events), start_index + max_events)),
        start=1,
    ):
        event = autoplay_events[score_index]
        event_time += 0.14 if local_index == 1 else float(event["delay"])
        played_pitch = choose_autoplay_pitch(int(event["pitch"]), mode, rng)
        predicted_index = int(follower.process_event(played_pitch, event_time))
        error = abs(predicted_index - score_index)
        errors.append(error)

        if anchor_event is None and follower.last_anchor_target is not None and follower.last_resynced:
            anchor_event = local_index

    if not errors:
        raise ValueError(f"no autoplay events available from start index {start_index}")

    return SweepResult(
        mode=mode,
        start_index=start_index,
        final_error=int(errors[-1]),
        average_error=float(np.mean(errors, dtype=np.float64)),
        min_error=int(min(errors)),
        first_within_10=first_within_threshold(errors, 10),
        first_within_5=first_within_threshold(errors, 5),
        first_within_3=first_within_threshold(errors, 3),
        anchor_event=anchor_event,
    )


def print_results(score_path: Path, note_count: int, results: list[SweepResult]) -> None:
    print(f"score={score_path}")
    print(f"notes={note_count}")
    print()

    for mode in ("clean", "mistakes"):
        mode_results = [result for result in results if result.mode == mode]
        print(f"MODE {mode}")
        header = (
            f"{'start':>6}"
            f"{'final_err':>12}"
            f"{'avg_err':>10}"
            f"{'min_err':>10}"
            f"{'within10':>10}"
            f"{'within5':>9}"
            f"{'within3':>9}"
            f"{'anchor':>9}"
        )
        print(header)
        print("-" * len(header))
        for result in mode_results:
            print(
                f"{result.start_index:>6}"
                f"{result.final_error:>12}"
                f"{result.average_error:>10.1f}"
                f"{result.min_error:>10}"
                f"{str(result.first_within_10):>10}"
                f"{str(result.first_within_5):>9}"
                f"{str(result.first_within_3):>9}"
                f"{str(result.anchor_event):>9}"
            )
        print()


def main() -> None:
    args = build_parser().parse_args()
    score_path = args.score_json
    score_notes = load_score(score_path)
    autoplay_events = build_autoplay_events(score_notes)

    results: list[SweepResult] = []
    for mode in ("clean", "mistakes"):
        for start_index in args.starts:
            results.append(
                run_sweep(
                    score_path,
                    autoplay_events,
                    start_index=start_index,
                    mode=mode,
                    max_events=args.max_events,
                    sigma=args.sigma,
                )
            )

    print_results(score_path, len(score_notes), results)


if __name__ == "__main__":
    main()
