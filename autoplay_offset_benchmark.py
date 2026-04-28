from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import numpy as np

from hybrid_fusion import HybridScoreFollower
from output_dispatcher import TempoTracker

DEFAULT_SCORE_PATH = Path(__file__).resolve().parent / "midi" / "rach_solo.json"
DEFAULT_STARTS = (0, 50, 100, 200, 400, 800, 1200, 2000, 3000)
MIN_AUTOPLAY_GAP = 0.012
FIRST_AUTOPLAY_DELAY = 0.14
SOUND_START = 0
SOUND_END = 127


@dataclass(frozen=True)
class SegmentResult:
    scenario: str
    mode: str
    start_index: int
    states: int
    final_error: int
    average_error: float
    max_error: int
    first_within_10: int | None
    first_within_5: int | None
    first_within_3: int | None
    tempo_mean: float
    tempo_std: float
    tempo_min: float
    tempo_max: float
    tempo_abs_dev: float
    final_prediction: int
    final_target: int


@dataclass(frozen=True)
class JumpScenario:
    label: str
    first_start: int
    first_states: int
    second_start: int
    second_states: int


DEFAULT_JUMP_SCENARIOS = (
    JumpScenario("jump_0_to_400", first_start=0, first_states=40, second_start=400, second_states=200),
    JumpScenario("jump_300_to_1", first_start=0, first_states=320, second_start=1, second_states=200),
    JumpScenario("jump_700_to_100", first_start=0, first_states=720, second_start=100, second_states=200),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark HybridScoreFollower on full-piece chord-aware autoplay offsets and jumps.",
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
        help="Score-state offsets to test from a fresh follower.",
    )
    parser.add_argument(
        "--offset-states",
        type=int,
        default=0,
        help=(
            "Limit each offset-start segment to this many score states. "
            "Use 0 to run from each start to the end."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.5,
        help="HSMM sigma passed through to HybridScoreFollower.",
    )
    parser.add_argument(
        "--mode",
        choices=("clean", "mistakes", "both"),
        default="clean",
        help="Which autoplay mode(s) to benchmark.",
    )
    parser.add_argument(
        "--jump-states",
        type=int,
        default=200,
        help="How many score states to evaluate after each jump segment.",
    )
    parser.add_argument(
        "--skip-jumps",
        action="store_true",
        help="Disable jump scenario benchmarking.",
    )
    parser.add_argument(
        "--jump-grid-step",
        type=int,
        default=0,
        help=(
            "Generate additional ordered jump pairs from a regular score-state grid. "
            "Use 0 to disable (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--jump-random-samples",
        type=int,
        default=0,
        help="Generate this many additional random ordered jump pairs (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260420,
        help="Random seed for generated jump scenarios (default: %(default)s).",
    )
    parser.add_argument(
        "--jump-prime-states",
        type=int,
        default=0,
        help=(
            "Prime each jump locally around its first_start instead of replaying from score start. "
            "Use 0 to keep the full warmup behavior (default: %(default)s)."
        ),
    )
    return parser


def load_score(score_path: Path) -> list[dict[str, object]]:
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    notes = payload.get("notes", payload)
    if not isinstance(notes, list):
        raise ValueError(f"score_json must contain a note list: {score_path}")
    return notes


def note_pitches(note: dict[str, object]) -> list[int]:
    raw_pitches = note.get("pitches")
    if raw_pitches is None:
        raw_pitch = note.get("pitch")
        if raw_pitch is None:
            raise ValueError("score note is missing 'pitch'/'pitches'")
        return [int(raw_pitch)]
    if not isinstance(raw_pitches, list) or not raw_pitches:
        raise ValueError("score note 'pitches' must be a non-empty list")
    return sorted({int(pitch) for pitch in raw_pitches})


def build_autoplay_events(score_notes: list[dict[str, object]]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    previous_onset: float | None = None
    onset_cursor = 0.0

    for score_position, note in enumerate(score_notes):
        duration = max(0.0, float(note.get("nominal_duration", 0.25)))
        onset = float(note.get("nominal_onset", onset_cursor))
        chord_pitches = note_pitches(note)
        if previous_onset is None:
            first_delay = max(FIRST_AUTOPLAY_DELAY, onset)
        else:
            first_delay = max(MIN_AUTOPLAY_GAP, onset - previous_onset)

        events.append(
            {
                "pitches": [int(pitch) for pitch in chord_pitches],
                "delay": first_delay,
                "score_position": score_position,
                "nominal_duration": duration,
                "nominal_onset": onset,
            }
        )

        previous_onset = onset
        onset_cursor = onset + duration

    return events


def autoplay_event_start_index(
    autoplay_events: list[dict[str, object]],
    score_start_index: int,
) -> int:
    target_score_position = max(0, int(score_start_index))
    for event_index, event in enumerate(autoplay_events):
        if int(event["score_position"]) >= target_score_position:
            return event_index
    return len(autoplay_events)


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


def first_within_threshold(errors: Iterable[int], threshold: int) -> int | None:
    for index, error in enumerate(errors, start=1):
        if error <= threshold:
            return index
    return None


def summarize_segment(
    *,
    scenario: str,
    mode: str,
    start_index: int,
    samples: list[tuple[int, int, float]],
) -> SegmentResult:
    if not samples:
        raise ValueError(f"segment '{scenario}' from start {start_index} produced no state samples")

    errors = [abs(predicted - target) for target, predicted, _tempo in samples]
    tempos = [float(tempo) for _target, _predicted, tempo in samples]
    tempo_abs_dev = [abs(tempo - 1.0) for tempo in tempos]

    return SegmentResult(
        scenario=scenario,
        mode=mode,
        start_index=int(start_index),
        states=len(samples),
        final_error=int(errors[-1]),
        average_error=float(statistics.mean(errors)),
        max_error=int(max(errors)),
        first_within_10=first_within_threshold(errors, 10),
        first_within_5=first_within_threshold(errors, 5),
        first_within_3=first_within_threshold(errors, 3),
        tempo_mean=float(statistics.mean(tempos)),
        tempo_std=float(statistics.pstdev(tempos)) if len(tempos) > 1 else 0.0,
        tempo_min=float(min(tempos)),
        tempo_max=float(max(tempos)),
        tempo_abs_dev=float(statistics.mean(tempo_abs_dev)),
        final_prediction=int(samples[-1][1]),
        final_target=int(samples[-1][0]),
    )


def simulate_segment(
    follower: HybridScoreFollower,
    tempo_tracker: TempoTracker,
    autoplay_events: list[dict[str, object]],
    *,
    start_index: int,
    mode: str,
    rng: np.random.Generator,
    event_time: float,
    max_states: int | None = None,
    scenario: str,
) -> tuple[SegmentResult, float]:
    event_start = autoplay_event_start_index(autoplay_events, start_index)
    if event_start >= len(autoplay_events):
        raise ValueError(f"start_index {start_index} is beyond the end of the autoplay events")

    if max_states is None:
        last_score_position = int(autoplay_events[-1]["score_position"])
    else:
        last_score_position = int(start_index + max_states - 1)

    first_event = True
    current_tempo = float(tempo_tracker.tempo_ratio)
    samples: list[tuple[int, int, float]] = []

    for event in autoplay_events[event_start:]:
        score_position = int(event["score_position"])
        if score_position > last_score_position:
            break

        if first_event:
            event_time += FIRST_AUTOPLAY_DELAY
            first_event = False
        else:
            event_time += float(event["delay"])

        target_pitches = [int(pitch) for pitch in event.get("pitches", [])]
        played_pitches = [
            choose_autoplay_pitch(target_pitch, mode, rng)
            for target_pitch in target_pitches
        ]
        predicted_index = int(follower.process_event(played_pitches, event_time))
        current_tempo = float(tempo_tracker.update(predicted_index, event_time))
        samples.append((score_position, predicted_index, current_tempo))

    return summarize_segment(
        scenario=scenario,
        mode=mode,
        start_index=start_index,
        samples=samples,
    ), event_time


def run_offset_sweeps(
    score_source: str | Path | dict[str, object] | list[dict[str, object]],
    autoplay_events: list[dict[str, object]],
    *,
    starts: list[int],
    mode: str,
    sigma: float,
    max_states: int | None = None,
    follower_kwargs: dict[str, object] | None = None,
) -> list[SegmentResult]:
    results: list[SegmentResult] = []
    effective_follower_kwargs = {"load_tuning_profile": False, **(follower_kwargs or {})}
    follower = HybridScoreFollower(score_source, sigma=sigma, **effective_follower_kwargs)
    tempo_tracker = TempoTracker(score_source)
    for start_index in starts:
        follower.reset_to_start()
        tempo_tracker.reset()
        rng = np.random.default_rng(20260419)
        result, _ = simulate_segment(
            follower,
            tempo_tracker,
            autoplay_events,
            start_index=start_index,
            mode=mode,
            rng=rng,
            event_time=0.0,
            max_states=max_states,
            scenario=f"start_{start_index}",
        )
        results.append(result)
    return results


def run_jump_scenarios(
    score_source: str | Path | dict[str, object] | list[dict[str, object]],
    autoplay_events: list[dict[str, object]],
    *,
    mode: str,
    sigma: float,
    jump_states: int,
    scenarios: list[JumpScenario] | tuple[JumpScenario, ...] = DEFAULT_JUMP_SCENARIOS,
    jump_prime_states: int = 0,
    follower_kwargs: dict[str, object] | None = None,
) -> list[SegmentResult]:
    results: list[SegmentResult] = []
    effective_follower_kwargs = {"load_tuning_profile": False, **(follower_kwargs or {})}
    follower = HybridScoreFollower(score_source, sigma=sigma, **effective_follower_kwargs)
    tempo_tracker = TempoTracker(score_source)
    for scenario in scenarios:
        follower.reset_to_start()
        tempo_tracker.reset()
        rng = np.random.default_rng(20260419)

        warmup_start = int(scenario.first_start)
        warmup_states = int(scenario.first_states)
        event_time = 0.0
        if jump_prime_states > 0:
            warmup_start = max(0, int(scenario.first_start) - int(jump_prime_states) + 1)
            warmup_states = int(scenario.first_start) - warmup_start + 1
            follower.seek(warmup_start, event_time)
            tempo_tracker.reset()

        first_result, event_time = simulate_segment(
            follower,
            tempo_tracker,
            autoplay_events,
            start_index=warmup_start,
            mode=mode,
            rng=rng,
            event_time=event_time,
            max_states=warmup_states,
            scenario=f"{scenario.label}:warmup",
        )
        results.append(first_result)

        second_result, _ = simulate_segment(
            follower,
            tempo_tracker,
            autoplay_events,
            start_index=scenario.second_start,
            mode=mode,
            rng=rng,
            event_time=event_time,
            max_states=jump_states if scenario.second_states is None else scenario.second_states,
            scenario=f"{scenario.label}:after_jump",
        )
        results.append(second_result)
    return results


def build_generated_jump_scenarios(
    score_length: int,
    *,
    grid_step: int,
    random_samples: int,
    seed: int,
    jump_states: int,
) -> list[JumpScenario]:
    generated: list[JumpScenario] = []
    seen_pairs: set[tuple[int, int]] = set()

    if grid_step > 0:
        grid_positions = sorted(
            {
                0,
                score_length - 1,
                *range(0, score_length, grid_step),
            }
        )
        for first_start in grid_positions:
            for second_start in grid_positions:
                if first_start == second_start:
                    continue
                pair = (int(first_start), int(second_start))
                seen_pairs.add(pair)
                generated.append(
                    JumpScenario(
                        label=f"grid_{first_start}_to_{second_start}",
                        first_start=int(first_start),
                        first_states=max(40, int(first_start) + 1),
                        second_start=int(second_start),
                        second_states=int(jump_states),
                    )
                )

    if random_samples > 0:
        rng = np.random.default_rng(seed)
        attempts = 0
        while len([scenario for scenario in generated if scenario.label.startswith("random_")]) < random_samples:
            first_start = int(rng.integers(0, score_length))
            second_start = int(rng.integers(0, score_length))
            if first_start == second_start:
                continue
            pair = (first_start, second_start)
            attempts += 1
            if pair in seen_pairs:
                if attempts > random_samples * 20:
                    break
                continue
            seen_pairs.add(pair)
            generated.append(
                JumpScenario(
                    label=f"random_{first_start}_to_{second_start}",
                    first_start=first_start,
                    first_states=max(40, first_start + 1),
                    second_start=second_start,
                    second_states=int(jump_states),
                )
            )

    return generated


def print_results(title: str, results: list[SegmentResult]) -> None:
    print(title)
    header = (
        f"{'scenario':<28}"
        f"{'start':>7}"
        f"{'states':>8}"
        f"{'final':>8}"
        f"{'avg':>8}"
        f"{'max':>8}"
        f"{'w10':>7}"
        f"{'w5':>7}"
        f"{'w3':>7}"
        f"{'t_mean':>9}"
        f"{'t_std':>8}"
        f"{'t_min':>8}"
        f"{'t_max':>8}"
        f"{'t_dev':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.scenario:<28}"
            f"{result.start_index:>7}"
            f"{result.states:>8}"
            f"{result.final_error:>8}"
            f"{result.average_error:>8.2f}"
            f"{result.max_error:>8}"
            f"{str(result.first_within_10):>7}"
            f"{str(result.first_within_5):>7}"
            f"{str(result.first_within_3):>7}"
            f"{result.tempo_mean:>9.3f}"
            f"{result.tempo_std:>8.3f}"
            f"{result.tempo_min:>8.3f}"
            f"{result.tempo_max:>8.3f}"
            f"{result.tempo_abs_dev:>8.3f}"
        )
    print()


def main() -> None:
    args = build_parser().parse_args()
    score_path = args.score_json
    score_notes = load_score(score_path)
    autoplay_events = build_autoplay_events(score_notes)
    modes = ("clean", "mistakes") if args.mode == "both" else (args.mode,)

    print(f"score={score_path}")
    print(f"score_states={len(score_notes)}")
    print(f"autoplay_events={len(autoplay_events)}")
    print()

    for mode in modes:
        offset_results = run_offset_sweeps(
            score_notes,
            autoplay_events,
            starts=list(args.starts),
            mode=mode,
            sigma=args.sigma,
            max_states=args.offset_states if args.offset_states > 0 else None,
        )
        print_results(f"MODE {mode} | full-piece starts", offset_results)

        if args.skip_jumps:
            continue

        jump_results = run_jump_scenarios(
            score_notes,
            autoplay_events,
            mode=mode,
            sigma=args.sigma,
            jump_states=args.jump_states,
            jump_prime_states=args.jump_prime_states,
        )
        print_results(f"MODE {mode} | jump scenarios", jump_results)

        generated_jump_scenarios = build_generated_jump_scenarios(
            len(score_notes),
            grid_step=args.jump_grid_step,
            random_samples=args.jump_random_samples,
            seed=args.seed,
            jump_states=args.jump_states,
        )
        if generated_jump_scenarios:
            generated_results = run_jump_scenarios(
                score_notes,
                autoplay_events,
                mode=mode,
                sigma=args.sigma,
                jump_states=args.jump_states,
                scenarios=generated_jump_scenarios,
                jump_prime_states=args.jump_prime_states,
            )
            print_results(f"MODE {mode} | generated jump scenarios", generated_results)


if __name__ == "__main__":
    main()
