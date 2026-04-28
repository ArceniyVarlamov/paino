from __future__ import annotations

import argparse
import os
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoplay_offset_benchmark import (
    JumpScenario,
    SegmentResult,
    build_autoplay_events,
    load_score,
    run_jump_scenarios,
    run_offset_sweeps,
)
from hybrid_fusion import HYBRID_PROFILE_FORMAT_VERSION, hybrid_profile_path

DEFAULT_FOLLOWER_CONFIG: dict[str, object] = {
    "anchor_window_lengths": (20, 16, 12, 8, 6, 4),
    "anchor_confirmation_events": 2,
    "anchor_min_supporting_windows": 2,
    "anchor_margin_threshold": 0.05,
    "anchor_local_preference_margin": 0.05,
    "anchor_local_improvement_threshold": 0.35,
    "anchor_min_jump": 8,
    "max_forward_step": 3,
    "output_confirmation_events": 2,
    "anchor_time_weight": 1.25,
    "output_high_confidence": 0.4,
}

SEARCH_SPACE: dict[str, list[object]] = {
    "anchor_window_lengths": [
        (20, 16, 12, 8, 6, 4),
        (20, 16, 12, 8, 6),
        (20, 16, 12, 8),
        (16, 12, 8, 6),
        (16, 12, 8),
        (12, 8, 6, 4),
    ],
    "anchor_confirmation_events": [2, 3, 4],
    "anchor_min_supporting_windows": [2, 3],
    "anchor_margin_threshold": [0.05, 0.08, 0.12],
    "anchor_local_preference_margin": [0.05, 0.08, 0.12],
    "anchor_local_improvement_threshold": [0.35, 0.45, 0.60],
    "anchor_min_jump": [8, 12, 16],
    "max_forward_step": [3, 4],
    "output_confirmation_events": [2, 3],
    "anchor_time_weight": [1.0, 1.25, 1.5],
    "output_high_confidence": [0.35, 0.40, 0.45],
}

QUICK_SEARCH_SPACE: dict[str, list[object]] = {
    "anchor_window_lengths": [
        (20, 16, 12, 8),
        (16, 12, 8),
    ],
    "anchor_confirmation_events": [2, 3],
    "anchor_min_supporting_windows": [2, 3],
    "anchor_min_jump": [8, 12],
    "max_forward_step": [3, 4],
    "output_confirmation_events": [2, 3],
}


@dataclass(frozen=True)
class CalibrationLevel:
    mode: str
    passes: int
    search_preset: str
    offset_states: int
    max_starts: int
    jump_states: int
    jump_prime_states: int
    max_jump_scenarios: int


CALIBRATION_LEVELS: dict[str, CalibrationLevel] = {
    "baseline": CalibrationLevel(
        mode="clean",
        passes=1,
        search_preset="none",
        offset_states=260,
        max_starts=7,
        jump_states=80,
        jump_prime_states=24,
        max_jump_scenarios=6,
    ),
    "fast": CalibrationLevel(
        mode="clean",
        passes=1,
        search_preset="quick",
        offset_states=260,
        max_starts=7,
        jump_states=80,
        jump_prime_states=24,
        max_jump_scenarios=6,
    ),
    "medium": CalibrationLevel(
        mode="both",
        passes=1,
        search_preset="quick",
        offset_states=360,
        max_starts=9,
        jump_states=110,
        jump_prime_states=32,
        max_jump_scenarios=12,
    ),
    "long": CalibrationLevel(
        mode="both",
        passes=2,
        search_preset="full",
        offset_states=0,
        max_starts=0,
        jump_states=160,
        jump_prime_states=48,
        max_jump_scenarios=18,
    ),
}


@dataclass(frozen=True)
class CalibrationMetrics:
    objective: float
    offset_segments: int
    jump_segments: int
    offset_average_error: float
    offset_max_error: int
    offset_final_error_sum: int
    jump_average_error: float
    jump_max_error: int
    jump_final_error_sum: int
    jump_within5_mean: float
    jump_within5_max: int
    jump_failures: int
    steady_tempo_abs_dev: float


@dataclass(frozen=True)
class EvaluationContext:
    score_source: list[dict[str, object]]
    autoplay_events: list[dict[str, object]]
    starts: tuple[int, ...]
    modes: tuple[str, ...]
    sigma: float
    offset_states: int | None
    jump_states: int
    jump_prime_states: int
    jump_scenarios: tuple[JumpScenario, ...]


_WORKER_CONTEXT: EvaluationContext | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate HybridScoreFollower recovery thresholds for one score and save a "
            "sidecar *.hybrid_profile.json that interactive_tester.py will auto-load."
        ),
    )
    parser.add_argument("score_json", type=Path, help="Score JSON to calibrate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the generated profile JSON.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.5,
        help="HSMM sigma used during calibration runs.",
    )
    parser.add_argument(
        "--level",
        choices=tuple(CALIBRATION_LEVELS.keys()),
        default="fast",
        help=(
            "Calibration depth preset. baseline only evaluates defaults, fast is a short "
            "clean search, medium adds mistake/recovery checks, long uses the full search space."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("clean", "mistakes", "both"),
        default=None,
        help="Override the level playback mode(s) to optimize for.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=None,
        help="Override the level coordinate-descent passes over the search space.",
    )
    parser.add_argument(
        "--search-preset",
        choices=("full", "quick", "none"),
        default=None,
        help="Override the level search preset.",
    )
    parser.add_argument(
        "--offset-states",
        type=int,
        default=None,
        help="Override the level offset-start state limit. Use 0 to run from each start to the end.",
    )
    parser.add_argument(
        "--max-starts",
        type=int,
        default=None,
        help="Override the level automatic offset start limit. Use 0 for all.",
    )
    parser.add_argument(
        "--jump-states",
        type=int,
        default=None,
        help="Override how many score states to evaluate after each jump.",
    )
    parser.add_argument(
        "--jump-prime-states",
        type=int,
        default=None,
        help="Override how many local states to warm up before each jump scenario.",
    )
    parser.add_argument(
        "--max-jump-scenarios",
        type=int,
        default=None,
        help="Override the level jump scenario limit. Use 0 for all.",
    )
    parser.add_argument(
        "--starts",
        type=int,
        nargs="*",
        default=None,
        help="Optional fixed offset starts. If omitted, an automatic spread is used.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and print the best configuration without writing a profile file.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help=(
            "How many worker processes to use for candidate evaluation inside each "
            "coordinate-descent step. Use 0 for automatic sizing and 1 to disable parallelism."
        ),
    )
    return parser


def apply_level_defaults(args: argparse.Namespace) -> argparse.Namespace:
    level = CALIBRATION_LEVELS[str(args.level)]
    for attr in (
        "mode",
        "passes",
        "search_preset",
        "offset_states",
        "max_starts",
        "jump_states",
        "jump_prime_states",
        "max_jump_scenarios",
    ):
        if getattr(args, attr) is None:
            setattr(args, attr, getattr(level, attr))
    return args


def search_space_for_preset(preset: str) -> dict[str, list[object]]:
    if preset == "quick":
        return QUICK_SEARCH_SPACE
    if preset == "none":
        return {}
    return SEARCH_SPACE


def automatic_starts(score_length: int) -> list[int]:
    if score_length <= 1:
        return [0]
    candidates = {
        0,
        25,
        50,
        100,
        200,
        score_length // 8,
        score_length // 4,
        score_length // 2,
        (score_length * 3) // 4,
        max(0, score_length - 200),
        score_length - 1,
    }
    return sorted({value for value in candidates if 0 <= value < score_length})


def automatic_jump_positions(score_length: int) -> list[int]:
    if score_length <= 1:
        return [0]
    candidates = {
        0,
        min(100, score_length - 1),
        score_length // 4,
        score_length // 2,
        (score_length * 3) // 4,
        score_length - 1,
    }
    return sorted({value for value in candidates if 0 <= value < score_length})


def calibration_jump_scenarios(score_length: int, jump_states: int) -> list[JumpScenario]:
    positions = automatic_jump_positions(score_length)
    scenarios: list[JumpScenario] = []
    for first_start in positions:
        for second_start in positions:
            if first_start == second_start:
                continue
            scenarios.append(
                JumpScenario(
                    label=f"cal_{first_start}_to_{second_start}",
                    first_start=int(first_start),
                    first_states=max(12, min(48, int(first_start) + 1)),
                    second_start=int(second_start),
                    second_states=int(jump_states),
                )
            )
    return scenarios


def limit_jump_scenarios(
    scenarios: list[JumpScenario],
    max_scenarios: int,
) -> list[JumpScenario]:
    if max_scenarios <= 0 or len(scenarios) <= max_scenarios:
        return scenarios
    if max_scenarios == 1:
        return [scenarios[len(scenarios) // 2]]

    last_index = len(scenarios) - 1
    selected_indices = {
        round((last_index * index) / (max_scenarios - 1))
        for index in range(max_scenarios)
    }
    return [scenarios[index] for index in sorted(selected_indices)]


def limit_starts(starts: list[int], max_starts: int) -> list[int]:
    if max_starts <= 0 or len(starts) <= max_starts:
        return starts
    if max_starts == 1:
        return [starts[0]]

    last_index = len(starts) - 1
    selected_indices = {
        round((last_index * index) / (max_starts - 1))
        for index in range(max_starts)
    }
    return [starts[index] for index in sorted(selected_indices)]


def normalized_config_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def config_cache_key(config: dict[str, object]) -> tuple[tuple[str, object], ...]:
    frozen_items: list[tuple[str, object]] = []
    for key, value in sorted(config.items()):
        if isinstance(value, tuple):
            frozen_items.append((key, tuple(value)))
        elif isinstance(value, list):
            frozen_items.append((key, tuple(value)))
        else:
            frozen_items.append((key, value))
    return tuple(frozen_items)


def init_evaluation_worker(context: EvaluationContext) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def resolve_parallel_workers(requested_workers: int, candidate_count: int) -> int:
    if candidate_count <= 1:
        return 1
    if requested_workers == 1:
        return 1
    if requested_workers > 1:
        return min(int(requested_workers), int(candidate_count))

    available_cpus = max(1, int(os.cpu_count() or 1))
    auto_workers = max(1, available_cpus - 1)
    return min(auto_workers, int(candidate_count))


def summarize_results(results: list[SegmentResult]) -> CalibrationMetrics:
    offset_results = [
        result
        for result in results
        if result.scenario.startswith("start_")
    ]
    jump_results = [
        result
        for result in results
        if result.scenario.endswith(":after_jump")
    ]

    def safe_mean(values: list[float]) -> float:
        return float(statistics.mean(values)) if values else 0.0

    def safe_max(values: list[int]) -> int:
        return int(max(values)) if values else 0

    objective = 0.0
    for result in offset_results:
        mode_weight = 1.35 if result.mode == "mistakes" else 1.0
        first_within_5 = float(result.first_within_5 if result.first_within_5 is not None else result.states + 50)
        objective += mode_weight * (
            12.0 * first_within_5
            + 18.0 * float(result.final_error)
            + 1.25 * float(result.average_error)
            + 0.04 * float(result.max_error)
            + 80.0 * float(result.tempo_abs_dev)
        )

    for result in jump_results:
        mode_weight = 1.60 if result.mode == "mistakes" else 1.20
        first_within_5 = float(result.first_within_5 if result.first_within_5 is not None else result.states + 50)
        first_within_3 = float(result.first_within_3 if result.first_within_3 is not None else result.states + 50)
        objective += mode_weight * (
            50.0 * first_within_5
            + 25.0 * first_within_3
            + 60.0 * float(result.final_error)
            + 3.0 * float(result.average_error)
            + 0.08 * float(result.max_error)
        )

    return CalibrationMetrics(
        objective=float(objective),
        offset_segments=len(offset_results),
        jump_segments=len(jump_results),
        offset_average_error=safe_mean([float(result.average_error) for result in offset_results]),
        offset_max_error=safe_max([int(result.max_error) for result in offset_results]),
        offset_final_error_sum=int(sum(int(result.final_error) for result in offset_results)),
        jump_average_error=safe_mean([float(result.average_error) for result in jump_results]),
        jump_max_error=safe_max([int(result.max_error) for result in jump_results]),
        jump_final_error_sum=int(sum(int(result.final_error) for result in jump_results)),
        jump_within5_mean=safe_mean(
            [
                float(result.first_within_5 if result.first_within_5 is not None else result.states + 50)
                for result in jump_results
            ]
        ),
        jump_within5_max=safe_max(
            [
                int(result.first_within_5 if result.first_within_5 is not None else result.states + 50)
                for result in jump_results
            ]
        ),
        jump_failures=sum(1 for result in jump_results if result.first_within_5 is None),
        steady_tempo_abs_dev=safe_mean([float(result.tempo_abs_dev) for result in offset_results]),
    )


def evaluate_config_uncached(
    *,
    context: EvaluationContext,
    config: dict[str, object],
) -> CalibrationMetrics:
    results: list[SegmentResult] = []
    follower_kwargs = {"load_tuning_profile": False, **config}
    for mode in context.modes:
        results.extend(
            run_offset_sweeps(
                context.score_source,
                context.autoplay_events,
                starts=list(context.starts),
                mode=mode,
                sigma=context.sigma,
                max_states=context.offset_states,
                follower_kwargs=follower_kwargs,
            )
        )
        results.extend(
            run_jump_scenarios(
                context.score_source,
                context.autoplay_events,
                mode=mode,
                sigma=context.sigma,
                jump_states=context.jump_states,
                scenarios=list(context.jump_scenarios),
                jump_prime_states=context.jump_prime_states,
                follower_kwargs=follower_kwargs,
            )
        )

    return summarize_results(results)


def evaluate_config_in_worker(config: dict[str, object]) -> tuple[tuple[tuple[str, object], ...], CalibrationMetrics]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Evaluation worker was started without a context.")
    metrics = evaluate_config_uncached(context=_WORKER_CONTEXT, config=config)
    return config_cache_key(config), metrics


def evaluate_config(
    *,
    context: EvaluationContext,
    config: dict[str, object],
    cache: dict[tuple[tuple[str, object], ...], CalibrationMetrics],
) -> CalibrationMetrics:
    key = config_cache_key(config)
    cached = cache.get(key)
    if cached is not None:
        return cached

    metrics = evaluate_config_uncached(context=context, config=config)
    cache[key] = metrics
    return metrics


def normalize_config_for_json(config: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in sorted(config.items()):
        if isinstance(value, tuple):
            normalized[key] = list(value)
        else:
            normalized[key] = value
    return normalized


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = apply_level_defaults(build_parser().parse_args())
    score_path = args.score_json.resolve()
    score_notes = load_score(score_path)
    autoplay_events = build_autoplay_events(score_notes)
    starts = automatic_starts(len(score_notes)) if args.starts is None else sorted(
        {max(0, min(int(value), len(score_notes) - 1)) for value in args.starts}
    )
    starts = limit_starts(starts, int(args.max_starts)) if args.starts is None else starts
    modes = ("clean", "mistakes") if args.mode == "both" else (args.mode,)
    jump_scenarios = limit_jump_scenarios(
        calibration_jump_scenarios(len(score_notes), args.jump_states),
        int(args.max_jump_scenarios),
    )
    search_space = search_space_for_preset(args.search_preset)
    output_path = args.output.resolve() if args.output is not None else hybrid_profile_path(score_path)
    evaluation_context = EvaluationContext(
        score_source=score_notes,
        autoplay_events=autoplay_events,
        starts=tuple(starts),
        modes=tuple(modes),
        sigma=float(args.sigma),
        offset_states=args.offset_states if args.offset_states > 0 else None,
        jump_states=int(args.jump_states),
        jump_prime_states=int(args.jump_prime_states),
        jump_scenarios=tuple(jump_scenarios),
    )
    max_parallel_candidates = max((len(candidates) - 1 for candidates in search_space.values()), default=1)
    parallel_workers = resolve_parallel_workers(int(args.parallel_workers), max_parallel_candidates)
    executor: ProcessPoolExecutor | None = None

    print(f"score={score_path}")
    print(f"states={len(score_notes)}")
    print(f"level={args.level}")
    print(f"starts={starts}")
    print(f"offset_states={args.offset_states if args.offset_states > 0 else 'full'}")
    print(f"jump_scenarios={len(jump_scenarios)}")
    print(f"modes={','.join(modes)}")
    print(f"search_preset={args.search_preset}")
    print(f"parallel_workers={parallel_workers}")
    print()

    cache: dict[tuple[tuple[str, object], ...], CalibrationMetrics] = {}
    best_config: dict[str, object] = {}
    best_metrics = evaluate_config(
        context=evaluation_context,
        config=best_config,
        cache=cache,
    )
    baseline_metrics = best_metrics
    print(f"baseline objective={baseline_metrics.objective:.2f} metrics={baseline_metrics}")

    try:
        if parallel_workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=parallel_workers,
                initializer=init_evaluation_worker,
                initargs=(evaluation_context,),
            )

        if search_space:
            for pass_index in range(max(1, int(args.passes))):
                improved = False
                print(f"\npass {pass_index + 1}/{max(1, int(args.passes))}")
                for key, candidates in search_space.items():
                    local_best_config = dict(best_config)
                    local_best_metrics = best_metrics
                    current_value = local_best_config.get(key)
                    if current_value is None and key in DEFAULT_FOLLOWER_CONFIG:
                        current_value = DEFAULT_FOLLOWER_CONFIG[key]
                    normalized_current_value = normalized_config_value(current_value)
                    candidate_proposals: list[tuple[object, dict[str, object]]] = []

                    for candidate in candidates:
                        if normalized_current_value == normalized_config_value(candidate):
                            continue
                        proposal = dict(best_config)
                        proposal[key] = candidate
                        candidate_proposals.append((candidate, proposal))

                    if executor is None or len(candidate_proposals) <= 1:
                        ordered_metrics: list[tuple[object, dict[str, object], CalibrationMetrics]] = [
                            (
                                candidate,
                                proposal,
                                evaluate_config(
                                    context=evaluation_context,
                                    config=proposal,
                                    cache=cache,
                                ),
                            )
                            for candidate, proposal in candidate_proposals
                        ]
                    else:
                        ordered_metrics = []
                        future_map = {
                            executor.submit(evaluate_config_in_worker, proposal): (candidate, proposal)
                            for candidate, proposal in candidate_proposals
                        }
                        future_results: dict[tuple[tuple[str, object], ...], CalibrationMetrics] = {}
                        for future in future_map:
                            cache_key, metrics = future.result()
                            cache[cache_key] = metrics
                            future_results[cache_key] = metrics
                        for candidate, proposal in candidate_proposals:
                            ordered_metrics.append(
                                (
                                    candidate,
                                    proposal,
                                    future_results[config_cache_key(proposal)],
                                )
                            )

                    for candidate, proposal, metrics in ordered_metrics:
                        print(
                            f"  {key}={candidate} -> objective={metrics.objective:.2f} "
                            f"(jump_within5_mean={metrics.jump_within5_mean:.2f}, "
                            f"jump_failures={metrics.jump_failures}, "
                            f"offset_avg={metrics.offset_average_error:.2f})"
                        )
                        if metrics.objective + 1e-9 < local_best_metrics.objective:
                            local_best_config = proposal
                            local_best_metrics = metrics

                    if local_best_metrics.objective + 1e-9 < best_metrics.objective:
                        best_config = local_best_config
                        best_metrics = local_best_metrics
                        improved = True
                        print(
                            f"  accepted {key}={best_config[key]!r}; "
                            f"best objective={best_metrics.objective:.2f}"
                        )

                if not improved:
                    print("  no further improvement")
                    break
        else:
            print("\nsearch skipped (--search-preset none)")
    finally:
        if executor is not None:
            executor.shutdown()

    print("\nbest tuning:")
    profile_tuning = normalize_config_for_json({**best_config, "sigma": float(args.sigma)})
    print(json.dumps(profile_tuning, indent=2, ensure_ascii=False))
    print(f"best metrics={best_metrics}")
    print(f"baseline objective={baseline_metrics.objective:.2f}")
    print(f"best objective={best_metrics.objective:.2f}")

    if args.dry_run:
        return 0

    profile_payload = {
        "format_version": HYBRID_PROFILE_FORMAT_VERSION,
        "score_json": str(score_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tuning": profile_tuning,
        "baseline_metrics": asdict(baseline_metrics),
        "best_metrics": asdict(best_metrics),
        "calibration": {
            "level": str(args.level),
            "modes": list(modes),
            "starts": starts,
            "offset_states": int(args.offset_states),
            "max_starts": int(args.max_starts),
            "jump_states": int(args.jump_states),
            "jump_prime_states": int(args.jump_prime_states),
            "passes": int(args.passes),
            "sigma": float(args.sigma),
            "search_preset": str(args.search_preset),
            "max_jump_scenarios": int(args.max_jump_scenarios),
            "parallel_workers": int(parallel_workers),
        },
    }
    output_path.write_text(
        json.dumps(profile_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote profile: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
