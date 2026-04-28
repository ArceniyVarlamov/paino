from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Callable

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import numpy as np

from hybrid_fusion import HybridScoreFollower


@dataclass(frozen=True)
class Event:
    pitch: int
    timestamp: float
    target_index: int
    role: str


@dataclass(frozen=True)
class EventTrace:
    event_index: int
    pitch: int
    timestamp: float
    target_index: int
    role: str
    prediction: int
    confidence: float
    hsmm_index: int
    oltw_index: int
    selected_model: str
    forced_advance: bool
    resynced: bool


@dataclass(frozen=True)
class CheckResult:
    success: bool
    lost_event: int | None
    message: str


@dataclass(frozen=True)
class TestCase:
    name: str
    description: str
    events: list[Event]
    checker: Callable[["TestCase", list[EventTrace], list[dict[str, int | float]]], CheckResult]
    score: list[dict[str, int | float]] | None = None


@dataclass(frozen=True)
class TestResult:
    name: str
    success: bool
    final_index: int
    confidence_avg: float
    lost_event: int | None
    message: str
    failure_trace: EventTrace | None


def build_reference_score() -> list[dict[str, int | float]]:
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    return [
        {
            "index": index,
            "pitch": pitch,
            "nominal_duration": 0.5,
        }
        for index, pitch in enumerate(pitches)
    ]


def score_pitch(score: list[dict[str, int | float]], index: int) -> int:
    return int(score[index]["pitch"])


def average_absolute_error(traces: list[EventTrace]) -> float:
    if not traces:
        return 0.0
    return float(
        np.mean(
            [abs(trace.prediction - trace.target_index) for trace in traces],
            dtype=np.float64,
        )
    )


def first_backward_event(traces: list[EventTrace]) -> int | None:
    for previous, current in zip(traces, traces[1:], strict=False):
        if current.prediction < previous.prediction:
            return current.event_index
    return None


def first_large_jump_event(traces: list[EventTrace], *, max_step: int) -> int | None:
    for previous, current in zip(traces, traces[1:], strict=False):
        if (current.prediction - previous.prediction) > max_step:
            return current.event_index
    return None


def first_error_event(
    traces: list[EventTrace],
    *,
    tolerance: int,
    roles: set[str] | None = None,
) -> int | None:
    for trace in traces:
        if roles is not None and trace.role not in roles:
            continue
        if abs(trace.prediction - trace.target_index) > tolerance:
            return trace.event_index
    return None


def fail(event_index: int | None, message: str) -> CheckResult:
    return CheckResult(success=False, lost_event=event_index, message=message)


def succeed(message: str) -> CheckResult:
    return CheckResult(success=True, lost_event=None, message=message)


def check_case_stall(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case, score
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards during a stall")

    first_advance_event = next(
        (trace.event_index for trace in traces if trace.prediction > 0),
        None,
    )
    if first_advance_event is None or first_advance_event > 5:
        return fail(5, "OLTW fail-safe did not advance by the fifth repeated note")

    if traces[-1].prediction < 2:
        return fail(traces[-1].event_index, "tracker did not continue advancing under a long stall")

    return succeed("stall handling advanced as expected")


def check_case_skip(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case, score
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards during a skip")

    if traces[2].prediction < 3:
        return fail(
            traces[2].event_index,
            "tracker lagged too far behind the large score skip at event 3",
        )

    if traces[-1].prediction < 5:
        return fail(
            traces[-1].event_index,
            "tracker failed to resynchronize to the skipped-ahead region",
        )

    return succeed("large skip was recovered")


def check_case_noise(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case, score
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards under noisy inputs")

    noisy_error_event = first_error_event(traces, tolerance=1, roles={"noise"})
    if noisy_error_event is not None:
        return fail(noisy_error_event, "a random noise note pulled the tracker too far off course")

    score_error_event = first_error_event(traces, tolerance=1, roles={"score"})
    if score_error_event is not None:
        return fail(score_error_event, "the tracker failed to recover on a ground-truth note")

    return succeed("noise was absorbed without losing the main path")


def check_case_fast(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards at high speed")

    if traces[-1].prediction != len(score) - 1:
        return fail(traces[-1].event_index, "tracker did not reach the final state at 3x tempo")

    avg_error = average_absolute_error(traces)
    if avg_error > 1.0:
        error_event = first_error_event(traces, tolerance=2) or traces[-1].event_index
        return fail(error_event, f"average alignment error is too high ({avg_error:.2f})")

    return succeed("3x tempo remained trackable")


def check_case_wrong_pitch(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards under consistently wrong pitches")

    large_jump_event = first_large_jump_event(traces, max_step=2)
    if large_jump_event is not None:
        return fail(large_jump_event, "tracker made an implausibly large jump on weak evidence")

    avg_error = average_absolute_error(traces)
    if avg_error > 1.5:
        error_event = first_error_event(traces, tolerance=2) or traces[-1].event_index
        return fail(error_event, f"average alignment error is too high ({avg_error:.2f})")

    if traces[-1].prediction < (len(score) // 2):
        return fail(traces[-1].event_index, "tracker failed to make meaningful progress")

    return succeed("tracker stayed stable under weak pitch evidence")


def check_case_backward_time(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards after a timestamp reversal")

    avg_error = average_absolute_error(traces)
    if avg_error > 1.0:
        error_event = first_error_event(traces, tolerance=2) or traces[-1].event_index
        return fail(error_event, f"timestamp jitter caused too much alignment drift ({avg_error:.2f})")

    if traces[-1].prediction < len(score[:6]) - 2:
        return fail(traces[-1].event_index, "tracker failed to recover after timestamps went backwards")

    return succeed("non-monotonic timestamps were clamped safely")


def check_case_repeat_final(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case
    final_index = len(score) - 1
    reach_event = next((trace.event_index for trace in traces if trace.prediction == final_index), None)
    if reach_event is None:
        return fail(traces[-1].event_index, "tracker never reached the final note")

    for trace in traces[reach_event:]:
        if trace.prediction != final_index:
            return fail(trace.event_index, "tracker drifted away after reaching the final note")

    return succeed("final-state repetitions remained stable")


def check_case_double_hit(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards on repeated correct notes")

    avg_error = average_absolute_error(traces)
    if avg_error > 1.0:
        error_event = first_error_event(traces, tolerance=1) or traces[-1].event_index
        return fail(error_event, f"double-hit sequence drifted too far ({avg_error:.2f})")

    if traces[-1].prediction != len(score) - 1:
        return fail(traces[-1].event_index, "tracker did not finish the score on doubled notes")

    return succeed("repeated correct notes stayed aligned")


def check_case_late_recovery(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case, score
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards during recovery")

    recovery_traces = traces[-4:]
    recovery_error = average_absolute_error(recovery_traces)
    if recovery_error > 0.75:
        error_event = first_error_event(recovery_traces, tolerance=1) or traces[-1].event_index
        return fail(error_event, f"tracker did not relock quickly enough ({recovery_error:.2f})")

    return succeed("tracker recovered after a bad opening")


def check_case_ornament(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case, score
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards during ornament handling")

    early_predictions = [trace.prediction for trace in traces[:4]]
    if any(prediction != 0 for prediction in early_predictions):
        ornament_event = next(
            trace.event_index for trace in traces[:4] if trace.prediction != 0
        )
        return fail(ornament_event, "ornaments caused premature advancement")

    if traces[-1].prediction < 2:
        return fail(traces[-1].event_index, "tracker did not resume the main melody after ornaments")

    return succeed("ornaments stayed local and recoverable")


def check_case_far_outlier(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case, score
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards after a far outlier")

    outlier_trace = traces[3]
    if outlier_trace.prediction > 6:
        return fail(
            outlier_trace.event_index,
            "single far outlier caused an implausibly large forward jump",
        )

    if traces[-1].prediction > 6:
        return fail(
            traces[-1].event_index,
            "tracker did not recover locally after the far outlier",
        )

    return succeed("far outlier was treated as a local disturbance")


def check_case_same_timestamp_burst(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case, score
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards when timestamps were identical")

    if traces[-1].prediction < 4:
        return fail(
            traces[-1].event_index,
            "tracker failed to keep up with a dense burst of correct notes at one timestamp",
        )

    avg_error = average_absolute_error(traces)
    if avg_error > 0.8:
        error_event = first_error_event(traces, tolerance=1) or traces[-1].event_index
        return fail(error_event, f"dense zero-delta timing drifted too far ({avg_error:.2f})")

    return succeed("identical timestamps stayed trackable")


def check_case_final_noise_lock(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case
    final_index = len(score) - 1
    reach_event = next((trace.event_index for trace in traces if trace.prediction == final_index), None)
    if reach_event is None:
        return fail(traces[-1].event_index, "tracker never reached the final note before noisy tail")

    for trace in traces[reach_event:]:
        if trace.prediction != final_index:
            return fail(trace.event_index, "tracker left the final state because of post-finish noise")

    return succeed("final lock held against noisy post-finish inputs")


def check_case_outlier_burst(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case, score
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards during an outlier burst")

    outlier_error_event = first_error_event(traces, tolerance=1, roles={"outlier"})
    if outlier_error_event is not None:
        return fail(outlier_error_event, "burst outliers pulled the tracker too far off course")

    if traces[-1].prediction < 4:
        return fail(traces[-1].event_index, "tracker failed to recover after the outlier burst")

    return succeed("multiple far outliers stayed local and recoverable")


def check_case_mid_stall_recovery(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case, score
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards during a mid-score stall")

    if not any(trace.forced_advance for trace in traces[4:8]):
        return fail(8, "run-count recovery never triggered during the repeated mid-score note")

    tail_error = average_absolute_error(traces[-3:])
    if tail_error > 0.75:
        error_event = first_error_event(traces[-3:], tolerance=1) or traces[-1].event_index
        return fail(error_event, f"tracker did not relock cleanly after the stall ({tail_error:.2f})")

    if traces[-1].prediction < 6:
        return fail(traces[-1].event_index, "tracker did not make it back onto the main path")

    return succeed("mid-score stall recovered without runaway drift")


def check_case_octave_confusion(
    case: TestCase,
    traces: list[EventTrace],
    score: list[dict[str, int | float]],
) -> CheckResult:
    del case, score
    backward_event = first_backward_event(traces)
    if backward_event is not None:
        return fail(backward_event, "prediction moved backwards under octave-shifted evidence")

    large_jump_event = first_large_jump_event(traces, max_step=1)
    if large_jump_event is not None:
        return fail(large_jump_event, "octave-shifted notes caused an implausibly fast jump")

    if any(trace.prediction > 1 for trace in traces):
        jump_event = next(trace.event_index for trace in traces if trace.prediction > 1)
        return fail(jump_event, "wrong-octave notes looked too much like a valid forward path")

    return succeed("octave confusion stayed bounded and conservative")


def build_test_cases(score: list[dict[str, int | float]]) -> list[TestCase]:
    rng = np.random.default_rng(20260419)
    far_noise_pool = np.asarray([41, 43, 46, 84, 86, 89], dtype=np.int64)
    noise_pitches = [int(value) for value in rng.choice(far_noise_pool, size=3, replace=False)]

    stall_pitch = score_pitch(score, 0)
    repeat_final_pitch = score_pitch(score, len(score) - 1)
    far_outlier_score = [
        {
            "index": index,
            "pitch": pitch,
            "nominal_duration": 0.5,
        }
        for index, pitch in enumerate(([60, 62, 64, 65, 67, 69, 71, 72] * 10))
    ]
    far_outlier_score[60]["pitch"] = 88

    return [
        TestCase(
            name="CASE_STALL",
            description="Same correct note repeated 10 times.",
            events=[
                Event(
                    pitch=stall_pitch,
                    timestamp=0.05 * event_index,
                    target_index=0 if event_index < 4 else 1 if event_index < 8 else 2,
                    role="stall",
                )
                for event_index in range(10)
            ],
            checker=check_case_stall,
        ),
        TestCase(
            name="CASE_SKIP",
            description="Large forward jump from score note 1 to 4.",
            events=[
                Event(score_pitch(score, 0), 0.0, 0, "score"),
                Event(score_pitch(score, 1), 0.5, 1, "score"),
                Event(score_pitch(score, 4), 1.0, 4, "score"),
                Event(score_pitch(score, 5), 1.5, 5, "score"),
            ],
            checker=check_case_skip,
        ),
        TestCase(
            name="CASE_NOISE",
            description="Correct notes interleaved with random outliers.",
            events=[
                Event(score_pitch(score, 0), 0.0, 0, "score"),
                Event(noise_pitches[0], 0.2, 0, "noise"),
                Event(score_pitch(score, 1), 0.5, 1, "score"),
                Event(noise_pitches[1], 0.7, 1, "noise"),
                Event(score_pitch(score, 2), 1.0, 2, "score"),
                Event(noise_pitches[2], 1.2, 2, "noise"),
                Event(score_pitch(score, 3), 1.5, 3, "score"),
            ],
            checker=check_case_noise,
        ),
        TestCase(
            name="CASE_FAST",
            description="Whole score played at 3x speed.",
            events=[
                Event(
                    pitch=score_pitch(score, index),
                    timestamp=index * (0.5 / 3.0),
                    target_index=index,
                    role="score",
                )
                for index in range(len(score))
            ],
            checker=check_case_fast,
        ),
        TestCase(
            name="CASE_WRONG_PITCH",
            description="C-minor scale played against the C-major reference.",
            events=[
                Event(pitch, index * 0.5, index, "wrong_pitch")
                for index, pitch in enumerate([60, 62, 63, 65, 67, 68, 70, 72])
            ],
            checker=check_case_wrong_pitch,
        ),
        TestCase(
            name="CASE_BACKWARD_TIME",
            description="Timestamps occasionally go backwards.",
            events=[
                Event(score_pitch(score, 0), 0.0, 0, "score"),
                Event(score_pitch(score, 1), 0.4, 1, "score"),
                Event(score_pitch(score, 2), 0.2, 2, "score"),
                Event(score_pitch(score, 3), 0.6, 3, "score"),
                Event(score_pitch(score, 4), 0.5, 4, "score"),
                Event(score_pitch(score, 5), 0.9, 5, "score"),
            ],
            checker=check_case_backward_time,
        ),
        TestCase(
            name="CASE_REPEAT_FINAL",
            description="Reach the last note and keep repeating it.",
            events=[
                *[
                    Event(score_pitch(score, index), index * 0.5, index, "score")
                    for index in range(len(score))
                ],
                *[
                    Event(
                        repeat_final_pitch,
                        4.0 + (repeat_index * 0.05),
                        len(score) - 1,
                        "final_repeat",
                    )
                    for repeat_index in range(6)
                ],
            ],
            checker=check_case_repeat_final,
        ),
        TestCase(
            name="CASE_DOUBLE_HIT",
            description="Every score note is played twice.",
            events=[
                Event(
                    score_pitch(score, index // 2),
                    step * 0.25,
                    index // 2,
                    "double_hit",
                )
                for step, index in enumerate(range(len(score) * 2))
            ],
            checker=check_case_double_hit,
        ),
        TestCase(
            name="CASE_LATE_RECOVERY",
            description="Bad opening notes, then a clean entry onto the score.",
            events=[
                Event(86, 0.0, 0, "noise"),
                Event(43, 0.2, 0, "noise"),
                Event(score_pitch(score, 0), 0.5, 0, "score"),
                Event(score_pitch(score, 1), 1.0, 1, "score"),
                Event(score_pitch(score, 2), 1.5, 2, "score"),
                Event(score_pitch(score, 3), 2.0, 3, "score"),
            ],
            checker=check_case_late_recovery,
        ),
        TestCase(
            name="CASE_ORNAMENT",
            description="Short ornamental notes around the opening pitch.",
            events=[
                Event(score_pitch(score, 0), 0.0, 0, "score"),
                Event(61, 0.05, 0, "ornament"),
                Event(score_pitch(score, 0), 0.10, 0, "score"),
                Event(61, 0.15, 0, "ornament"),
                Event(score_pitch(score, 1), 0.5, 1, "score"),
                Event(score_pitch(score, 2), 1.0, 2, "score"),
            ],
            checker=check_case_ornament,
        ),
        TestCase(
            name="CASE_FAR_OUTLIER",
            description="One extreme note must not yank the tracker deep into the future.",
            score=far_outlier_score,
            events=[
                Event(int(far_outlier_score[0]["pitch"]), 0.0, 0, "score"),
                Event(int(far_outlier_score[1]["pitch"]), 0.5, 1, "score"),
                Event(int(far_outlier_score[2]["pitch"]), 1.0, 2, "score"),
                Event(88, 1.5, 2, "outlier"),
                Event(int(far_outlier_score[3]["pitch"]), 2.0, 3, "score"),
                Event(int(far_outlier_score[4]["pitch"]), 2.5, 4, "score"),
            ],
            checker=check_case_far_outlier,
        ),
        TestCase(
            name="CASE_SAME_TIMESTAMP_BURST",
            description="Several correct notes arrive with exactly the same timestamp.",
            events=[
                Event(score_pitch(score, 0), 0.0, 0, "score"),
                Event(score_pitch(score, 1), 0.0, 1, "score"),
                Event(score_pitch(score, 2), 0.0, 2, "score"),
                Event(score_pitch(score, 3), 0.0, 3, "score"),
                Event(score_pitch(score, 4), 0.0, 4, "score"),
            ],
            checker=check_case_same_timestamp_burst,
        ),
        TestCase(
            name="CASE_FINAL_NOISE_LOCK",
            description="After reaching the final note, random outliers must not unlock the final state.",
            events=[
                *[
                    Event(score_pitch(score, index), index * 0.5, index, "score")
                    for index in range(len(score))
                ],
                Event(43, 3.7, len(score) - 1, "noise"),
                Event(86, 3.9, len(score) - 1, "noise"),
                Event(41, 4.1, len(score) - 1, "noise"),
            ],
            checker=check_case_final_noise_lock,
        ),
        TestCase(
            name="CASE_OUTLIER_BURST",
            description="Multiple far outliers in a row should remain a local disturbance.",
            events=[
                Event(score_pitch(score, 0), 0.0, 0, "score"),
                Event(score_pitch(score, 1), 0.5, 1, "score"),
                Event(score_pitch(score, 2), 1.0, 2, "score"),
                Event(86, 1.2, 2, "outlier"),
                Event(41, 1.4, 2, "outlier"),
                Event(89, 1.6, 2, "outlier"),
                Event(score_pitch(score, 3), 2.0, 3, "score"),
                Event(score_pitch(score, 4), 2.5, 4, "score"),
            ],
            checker=check_case_outlier_burst,
        ),
        TestCase(
            name="CASE_MID_STALL_RECOVERY",
            description="A repeated mid-score note should trigger local recovery, then relock on the melody.",
            events=[
                Event(score_pitch(score, 0), 0.0, 0, "score"),
                Event(score_pitch(score, 1), 0.5, 1, "score"),
                Event(score_pitch(score, 2), 1.0, 2, "score"),
                Event(score_pitch(score, 3), 1.5, 3, "score"),
                Event(score_pitch(score, 3), 1.55, 3, "stall"),
                Event(score_pitch(score, 3), 1.60, 3, "stall"),
                Event(score_pitch(score, 3), 1.65, 3, "stall"),
                Event(score_pitch(score, 3), 1.70, 4, "stall"),
                Event(score_pitch(score, 4), 2.30, 4, "score"),
                Event(score_pitch(score, 5), 2.80, 5, "score"),
                Event(score_pitch(score, 6), 3.30, 6, "score"),
            ],
            checker=check_case_mid_stall_recovery,
        ),
        TestCase(
            name="CASE_OCTAVE_CONFUSION",
            description="Correct pitch classes in the wrong octave must not look like a strong forward path.",
            events=[
                Event(score_pitch(score, 0), 0.0, 0, "score"),
                Event(score_pitch(score, 1) + 12, 0.5, 0, "wrong_octave"),
                Event(score_pitch(score, 2) + 12, 1.0, 0, "wrong_octave"),
                Event(score_pitch(score, 3) + 12, 1.5, 0, "wrong_octave"),
                Event(score_pitch(score, 4) + 12, 2.0, 0, "wrong_octave"),
            ],
            checker=check_case_octave_confusion,
        ),
    ]


def run_test_case(
    case: TestCase,
    score: list[dict[str, int | float]],
) -> TestResult:
    case_score = case.score if case.score is not None else score
    follower = HybridScoreFollower(case_score, load_tuning_profile=False)
    traces: list[EventTrace] = []

    try:
        for event_index, event in enumerate(case.events, start=1):
            prediction = int(follower.process_event(event.pitch, event.timestamp))
            traces.append(
                EventTrace(
                    event_index=event_index,
                    pitch=event.pitch,
                    timestamp=event.timestamp,
                    target_index=event.target_index,
                    role=event.role,
                    prediction=prediction,
                    confidence=float(follower.confidence),
                    hsmm_index=int(follower.last_hsmm_index),
                    oltw_index=int(follower.last_oltw_index),
                    selected_model=str(follower.last_selected_model),
                    forced_advance=bool(follower.oltw.last_forced_advance),
                    resynced=bool(follower.last_resynced),
                )
            )

        check = case.checker(case, traces, case_score)
    except Exception as exc:
        return TestResult(
            name=case.name,
            success=False,
            final_index=traces[-1].prediction if traces else -1,
            confidence_avg=float(np.mean([trace.confidence for trace in traces])) if traces else 0.0,
            lost_event=(traces[-1].event_index + 1) if traces else 1,
            message=f"exception during run: {exc}",
            failure_trace=traces[-1] if traces else None,
        )

    failure_trace = None
    if check.lost_event is not None:
        failure_trace = next(
            (trace for trace in traces if trace.event_index == check.lost_event),
            None,
        )

    return TestResult(
        name=case.name,
        success=check.success,
        final_index=traces[-1].prediction if traces else -1,
        confidence_avg=float(np.mean([trace.confidence for trace in traces])) if traces else 0.0,
        lost_event=check.lost_event,
        message=check.message,
        failure_trace=failure_trace,
    )


def format_failure_details(result: TestResult) -> str:
    if result.failure_trace is None or result.lost_event is None:
        return f"{result.name}: lost at event #{result.lost_event}: {result.message}"

    trace = result.failure_trace
    return (
        f"{result.name}: lost at event #{result.lost_event}: {result.message} "
        f"(pitch={trace.pitch}, target={trace.target_index}, pred={trace.prediction}, "
        f"conf={trace.confidence:.3f}, hsmm={trace.hsmm_index}, oltw={trace.oltw_index}, "
        f"selected={trace.selected_model}, forced={trace.forced_advance}, resync={trace.resynced})"
    )


def main() -> None:
    score = build_reference_score()
    cases = build_test_cases(score)
    results = [run_test_case(case, score) for case in cases]

    score_summary = ", ".join(str(int(note["pitch"])) for note in score)
    print(f"Reference score pitches: {score_summary}")
    print(f"Total tests: {len(results)}")
    print()

    name_width = 28
    header = (
        f"{'Test Name':<{name_width}}"
        f"{'Success':<10}"
        f"{'Final Index':>14}"
        f"{'Confidence Avg':>18}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        status = "PASS" if result.success else "FAIL"
        print(
            f"{result.name:<{name_width}}"
            f"{status:<10}"
            f"{result.final_index:>14}"
            f"{result.confidence_avg:>18.3f}"
        )

    failed_results = [result for result in results if not result.success]
    print()
    print(f"Passed: {len(results) - len(failed_results)}/{len(results)}")

    if failed_results:
        print("Failure details:")
        for result in failed_results:
            print(format_failure_details(result))


if __name__ == "__main__":
    main()
