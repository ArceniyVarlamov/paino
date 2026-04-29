from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.append(vendor_path)

from hmm_follower import ScoreFollowerHMM
from live_midi_receiver import LiveMidiReceiver, MidiEmulator, mido as live_mido


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime score-following entry point for the HMM tracker.",
    )
    parser.add_argument(
        "score_json",
        type=Path,
        help="Path to the score JSON file.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian emission sigma in semitones (default: %(default)s).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.01,
        help="Seconds between queue polls in live mode (default: %(default)s).",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--midi-file",
        type=Path,
        help="Replay a MIDI file in realtime via MidiEmulator.",
    )
    source_group.add_argument(
        "--live",
        action="store_true",
        help="Use LiveMidiReceiver to read note_on events from a MIDI input port.",
    )

    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Specific MIDI input port name for --live mode.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the emulated MIDI file until interrupted.",
    )
    return parser


def looks_like_midi_path(value: str | None) -> bool:
    if not value:
        return False

    candidate = Path(value)
    return candidate.suffix.lower() in {".mid", ".midi"}


def available_input_ports() -> list[str]:
    if live_mido is None:
        return []

    try:
        return list(live_mido.get_input_names())
    except Exception:
        return []


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.live and looks_like_midi_path(args.port):
        parser.error(
            "`--live --port` expects a MIDI input port name, not a MIDI file path. "
            "Use `--midi-file <path.mid>` for playback from a MIDI file."
        )


def resolve_score_path(path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return path

    if suffix in {".mid", ".midi"}:
        sibling_json = path.with_suffix(".json")
        if sibling_json.exists():
            print(f"[INFO] Using sibling score JSON: {sibling_json}")
            return sibling_json

        raise SystemExit(
            f"No sibling score JSON found for {path.name}. "
            "Run `midi_to_score.py` first or pass the `.json` file directly."
        )

    raise SystemExit(
        f"Unsupported score input: {path}. "
        "Pass a score `.json`, or a `.mid`/`.midi` that already has a sibling `.json`."
    )


def format_prediction(
    follower: ScoreFollowerHMM,
    event: dict[str, float | int],
    predicted_index: int,
    event_count: int,
    session_start_ts: float,
) -> str:
    relative_time = float(event["timestamp"]) - session_start_ts
    score_pitch = int(follower.pitches[predicted_index])
    probability = float(follower.alpha[predicted_index])
    return (
        f"[LIVE] #{event_count:04d} "
        f"t={relative_time:7.3f}s "
        f"pitch={int(event['pitch']):>3} "
        f"-> progress={predicted_index + 1:>3}/{follower.N} "
        f"(idx={predicted_index:>3}, score_pitch={score_pitch:>3}, prob={probability:0.3f})"
    )


def run_tracker(args: argparse.Namespace) -> int:
    follower = ScoreFollowerHMM(args.score_json, sigma=args.sigma)

    if args.live:
        ports = available_input_ports()
        if not ports:
            print("[ERROR] No live MIDI input ports are available.")
            print(
                "[ERROR] Use `--midi-file <path.mid>` for file playback, or connect a MIDI "
                "device/virtual port before using `--live`."
            )
            return 2

        if args.port is not None and args.port not in ports:
            print(f"[ERROR] MIDI input port not found: {args.port}")
            print("[ERROR] Available input ports:")
            for port_name in ports:
                print(f"  - {port_name}")
            return 2

        source = LiveMidiReceiver(
            args.port,
            poll_interval=args.poll_interval,
            open_immediately=False,
        )
        source_label = f"live MIDI input ({args.port or 'default port'})"
    else:
        source = MidiEmulator(
            args.midi_file,
            loop=args.loop,
            start_immediately=False,
        )
        source_label = f"MIDI emulation from {args.midi_file}"

    print(f"[INFO] Score: {args.score_json}")
    print(f"[INFO] States: {follower.N}, sigma={follower.sigma:.2f}")
    print(f"[INFO] Source: {source_label}")
    print("[INFO] Press Ctrl+C to stop.")

    event_count = 0
    session_start_ts: float | None = None

    try:
        with source:
            while True:
                events = source.get_events()

                for event in events:
                    if session_start_ts is None:
                        session_start_ts = float(event["timestamp"])

                    predicted_index = follower.process_event(event)
                    event_count += 1
                    print(
                        format_prediction(
                            follower,
                            event,
                            predicted_index,
                            event_count,
                            session_start_ts,
                        )
                    )

                if args.live:
                    time.sleep(args.poll_interval)
                    continue

                if not source.is_running and not events:
                    break

                time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
        return 130
    except ModuleNotFoundError as exc:
        if exc.name == "rtmidi":
            print("[ERROR] Live MIDI mode requires the `python-rtmidi` package.")
            print(
                "[ERROR] Install it into the local environment or use `--midi-file` instead."
            )
            return 2
        raise
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 2

    print(f"[INFO] Finished after {event_count} events.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    args.score_json = resolve_score_path(args.score_json)
    return run_tracker(args)


if __name__ == "__main__":
    raise SystemExit(main())
