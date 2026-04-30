from __future__ import annotations

import argparse
import heapq
import json
import os
import queue
import re
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.append(vendor_path)

import numpy as np

from compat import compat_zip

try:
    import mido
    import pygame
    import pygame.midi
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"{exc.name} is not installed. Install it into the local .vendor directory first."
    ) from exc

from hybrid_fusion import HybridScoreFollower
from midi_workspace import (
    DEFAULT_CHORD_EPSILONS as WORKSPACE_DEFAULT_CHORD_EPSILONS,
    DEFAULT_LIBRARY_ROOT as WORKSPACE_DEFAULT_LIBRARY_ROOT,
    DEFAULT_SPLIT_POINTS as WORKSPACE_DEFAULT_SPLIT_POINTS,
    import_piece_workspace_with_progress,
    load_library_entries,
    pick_midi_file,
)
from midi.real_orchestra_player import DynamicOrchestraPlayer
from output_dispatcher import PygameMidiOrchestra, ScoreEventDispatcher, TempoTracker
from portable_paths import resolve_project_path

DEFAULT_SCORE_PATH = Path(__file__).resolve().parent / "generated_dataset" / "ideal.json"
SAMPLE_RATE = 44_100
VISIBLE_PIANO_START = 21
VISIBLE_PIANO_END = 108
KEYBOARD_PIANO_START = 36
KEYBOARD_PIANO_END = 96
SOUND_START = 0
SOUND_END = 127
REAL_PIANO_SAMPLE_DIR = (
    Path(__file__).resolve().parent / "assets" / "piano_samples" / "salamander_mp3"
)
ORCHESTRA_SAMPLE_DIR = Path(__file__).resolve().parent / "assets" / "orchestra_samples"
DEFAULT_MIDI_TEMPO = 500000
WHITE_PITCH_CLASSES = {0, 2, 4, 5, 7, 9, 11}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SAMPLE_NOTE_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

BACKGROUND = (247, 242, 235)
SURFACE = (236, 229, 219)
SURFACE_ALT = (225, 235, 245)
TEXT_COLOR = (28, 32, 40)
SUBTLE_TEXT = (85, 91, 102)
ACCENT = (35, 110, 215)
ACCENT_SOFT = (212, 231, 255)
SUCCESS = (51, 153, 98)
WARNING = (195, 84, 58)
BUTTON_IDLE = (224, 234, 245)
BUTTON_HOVER = (210, 225, 243)
BUTTON_ACTIVE = (35, 110, 215)
BUTTON_ACTIVE_HOVER = (28, 96, 190)
ORCHESTRA_VOLUME_MIN = 0.05
ORCHESTRA_VOLUME_MAX = 2.0
ORCHESTRA_VOLUME_DEFAULT = 1.0
ORCHESTRA_VOLUME_STEP = 0.10
MIN_AUTOPLAY_GAP = 0.012
LIVE_FOLLOW_BATCH_SECONDS = 0.025
SETTINGS_PATH = Path(__file__).resolve().parent / "interactive_tester_settings.json"

WINDOW_PADDING = 48
HEADER_HEIGHT = 188
PIANO_TOP = 246
WHITE_KEY_WIDTH = 26
WHITE_KEY_HEIGHT = 370
BLACK_KEY_WIDTH = 16
BLACK_KEY_HEIGHT = 220


@dataclass(frozen=True)
class PianoKey:
    midi_pitch: int
    rect: pygame.Rect
    is_black: bool


@dataclass(frozen=True)
class LiveMidiMessage:
    type: str
    note: int
    velocity: int


class PygameLiveMidiInputPort:
    def __init__(self, device_id: int) -> None:
        self.device_id = int(device_id)
        self._input = pygame.midi.Input(self.device_id)

    def iter_pending(self) -> list[LiveMidiMessage]:
        messages: list[LiveMidiMessage] = []
        while self._input.poll():
            batch = self._input.read(64)
            if not batch:
                break
            for payload, _timestamp in batch:
                if len(payload) < 3:
                    continue
                status = int(payload[0]) & 0xF0
                note = int(payload[1])
                velocity = int(payload[2])
                if status == 0x90:
                    messages.append(LiveMidiMessage("note_on", note, velocity))
                elif status == 0x80:
                    messages.append(LiveMidiMessage("note_off", note, velocity))
        return messages

    def close(self) -> None:
        self._input.close()


WHITE_KEY_COUNT = sum(
    1
    for pitch in range(VISIBLE_PIANO_START, VISIBLE_PIANO_END + 1)
    if pitch % 12 in WHITE_PITCH_CLASSES
)
WINDOW_SIZE = (
    max(1500, (WINDOW_PADDING * 2) + (WHITE_KEY_COUNT * WHITE_KEY_WIDTH)),
    720,
)
PIANO_LEFT = (WINDOW_SIZE[0] - (WHITE_KEY_COUNT * WHITE_KEY_WIDTH)) // 2


def clamp_orchestra_mix(value: float) -> float:
    return float(np.clip(float(value), ORCHESTRA_VOLUME_MIN, ORCHESTRA_VOLUME_MAX))


def apply_orchestra_mix_level(base_level: float, mix_scale: float) -> float:
    normalized_level = float(np.clip(float(base_level), 0.0, 1.0))
    normalized_mix = clamp_orchestra_mix(mix_scale)
    if normalized_mix <= 1.0:
        return float(np.clip(normalized_level * normalized_mix, 0.0, 1.0))

    boost_progress = (normalized_mix - 1.0) / max(1e-6, ORCHESTRA_VOLUME_MAX - 1.0)
    boosted_level = normalized_level + ((1.0 - normalized_level) * boost_progress)
    return float(np.clip(boosted_level, 0.0, 1.0))

KEYBOARD_ROWS = (
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'"],
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]"],
)
KEY_LABEL_TO_CODE = {
    "Q": pygame.K_q,
    "W": pygame.K_w,
    "E": pygame.K_e,
    "R": pygame.K_r,
    "T": pygame.K_t,
    "Y": pygame.K_y,
    "U": pygame.K_u,
    "I": pygame.K_i,
    "O": pygame.K_o,
    "P": pygame.K_p,
    "[": pygame.K_LEFTBRACKET,
    "]": pygame.K_RIGHTBRACKET,
    "A": pygame.K_a,
    "S": pygame.K_s,
    "D": pygame.K_d,
    "F": pygame.K_f,
    "G": pygame.K_g,
    "H": pygame.K_h,
    "J": pygame.K_j,
    "K": pygame.K_k,
    "L": pygame.K_l,
    ";": pygame.K_SEMICOLON,
    "'": pygame.K_QUOTE,
    "Z": pygame.K_z,
    "X": pygame.K_x,
    "C": pygame.K_c,
    "V": pygame.K_v,
    "B": pygame.K_b,
    "N": pygame.K_n,
    "M": pygame.K_m,
    ",": pygame.K_COMMA,
    ".": pygame.K_PERIOD,
    "/": pygame.K_SLASH,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive QWERTY piano for testing the hybrid realtime score follower.",
    )
    parser.add_argument(
        "score_json",
        nargs="?",
        type=Path,
        default=DEFAULT_SCORE_PATH,
        help=(
            "Target score JSON. A `.mid`/`.midi` path is also accepted if a sibling `.json` "
            f"with the same stem already exists (default: {DEFAULT_SCORE_PATH})."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian emission sigma in semitones (default: %(default)s).",
    )
    parser.add_argument(
        "--orchestra-midi",
        type=Path,
        default=None,
        help="Optional orchestra MIDI file to play dynamically in the background.",
    )
    parser.add_argument(
        "--practice-hand",
        choices=("left", "right"),
        default=None,
        help=(
            "Study mode. Load `<hand>_hand.json` for tracking and the opposite hand MIDI "
            "as accompaniment from the same folder as the provided score path."
        ),
    )
    parser.add_argument(
        "--midi-out",
        type=int,
        default=-1,
        help=(
            "MIDI output device ID for the real orchestra player. "
            "Use -1 to select the system default output automatically."
        ),
    )
    parser.add_argument(
        "--piano-midi-out",
        type=int,
        default=None,
        help=(
            "Optional MIDI output device ID for live piano notes from the QWERTY keyboard/mouse. "
            "Use this to route your played notes into GarageBand or Audio MIDI Setup."
        ),
    )
    parser.add_argument(
        "--piano-midi-channel",
        type=int,
        default=1,
        help="MIDI channel for live piano notes, 1-16 in musician terms (default: %(default)s).",
    )
    parser.add_argument(
        "--orchestra-midi-channel",
        type=int,
        default=None,
        help=(
            "MIDI channel for orchestra playback, 1-16 in musician terms. "
            "Default is channel 2 in practice mode and channel 3 in full orchestra mode."
        ),
    )
    parser.add_argument(
        "--merge-orchestra-to-channel",
        action="store_true",
        help=(
            "Collapse the orchestra MIDI onto the single channel selected by "
            "--orchestra-midi-channel. This is now the default and the flag is kept "
            "for backwards-compatible commands."
        ),
    )
    parser.add_argument(
        "--preserve-orchestra-channels",
        action="store_true",
        help=(
            "Advanced mode: preserve source MIDI channels by offsetting them from "
            "--orchestra-midi-channel. Requires a multitimbral synth/Logic setup."
        ),
    )
    parser.add_argument(
        "--orchestra-volume",
        type=float,
        default=ORCHESTRA_VOLUME_DEFAULT,
        help=(
            "Master orchestra mix scale from "
            f"{ORCHESTRA_VOLUME_MIN:.2f} to {ORCHESTRA_VOLUME_MAX:.1f} "
            "(default: %(default)s). Values above 1.0 boost the orchestra mix."
        ),
    )
    parser.add_argument(
        "--force-instrument",
        type=int,
        default=None,
        help=(
            "Optional General MIDI program override (0-127). "
            "When set, force both live piano MIDI and orchestra MIDI to this instrument."
        ),
    )
    parser.add_argument(
        "--force-orchestra-instrument",
        type=int,
        default=None,
        help=(
            "Optional General MIDI program override (0-127) for the orchestra only. "
            "Use 48 for the old single-track strings behavior without changing live piano."
        ),
    )
    parser.add_argument(
        "--mute-local-piano",
        action="store_true",
        help="Disable local Salamander/sample playback and use only MIDI piano output for live notes.",
    )
    parser.add_argument(
        "--local-practice-audio",
        action="store_true",
        help=(
            "In one-hand study mode, play the accompaniment with local piano samples "
            "instead of routing it through MIDI/Logic. Works with --practice-hand or "
            "with direct left_hand.json/right_hand.mid and right_hand.json/left_hand.mid pairs."
        ),
    )
    parser.add_argument(
        "--local-orchestra",
        dest="local_orchestra",
        action="store_true",
        default=True,
        help="Play orchestra accompaniment with the built-in local sample engine instead of Logic/IAC MIDI.",
    )
    parser.add_argument(
        "--midi-orchestra",
        dest="local_orchestra",
        action="store_false",
        help="Force orchestra accompaniment through external MIDI output, for example Logic Pro via IAC.",
    )
    parser.add_argument(
        "--fallback-midi-orchestra",
        action="store_true",
        help="Enable the old simple MIDI accompaniment stub when --orchestra-midi is not provided.",
    )
    parser.add_argument(
        "--midi-routing-test",
        action="store_true",
        help=(
            "Send a short direct MIDI test to the configured piano/orchestra ports and exit. "
            "This bypasses score-following and orchestra playback."
        ),
    )
    parser.add_argument(
        "--live-midi-in",
        action="store_true",
        help=(
            "Use live MIDI note input from an external keyboard (for example, a digital piano) "
            "to drive score-following."
        ),
    )
    parser.add_argument(
        "--midi-in-port",
        type=str,
        default=None,
        help=(
            "Specific MIDI input port name (or unique substring) for --live-midi-in. "
            "If omitted, the app auto-selects the best non-IAC port."
        ),
    )
    parser.add_argument(
        "--setup-wizard",
        action="store_true",
        help="Open the first-run MIDI setup wizard before launching the tester.",
    )
    parser.add_argument(
        "--launcher",
        action="store_true",
        help="Open the startup launcher instead of jumping directly into a score.",
    )
    return parser


def cli_option_present(flag: str) -> bool:
    for token in sys.argv[1:]:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def load_saved_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}

    try:
        raw_data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Failed to read saved tester settings: {exc}")
        return {}

    if not isinstance(raw_data, dict):
        print("[WARN] Ignoring saved tester settings: expected a JSON object.")
        return {}

    settings: dict[str, Any] = {}
    for key in (
        "midi_out",
        "piano_midi_out",
        "piano_midi_channel",
        "orchestra_midi_channel",
        "orchestra_volume",
        "midi_in_port",
        "local_orchestra",
    ):
        if key in raw_data:
            settings[key] = raw_data[key]

    for key in ("live_midi_in", "mute_local_piano"):
        if key in raw_data:
            settings[key] = bool(raw_data[key])

    return settings


def save_settings(settings: dict[str, Any]) -> None:
    serializable = {
        "midi_out": settings.get("midi_out"),
        "piano_midi_out": settings.get("piano_midi_out"),
        "piano_midi_channel": settings.get("piano_midi_channel"),
        "orchestra_midi_channel": settings.get("orchestra_midi_channel"),
        "orchestra_volume": settings.get("orchestra_volume"),
        "local_orchestra": bool(settings.get("local_orchestra", True)),
        "live_midi_in": bool(settings.get("live_midi_in", False)),
        "midi_in_port": settings.get("midi_in_port"),
        "mute_local_piano": bool(settings.get("mute_local_piano", False)),
    }
    try:
        SETTINGS_PATH.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[WARN] Failed to save tester settings: {exc}")


def apply_saved_settings(args: argparse.Namespace) -> dict[str, Any]:
    saved = load_saved_settings()
    if not saved:
        return {}

    if not cli_option_present("--midi-out") and "midi_out" in saved:
        args.midi_out = -1 if saved["midi_out"] is None else int(saved["midi_out"])
    if not cli_option_present("--piano-midi-out") and "piano_midi_out" in saved:
        args.piano_midi_out = None if saved["piano_midi_out"] is None else int(saved["piano_midi_out"])
    if not cli_option_present("--piano-midi-channel") and "piano_midi_channel" in saved:
        args.piano_midi_channel = int(saved["piano_midi_channel"])
    if not cli_option_present("--orchestra-midi-channel") and "orchestra_midi_channel" in saved:
        args.orchestra_midi_channel = (
            None
            if saved["orchestra_midi_channel"] is None
            else int(saved["orchestra_midi_channel"])
        )
    if not cli_option_present("--orchestra-volume") and "orchestra_volume" in saved:
        args.orchestra_volume = float(saved["orchestra_volume"])
    if (
        not cli_option_present("--local-orchestra")
        and not cli_option_present("--midi-orchestra")
        and "local_orchestra" in saved
    ):
        args.local_orchestra = bool(saved["local_orchestra"])
    if not cli_option_present("--live-midi-in") and "live_midi_in" in saved:
        args.live_midi_in = bool(saved["live_midi_in"])
    if not cli_option_present("--midi-in-port") and "midi_in_port" in saved:
        args.midi_in_port = saved["midi_in_port"]
    if not cli_option_present("--mute-local-piano") and "mute_local_piano" in saved:
        args.mute_local_piano = bool(saved["mute_local_piano"])

    return saved


def resolve_score_path(path: Path) -> Path:
    resolved = resolve_project_path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        if not resolved.exists():
            raise SystemExit(f"Score JSON not found: {resolved}")
        return resolved

    if suffix in {".mid", ".midi"}:
        sibling_json = resolved.with_suffix(".json")
        if sibling_json.exists():
            print(f"[INFO] Using sibling score JSON: {sibling_json}")
            return sibling_json

        raise SystemExit(
            f"No sibling score JSON found for {resolved.name}. "
            "Run `midi_to_score.py` first or pass the `.json` file directly."
        )

    raise SystemExit(
        f"Unsupported score input: {resolved}. "
        "Pass a score `.json`, or a `.mid`/`.midi` that already has a sibling `.json`."
    )


def resolve_optional_midi_path(path: Path | None) -> Path | None:
    if path is None:
        return None

    resolved = resolve_project_path(path)
    suffix = resolved.suffix.lower()
    if suffix not in {".mid", ".midi"}:
        raise SystemExit(f"Unsupported orchestra MIDI input: {resolved}")
    if not resolved.exists():
        raise SystemExit(f"Orchestra MIDI file not found: {resolved}")
    return resolved


def midi_device_name(device_id: int) -> str:
    info = pygame.midi.get_device_info(device_id)
    if info is None:
        return f"#{device_id} missing"
    return info[1].decode(errors="ignore")


def available_midi_output_devices() -> list[tuple[int, str]]:
    outputs: list[tuple[int, str]] = []
    for device_id in range(pygame.midi.get_count()):
        info = pygame.midi.get_device_info(device_id)
        if info is None or not bool(info[3]):
            continue
        outputs.append((int(device_id), info[1].decode(errors="ignore")))
    return outputs


def describe_available_midi_output_devices() -> str:
    outputs = available_midi_output_devices()
    if not outputs:
        return "No MIDI output devices found."
    return "\n".join(f"  - {device_id}: {device_name}" for device_id, device_name in outputs)


def resolve_midi_output_id(requested_output_id: int) -> int:
    requested_output_id = int(requested_output_id)
    if requested_output_id >= 0:
        info = pygame.midi.get_device_info(requested_output_id)
        if info is None:
            raise RuntimeError(
                f"MIDI output device {requested_output_id} does not exist.\n"
                f"Available MIDI output devices:\n{describe_available_midi_output_devices()}"
            )
        if not bool(info[3]):
            raise RuntimeError(
                f"MIDI device {requested_output_id} is not an output port.\n"
                f"Available MIDI output devices:\n{describe_available_midi_output_devices()}"
            )
        return requested_output_id

    default_output_id = pygame.midi.get_default_output_id()
    if default_output_id >= 0:
        return int(default_output_id)

    for device_id in range(pygame.midi.get_count()):
        info = pygame.midi.get_device_info(device_id)
        if info is not None and bool(info[3]):
            return int(device_id)

    raise RuntimeError("No MIDI output device found.")


def available_midi_input_ports() -> list[str]:
    return [name for _device_id, name in available_midi_input_devices()]


def available_midi_input_devices() -> list[tuple[int, str]]:
    devices: list[tuple[int, str]] = []
    if not pygame.midi.get_init():
        return devices
    for device_id in range(pygame.midi.get_count()):
        info = pygame.midi.get_device_info(device_id)
        if info is None or not bool(info[2]):
            continue
        devices.append((int(device_id), info[1].decode(errors="ignore")))
    return devices


def is_virtual_midi_input_port_name(port_name: str) -> bool:
    lowered = str(port_name).strip().lower()
    return any(
        token in lowered
        for token in (
            "iac",
            "logic pro virtual out",
            "network session",
            "loopback",
        )
    )


def choose_default_midi_input_port(port_names: list[str]) -> str:
    if not port_names:
        raise RuntimeError("No MIDI input ports are available.")

    normalized = [(name, name.lower()) for name in port_names]

    preferred_exact = next((name for name, lower_name in normalized if lower_name == "digital piano"), None)
    if preferred_exact is not None:
        return preferred_exact

    preferred_keyword = next(
        (
            name
            for name, lower_name in normalized
            if any(keyword in lower_name for keyword in ("digital piano", "keyboard", "piano"))
        ),
        None,
    )
    if preferred_keyword is not None:
        return preferred_keyword

    non_virtual = [
        name
        for name, lower_name in normalized
        if "iac" not in lower_name and "logic pro virtual out" not in lower_name
    ]
    if non_virtual:
        return non_virtual[0]

    return port_names[0]


def choose_default_midi_input_device(devices: list[tuple[int, str]]) -> tuple[int, str]:
    if not devices:
        raise RuntimeError("No MIDI input ports are available.")

    preferred_name = choose_default_midi_input_port([name for _device_id, name in devices])
    for device_id, name in devices:
        if name == preferred_name:
            return device_id, name
    return devices[0]


def preferred_auto_midi_input_port(port_names: list[str]) -> str | None:
    physical_ports = [name for name in port_names if not is_virtual_midi_input_port_name(name)]
    if not physical_ports:
        return None
    return choose_default_midi_input_port(physical_ports)


def sanitize_auto_selected_midi_input_port(
    requested_port: str | None,
    port_names: list[str],
) -> tuple[str | None, str | None]:
    preferred_port = preferred_auto_midi_input_port(port_names)
    requested = requested_port.strip() if isinstance(requested_port, str) and requested_port.strip() else None

    if preferred_port is None:
        if requested is None:
            return None, None
        if requested in port_names and not is_virtual_midi_input_port_name(requested):
            return requested, None
        return None, "Only virtual MIDI inputs were detected; live MIDI input was disabled."

    if requested is None:
        return preferred_port, None
    if requested == preferred_port:
        return requested, None
    if requested not in port_names:
        return preferred_port, (
            f"Saved MIDI input '{requested}' is unavailable; using '{preferred_port}' instead."
        )
    if is_virtual_midi_input_port_name(requested):
        return preferred_port, (
            f"Saved MIDI input '{requested}' is virtual; using '{preferred_port}' instead."
        )
    return requested, None


def choose_default_midi_output_id(output_devices: list[tuple[int, str]]) -> int | None:
    if not output_devices:
        return None

    normalized = [(device_id, name, name.lower()) for device_id, name in output_devices]
    preferred_iac = next(
        (device_id for device_id, _name, lower_name in normalized if "iac" in lower_name),
        None,
    )
    if preferred_iac is not None:
        return preferred_iac

    preferred_non_keyboard = next(
        (
            device_id
            for device_id, _name, lower_name in normalized
            if not any(keyword in lower_name for keyword in ("digital piano", "keyboard", "piano"))
        ),
        None,
    )
    if preferred_non_keyboard is not None:
        return preferred_non_keyboard

    return int(output_devices[0][0])


def settings_require_setup(
    args: argparse.Namespace,
    *,
    settings_exist: bool,
    detected_outputs: list[tuple[int, str]],
    detected_inputs: list[str],
) -> bool:
    if bool(args.setup_wizard):
        return True
    if not settings_exist:
        return True

    output_ids = {device_id for device_id, _name in detected_outputs}
    if not bool(args.local_orchestra) and int(args.midi_out) >= 0 and int(args.midi_out) not in output_ids:
        return True
    if args.piano_midi_out is not None and int(args.piano_midi_out) not in output_ids:
        return True

    if bool(args.live_midi_in):
        if not detected_inputs:
            return True
        if args.midi_in_port is not None and args.midi_in_port not in detected_inputs:
            return True

    return False


def resolve_midi_input_port_name(requested_port: str | None) -> str:
    _device_id, port_name = resolve_midi_input_device(requested_port)
    return port_name


def resolve_midi_input_device(requested_port: str | None) -> tuple[int, str]:
    devices = available_midi_input_devices()
    if not devices:
        raise RuntimeError("No MIDI input ports are available.")

    if requested_port is None or not requested_port.strip():
        preferred_port = preferred_auto_midi_input_port([name for _device_id, name in devices])
        if preferred_port is not None:
            for device_id, name in devices:
                if name == preferred_port:
                    return device_id, name
        return choose_default_midi_input_device(devices)

    requested = requested_port.strip()
    exact_matches = [(device_id, name) for device_id, name in devices if name == requested]
    if exact_matches:
        return exact_matches[0]

    needle = requested.lower()
    matches = [(device_id, name) for device_id, name in devices if needle in name.lower()]
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        listing = "\n  - ".join(f"{device_id}: {name}" for device_id, name in matches)
        raise RuntimeError(
            "MIDI input port is ambiguous. Matched multiple ports:\n"
            f"  - {listing}\n"
            "Pass the exact --midi-in-port value."
        )

    listing = "\n  - ".join(f"{device_id}: {name}" for device_id, name in devices)
    raise RuntimeError(
        f"MIDI input port not found: {requested_port}\n"
        "Available ports:\n"
        f"  - {listing}"
    )


def resolved_orchestra_midi_channel(args: argparse.Namespace) -> int:
    if args.orchestra_midi_channel is not None:
        return max(1, min(16, int(args.orchestra_midi_channel)))
    return 2 if args.practice_hand is not None else 3


def run_midi_routing_test(args: argparse.Namespace) -> int:
    pygame.midi.init()
    outputs: dict[int, pygame.midi.Output] = {}
    try:
        orchestra_output_id = resolve_midi_output_id(args.midi_out)
        piano_output_id = resolve_midi_output_id(
            args.piano_midi_out if args.piano_midi_out is not None else orchestra_output_id
        )
        piano_channel = max(0, min(15, int(args.piano_midi_channel) - 1))
        orchestra_channel = max(0, min(15, resolved_orchestra_midi_channel(args) - 1))

        for output_id in sorted({piano_output_id, orchestra_output_id}):
            outputs[output_id] = pygame.midi.Output(output_id, latency=0)

        piano_output = outputs[piano_output_id]
        orchestra_output = outputs[orchestra_output_id]

        print(
            f"[MIDI TEST] Piano: output #{piano_output_id} "
            f"({midi_device_name(piano_output_id)}), channel {piano_channel + 1}"
        )
        print(
            f"[MIDI TEST] Orchestra: output #{orchestra_output_id} "
            f"({midi_device_name(orchestra_output_id)}), channel {orchestra_channel + 1}"
        )
        print("[MIDI TEST] Listen for: 3 piano notes, then 3 orchestra-string chords.")

        piano_output.write_short(0xC0 | piano_channel, 0, 0)
        piano_output.write_short(0xB0 | piano_channel, 7, 118)
        piano_output.write_short(0xB0 | piano_channel, 10, 64)
        piano_output.write_short(0xB0 | piano_channel, 11, 127)
        piano_output.write_short(0xB0 | piano_channel, 91, 0)
        piano_output.write_short(0xB0 | piano_channel, 93, 0)
        orchestra_output.write_short(0xC0 | orchestra_channel, 48, 0)
        orchestra_output.write_short(0xB0 | orchestra_channel, 7, 127)
        orchestra_output.write_short(0xB0 | orchestra_channel, 11, 120)
        orchestra_output.write_short(0xB0 | orchestra_channel, 10, 64)
        orchestra_output.write_short(0xB0 | orchestra_channel, 91, 0)
        orchestra_output.write_short(0xB0 | orchestra_channel, 93, 0)

        for note in (60, 64, 67):
            piano_output.write_short(0x90 | piano_channel, note, 112)
            time.sleep(0.18)
            piano_output.write_short(0x80 | piano_channel, note, 0)
            time.sleep(0.08)

        for chord in ((48, 55, 60), (50, 57, 62), (52, 59, 64)):
            for note in chord:
                orchestra_output.write_short(0x90 | orchestra_channel, note, 116)
            time.sleep(0.45)
            for note in chord:
                orchestra_output.write_short(0x80 | orchestra_channel, note, 0)
            time.sleep(0.12)

        for output_id, output in outputs.items():
            del output_id
            for channel in range(16):
                output.write_short(0xB0 | channel, 120, 0)
                output.write_short(0xB0 | channel, 123, 0)

        return 0
    finally:
        for output in outputs.values():
            output.close()
        pygame.midi.quit()


def hand_display_name(hand: str) -> str:
    return "Left Hand" if hand == "left" else "Right Hand"


def other_hand(hand: str) -> str:
    return "right" if hand == "left" else "left"


def resolve_practice_materials(
    reference_path: Path,
    practice_hand: str,
) -> tuple[Path, Path, str, str, Path]:
    expanded_reference = reference_path.expanduser()
    resolved_reference = expanded_reference.resolve()
    if resolved_reference.exists() and resolved_reference.is_dir():
        base_dir = resolved_reference
    elif resolved_reference.suffix:
        base_dir = resolved_reference.parent
    else:
        base_dir = resolved_reference

    practice_score_candidate = base_dir / f"{practice_hand}_hand.json"
    if not practice_score_candidate.exists():
        practice_score_midi_candidate = base_dir / f"{practice_hand}_hand.mid"
        if practice_score_midi_candidate.exists():
            practice_score_candidate = practice_score_midi_candidate

    accompaniment_hand = other_hand(practice_hand)
    accompaniment_candidate = base_dir / f"{accompaniment_hand}_hand.mid"

    score_path = resolve_score_path(practice_score_candidate)
    accompaniment_midi_path = resolve_optional_midi_path(accompaniment_candidate)
    if accompaniment_midi_path is None:
        raise SystemExit(f"Accompaniment MIDI file not found: {accompaniment_candidate}")

    display_reference = resolved_reference if resolved_reference.exists() else base_dir
    return (
        score_path,
        accompaniment_midi_path,
        hand_display_name(practice_hand),
        hand_display_name(accompaniment_hand),
        display_reference,
    )


def display_piece_title(reference_path: Path | None, fallback_name: str) -> str:
    if reference_path is None:
        return fallback_name.replace("_", " ").strip()

    candidate = reference_path.stem.replace("_", " ").strip()
    if reference_path.stem.lower() in {"left_hand", "right_hand"}:
        candidate = reference_path.parent.name.replace("_", " ").strip()

    return candidate or fallback_name.replace("_", " ").strip()


def is_hand_study_pair(score_path: Path, accompaniment_path: Path | None) -> bool:
    if accompaniment_path is None:
        return False

    score_stem = score_path.stem.lower()
    accompaniment_stem = accompaniment_path.stem.lower()
    return (
        score_stem in {"left_hand", "right_hand"}
        and accompaniment_stem in {"left_hand", "right_hand"}
        and score_stem != accompaniment_stem
    )


def is_white_key(midi_pitch: int) -> bool:
    return midi_pitch % 12 in WHITE_PITCH_CLASSES


def clamp_midi_pitch(midi_pitch: int) -> int:
    return int(max(SOUND_START, min(SOUND_END, int(midi_pitch))))


def is_visible_piano_pitch(midi_pitch: int) -> bool:
    pitch = int(midi_pitch)
    return VISIBLE_PIANO_START <= pitch <= VISIBLE_PIANO_END


def pitch_to_note_name(midi_pitch: int) -> str:
    octave = (midi_pitch // 12) - 1
    return f"{NOTE_NAMES[midi_pitch % 12]}{octave}"


def pitch_to_sample_name(midi_pitch: int) -> str:
    octave = (midi_pitch // 12) - 1
    return f"{SAMPLE_NOTE_NAMES[midi_pitch % 12]}{octave}"


def midi_to_frequency(midi_pitch: int) -> float:
    return 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))


def sample_path_for_pitch(midi_pitch: int) -> Path:
    return REAL_PIANO_SAMPLE_DIR / f"{pitch_to_sample_name(midi_pitch)}.mp3"


def load_real_piano_sound(midi_pitch: int) -> pygame.mixer.Sound | None:
    sample_path = sample_path_for_pitch(midi_pitch)
    if not sample_path.exists():
        return None

    try:
        sound = pygame.mixer.Sound(str(sample_path))
    except pygame.error:
        return None

    sound.set_volume(0.82)
    return sound


def normalized_triangle(phase: np.ndarray) -> np.ndarray:
    return (2.0 / np.pi) * np.arcsin(np.sin(phase))


def apply_lowpass(
    signal: np.ndarray,
    cutoff_hz: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    alpha = 1.0 - np.exp((-2.0 * np.pi * max(20.0, cutoff_hz)) / sample_rate)
    filtered = np.empty_like(signal)
    filtered[0] = signal[0]
    for index in range(1, signal.size):
        filtered[index] = filtered[index - 1] + alpha * (signal[index] - filtered[index - 1])
    return filtered


def white_pitches_in_range(start_pitch: int, end_pitch: int) -> list[int]:
    return [pitch for pitch in range(start_pitch, end_pitch + 1) if is_white_key(pitch)]


def build_keyboard_map() -> tuple[dict[int, int], dict[int, str]]:
    white_pitches = white_pitches_in_range(KEYBOARD_PIANO_START, KEYBOARD_PIANO_END)
    labels = [label for row in KEYBOARD_ROWS for label in row]
    keyboard_map: dict[int, int] = {}
    pitch_labels: dict[int, str] = {}

    for label, pitch in compat_zip(labels, white_pitches):
        key_code = KEY_LABEL_TO_CODE[label]
        keyboard_map[key_code] = pitch
        pitch_labels[pitch] = label

    return keyboard_map, pitch_labels


def build_piano_layout() -> tuple[list[PianoKey], list[PianoKey], dict[int, PianoKey]]:
    white_keys: list[PianoKey] = []
    black_keys: list[PianoKey] = []
    pitch_to_key: dict[int, PianoKey] = {}
    white_index = 0

    for pitch in range(VISIBLE_PIANO_START, VISIBLE_PIANO_END + 1):
        if is_white_key(pitch):
            rect = pygame.Rect(
                PIANO_LEFT + (white_index * WHITE_KEY_WIDTH),
                PIANO_TOP,
                WHITE_KEY_WIDTH,
                WHITE_KEY_HEIGHT,
            )
            key = PianoKey(midi_pitch=pitch, rect=rect, is_black=False)
            white_keys.append(key)
            pitch_to_key[pitch] = key
            white_index += 1
            continue

        rect = pygame.Rect(
            PIANO_LEFT + (white_index * WHITE_KEY_WIDTH) - (BLACK_KEY_WIDTH // 2),
            PIANO_TOP,
            BLACK_KEY_WIDTH,
            BLACK_KEY_HEIGHT,
        )
        key = PianoKey(midi_pitch=pitch, rect=rect, is_black=True)
        black_keys.append(key)
        pitch_to_key[pitch] = key

    return white_keys, black_keys, pitch_to_key


def pitch_at_position(
    position: tuple[int, int],
    white_keys: list[PianoKey],
    black_keys: list[PianoKey],
) -> int | None:
    for key in black_keys:
        if key.rect.collidepoint(position):
            return key.midi_pitch

    for key in white_keys:
        if key.rect.collidepoint(position):
            return key.midi_pitch

    return None


def make_piano_sound(
    midi_pitch: int,
    *,
    sample_rate: int = SAMPLE_RATE,
    duration: float | None = None,
) -> pygame.mixer.Sound:
    frequency = midi_to_frequency(midi_pitch)
    if duration is None:
        duration = float(np.clip(4.4 - ((midi_pitch - 30) * 0.038), 1.5, 4.2))

    times = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    rng = np.random.default_rng(midi_pitch * 13 + 7)
    base_phase = 2.0 * np.pi * frequency * times

    detunes = (-0.0014, 0.0, 0.0011)
    string_mix = np.zeros_like(times)
    for detune in detunes:
        detuned_phase = base_phase * (1.0 + detune)
        voice = np.zeros_like(times)
        for harmonic in range(1, 11):
            inharmonicity = 1.0 + (0.00009 * harmonic * harmonic * (frequency / 180.0))
            partial_phase = detuned_phase * harmonic * inharmonicity
            phase_offset = float(rng.uniform(-0.18, 0.18))
            partial_weight = np.exp(-0.54 * (harmonic - 1))
            partial_weight *= 1.05 if harmonic == 1 else 1.0
            partial_decay = np.exp(-times * (0.75 + (harmonic * 0.42) + (frequency / 4200.0)))
            voice += partial_weight * np.sin(partial_phase + phase_offset) * partial_decay
        string_mix += voice

    low_resonance = (
        0.22 * np.sin(2.0 * np.pi * max(46.0, frequency * 0.5) * times)
        + 0.12 * np.sin(2.0 * np.pi * max(92.0, frequency) * times)
        + 0.06 * np.sin(2.0 * np.pi * max(138.0, frequency * 1.5) * times)
    ) * np.exp(-1.05 * times)

    hammer_noise = rng.normal(0.0, 1.0, times.size)
    hammer_noise = apply_lowpass(hammer_noise, cutoff_hz=1800.0 + frequency * 2.0)
    hammer_noise *= np.exp(-70.0 * times)
    hammer_noise *= 0.035

    sympathetic = (
        0.10 * np.sin(base_phase * 0.5 + 0.14)
        + 0.06 * np.sin(base_phase * 0.25 + 0.38)
    ) * np.exp(-0.82 * times)

    attack = np.clip(times / 0.012, 0.0, 1.0)
    body = np.exp(-0.96 * times)
    release = np.exp(-5.2 * np.maximum(0.0, times - duration * 0.68))
    envelope = attack * body * release

    mono = (
        0.84 * string_mix
        + 0.22 * low_resonance
        + 0.10 * sympathetic
        + hammer_noise
    ) * envelope
    mono = apply_lowpass(mono, cutoff_hz=1700.0 + frequency * 5.5)
    mono = np.tanh(mono * 1.18)

    stereo_left = apply_lowpass(
        mono + (0.018 * np.sin(base_phase * 0.51 + 0.3) * np.exp(-1.6 * times)),
        cutoff_hz=2100.0 + frequency * 3.2,
    )
    stereo_right = apply_lowpass(
        mono + (0.017 * np.sin(base_phase * 0.49 - 0.18) * np.exp(-1.5 * times)),
        cutoff_hz=2200.0 + frequency * 3.0,
    )
    stereo = np.column_stack((stereo_left, stereo_right))
    fade_out = np.minimum(1.0, (duration - times) / 0.03)
    stereo *= np.clip(fade_out, 0.0, 1.0)[:, None]
    stereo *= 0.23
    audio = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)
    sound = pygame.sndarray.make_sound(audio)
    sound.set_volume(0.60)
    return sound


class PygameMidiPianoOutput:
    def __init__(
        self,
        output_id: int,
        *,
        channel: int = 15,
        program: int | None = None,
        velocity: int = 108,
        midi_output: pygame.midi.Output | None = None,
        write_lock: Any | None = None,
    ) -> None:
        if not (1 <= int(channel) + 1 <= 16):
            raise ValueError("piano MIDI channel must be between 1 and 16")

        self._shared_output = midi_output is not None
        self._owns_midi_init = False
        if midi_output is None and not pygame.midi.get_init():
            pygame.midi.init()
            self._owns_midi_init = True

        resolved_output_id = self._resolve_output_id(output_id) if midi_output is None else int(output_id)
        self._resolved_output_id = int(resolved_output_id)
        self._output = pygame.midi.Output(resolved_output_id, latency=0) if midi_output is None else midi_output
        self._write_lock = write_lock
        self.channel = int(channel)
        self.program = None if program is None else int(max(0, min(127, program)))
        self.velocity = int(max(1, min(127, velocity)))
        self.status_label = f"MIDI piano output #{resolved_output_id} (ch {self.channel + 1})"
        self._active_counts: dict[int, int] = {}
        if self.program is not None:
            self._write_short(0xC0 | self.channel, self.program, 0)

    @property
    def midi_output_id(self) -> int:
        return int(self._resolved_output_id)

    def _resolve_output_id(self, requested_output_id: int) -> int:
        if requested_output_id >= 0:
            info = pygame.midi.get_device_info(requested_output_id)
            if info is None:
                raise RuntimeError(
                    f"Piano MIDI output device {requested_output_id} does not exist.\n"
                    f"Available MIDI output devices:\n{describe_available_midi_output_devices()}"
                )
            if not bool(info[3]):
                raise RuntimeError(
                    f"MIDI device {requested_output_id} is not an output port.\n"
                    f"Available MIDI output devices:\n{describe_available_midi_output_devices()}"
                )
            return int(requested_output_id)

        output_id = pygame.midi.get_default_output_id()
        if output_id >= 0:
            return int(output_id)

        for device_id in range(pygame.midi.get_count()):
            info = pygame.midi.get_device_info(device_id)
            if info is not None and bool(info[3]):
                return int(device_id)

        raise RuntimeError("No MIDI output device found for live piano notes.")

    def press_note(self, midi_pitch: int, *, velocity: int | None = None) -> None:
        pitch = int(max(SOUND_START, min(SOUND_END, midi_pitch)))
        current_count = self._active_counts.get(pitch, 0)
        if current_count == 0:
            note_velocity = int(max(1, min(127, velocity if velocity is not None else self.velocity)))
            self._write_short(0x90 | self.channel, pitch, note_velocity)
        self._active_counts[pitch] = current_count + 1

    def release_note(self, midi_pitch: int) -> None:
        pitch = int(max(SOUND_START, min(SOUND_END, midi_pitch)))
        current_count = self._active_counts.get(pitch, 0)
        if current_count <= 1:
            if current_count <= 0:
                return
            self._active_counts.pop(pitch, None)
            self._write_short(0x80 | self.channel, pitch, 0)
            return
        self._active_counts[pitch] = current_count - 1

    def panic(self) -> None:
        active_pitches = sorted(self._active_counts)
        self._active_counts.clear()
        if not pygame.midi.get_init():
            return
        for pitch in active_pitches:
            self._write_short(0x80 | self.channel, pitch, 0)
        self._write_short(0xB0 | self.channel, 64, 0)
        self._write_short(0xB0 | self.channel, 120, 0)
        self._write_short(0xB0 | self.channel, 121, 0)
        self._write_short(0xB0 | self.channel, 123, 0)

    def close(self) -> None:
        self.panic()
        if not self._shared_output:
            self._output.close()
        if self._owns_midi_init and pygame.midi.get_init():
            pygame.midi.quit()

    def _write_short(self, status: int, data1: int, data2: int) -> None:
        if self._write_lock is None:
            self._output.write_short(status, data1, data2)
            return
        with self._write_lock:
            self._output.write_short(status, data1, data2)


@dataclass(frozen=True)
class HandPracticeNoteEvent:
    source_time: float
    note: int
    velocity: int
    duration: float


@dataclass(order=True)
class HandPracticeScheduledNoteOff:
    due_time: float
    order: int
    note: int
    generation: int


def dispatcher_event_anchor_time(dispatcher: ScoreEventDispatcher, clock: Callable[[], float]) -> float:
    now = float(clock())
    event_timestamp = getattr(dispatcher, "current_event_timestamp", None)
    if not isinstance(event_timestamp, (int, float)):
        return now

    event_time = float(event_timestamp)
    if (now - 2.0) <= event_time <= (now + 0.05):
        return min(event_time, now)
    return now


def load_hand_practice_note_events(midi_path: str | Path) -> list[HandPracticeNoteEvent]:
    midi_file = mido.MidiFile(midi_path)
    absolute_time = 0.0
    open_notes: dict[tuple[int, int], list[tuple[float, int]]] = {}
    note_spans: list[tuple[int, int, float, int, float]] = []
    sustain_down_since: dict[int, float] = {}
    sustain_intervals: dict[int, list[tuple[float, float]]] = {}
    events: list[HandPracticeNoteEvent] = []

    for message in midi_file:
        absolute_time += float(getattr(message, "time", 0.0))
        if getattr(message, "is_meta", False):
            continue

        message_type = getattr(message, "type", None)
        if message_type == "control_change" and int(getattr(message, "control", -1)) == 64:
            channel = int(getattr(message, "channel", 0))
            value = int(getattr(message, "value", 0))
            if value >= 64:
                sustain_down_since.setdefault(channel, absolute_time)
            else:
                start_time = sustain_down_since.pop(channel, None)
                if start_time is not None:
                    sustain_intervals.setdefault(channel, []).append(
                        (float(start_time), float(absolute_time))
                    )
            continue

        if message_type == "note_on" and int(getattr(message, "velocity", 0)) > 0:
            key = (int(getattr(message, "channel", 0)), int(message.note))
            open_notes.setdefault(key, []).append(
                (absolute_time, int(max(1, min(127, message.velocity))))
            )
            continue

        if message_type == "note_off" or (
            message_type == "note_on" and int(getattr(message, "velocity", 0)) == 0
        ):
            key = (int(getattr(message, "channel", 0)), int(message.note))
            note_stack = open_notes.get(key)
            if not note_stack:
                continue
            start_time, velocity = note_stack.pop()
            if not note_stack:
                open_notes.pop(key, None)
            note_spans.append(
                (
                    int(getattr(message, "channel", 0)),
                    int(message.note),
                    float(start_time),
                    int(velocity),
                    float(absolute_time),
                )
            )

    for channel, start_time in sustain_down_since.items():
        sustain_intervals.setdefault(channel, []).append((float(start_time), float(absolute_time)))

    for (channel, note), note_stack in open_notes.items():
        for start_time, velocity in note_stack:
            note_spans.append(
                (int(channel), int(note), float(start_time), int(velocity), float(absolute_time))
            )

    for channel, note, start_time, velocity, end_time in note_spans:
        sustained_end_time = float(end_time)
        for sustain_start, sustain_end in sustain_intervals.get(channel, []):
            if sustain_start <= end_time <= sustain_end:
                sustained_end_time = max(sustained_end_time, sustain_end)
                break

        events.append(
            HandPracticeNoteEvent(
                source_time=float(start_time),
                note=int(note),
                velocity=int(velocity),
                duration=max(0.05, float(sustained_end_time - start_time)),
            )
        )

    events.sort(key=lambda event: (event.source_time, event.note))
    return events


class HandPracticeMidiAccompaniment:
    """Lightweight two-hand practice MIDI player.

    This intentionally leaves Logic's sound settings alone. It uses note events
    and sustain timing from the hand MIDI, while the full orchestra path remains
    handled by DynamicOrchestraPlayer.
    """

    _MIN_TEMPO_RATIO = 0.25
    _WAIT_GRANULARITY = 0.002
    _SEEK_TIME_THRESHOLD = 0.75

    def __init__(
        self,
        midi_path: str | Path,
        dispatcher: ScoreEventDispatcher,
        *,
        midi_output_id: int = -1,
        channel: int = 1,
        program: int | None = None,
        volume_scale: float = ORCHESTRA_VOLUME_DEFAULT,
        midi_output: pygame.midi.Output | None = None,
        write_lock: Any | None = None,
    ) -> None:
        self._midi_path = Path(midi_path)
        self._dispatcher = dispatcher
        self._requested_midi_output_id = int(midi_output_id)
        self._channel = int(max(0, min(15, channel)))
        self._program = None if program is None else int(max(0, min(127, program)))
        self._volume_scale = clamp_orchestra_mix(volume_scale)
        self._shared_output = midi_output is not None
        self._owns_midi_init = False
        if midi_output is None and not pygame.midi.get_init():
            pygame.midi.init()
            self._owns_midi_init = True

        self._resolved_midi_output_id = (
            resolve_midi_output_id(self._requested_midi_output_id)
            if midi_output is None
            else int(midi_output_id)
        )
        self._output = (
            midi_output
            if midi_output is not None
            else pygame.midi.Output(self._resolved_midi_output_id, latency=0)
        )
        self._output_lock = write_lock or threading.Lock()
        self._clock = time.monotonic
        self._events = self._load_note_events()
        self._source_times = np.asarray(
            [event.source_time for event in self._events],
            dtype=np.float64,
        )
        self._event_index = 0
        self._master_index: int | None = None
        self._master_target_time: float | None = None
        self._master_next_target_time: float | None = None
        self._master_anchor_clock_time: float | None = None
        self._tempo_ratio = 1.0
        self._transport_paused = True
        self._seek_request_time: float | None = None
        self._last_source_time: float | None = None
        self._pending_note_offs: list[HandPracticeScheduledNoteOff] = []
        self._note_off_counter = 0
        self._note_generation_counter = 0
        self._active_note_generations: dict[int, int] = {}
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.status_label = (
            f"Hand practice MIDI output #{self._resolved_midi_output_id} "
            f"(ch {self._channel + 1})"
        )

        self._initialize_channel()
        self._dispatcher.subscribe(self.handle_dispatch)

    @property
    def midi_output_id(self) -> int:
        return int(self._resolved_midi_output_id)

    def set_volume_scale(self, volume_scale: float) -> None:
        with self._state_lock:
            self._volume_scale = clamp_orchestra_mix(volume_scale)

    @property
    def shared_output(self) -> pygame.midi.Output | None:
        return self._output

    @property
    def output_lock(self) -> Any:
        return self._output_lock

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def source_pitches(self) -> list[int]:
        return sorted({int(event.note) for event in self._events})

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._play_loop,
            name="HandPracticeMidiAccompaniment",
            daemon=True,
        )
        self._thread.start()

    def close(self, timeout: float = 1.0) -> None:
        self._dispatcher.unsubscribe(self.handle_dispatch)
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        self.panic()
        if not self._shared_output:
            self._output.close()
        if self._owns_midi_init and pygame.midi.get_init():
            pygame.midi.quit()

    def resume(self) -> None:
        with self._state_lock:
            self._transport_paused = False

    def halt(self) -> None:
        with self._state_lock:
            self._transport_paused = True
            self._master_index = None
            self._master_target_time = None
            self._master_next_target_time = None
            self._master_anchor_clock_time = None
            self._seek_request_time = None
            self._event_index = 0
            self._last_source_time = None
        self.panic()

    def panic(self) -> None:
        with self._state_lock:
            active_notes = sorted(self._active_note_generations)
            self._active_note_generations.clear()
            self._pending_note_offs.clear()
            self._note_off_counter = 0
            self._note_generation_counter = 0
        for note in active_notes:
            self._send_note_off(note)
        with self._output_lock:
            self._output.write_short(0xB0 | self._channel, 64, 0)
            self._output.write_short(0xB0 | self._channel, 120, 0)
            self._output.write_short(0xB0 | self._channel, 121, 0)
            self._output.write_short(0xB0 | self._channel, 123, 0)

    def reset_to_start(self) -> None:
        self.seek(0.0)

    def seek(self, target_time: float) -> None:
        with self._state_lock:
            self._seek_request_time = max(0.0, float(target_time))
            self._transport_paused = False

    def handle_dispatch(self, index: int, tempo_ratio: float) -> None:
        new_index = int(index)
        target_time = self._score_index_to_target_time(new_index)
        next_target_time = self._score_index_to_next_target_time(new_index)
        anchor_clock_time = dispatcher_event_anchor_time(self._dispatcher, self._clock)
        with self._state_lock:
            previous_target_time = self._master_target_time
            self._master_index = new_index
            self._master_target_time = target_time
            self._master_next_target_time = next_target_time
            self._master_anchor_clock_time = anchor_clock_time
            self._tempo_ratio = max(self._MIN_TEMPO_RATIO, float(tempo_ratio))
            self._transport_paused = False
            if previous_target_time is None:
                self._seek_request_time = target_time
            elif abs(target_time - previous_target_time) > self._SEEK_TIME_THRESHOLD:
                self._seek_request_time = target_time
            elif self._last_source_time is not None and target_time < (
                self._last_source_time - self._SEEK_TIME_THRESHOLD
            ):
                self._seek_request_time = target_time

    def _initialize_channel(self) -> None:
        with self._output_lock:
            if self._program is not None:
                self._output.write_short(0xC0 | self._channel, self._program, 0)

    def _load_note_events(self) -> list[HandPracticeNoteEvent]:
        return load_hand_practice_note_events(self._midi_path)

    def _play_loop(self) -> None:
        while not self._stop_event.is_set():
            self._flush_due_note_offs()
            seek_target = None
            with self._state_lock:
                if self._seek_request_time is not None:
                    seek_target = self._seek_request_time
                    self._seek_request_time = None

            if seek_target is not None:
                self._perform_seek(seek_target)
                continue

            with self._state_lock:
                paused = self._transport_paused
                playhead_time = self._projected_master_time_locked()
                tempo_ratio = self._tempo_ratio
                event_index = self._event_index

            if paused or playhead_time is None or event_index >= len(self._events):
                self._sleep_tick()
                continue

            event = self._events[event_index]
            if event.source_time > playhead_time:
                self._sleep_tick()
                continue

            self._emit_event(event, tempo_ratio)
            with self._state_lock:
                self._event_index = event_index + 1
                self._last_source_time = event.source_time

    def _perform_seek(self, target_time: float) -> None:
        self.panic()
        self._initialize_channel()
        with self._state_lock:
            self._event_index = int(np.searchsorted(self._source_times, target_time, side="left"))
            self._last_source_time = float(target_time)

    def _emit_event(self, event: HandPracticeNoteEvent, tempo_ratio: float) -> None:
        note = int(max(SOUND_START, min(SOUND_END, event.note)))
        normalized_velocity = max(108, int(event.velocity))
        velocity = int(
            np.clip(
                round(127 * apply_orchestra_mix_level(normalized_velocity / 127.0, self._volume_scale)),
                1,
                127,
            )
        )
        due_time = self._clock() + (
            max(0.05, event.duration) / max(self._MIN_TEMPO_RATIO, float(tempo_ratio))
        )
        with self._state_lock:
            previous_generation = self._active_note_generations.get(note)
            generation = self._note_generation_counter
            self._note_generation_counter += 1
            self._active_note_generations[note] = generation
            heapq.heappush(
                self._pending_note_offs,
                HandPracticeScheduledNoteOff(
                    due_time=due_time,
                    order=self._note_off_counter,
                    note=note,
                    generation=generation,
                ),
            )
            self._note_off_counter += 1
            if previous_generation is not None:
                self._send_note_off(note)
            with self._output_lock:
                self._output.write_short(0x90 | self._channel, note, velocity)

    def _flush_due_note_offs(self) -> None:
        now = self._clock()
        due_notes: list[int] = []
        with self._state_lock:
            while self._pending_note_offs and self._pending_note_offs[0].due_time <= now:
                scheduled = heapq.heappop(self._pending_note_offs)
                if self._active_note_generations.get(scheduled.note) != scheduled.generation:
                    continue
                self._active_note_generations.pop(scheduled.note, None)
                due_notes.append(scheduled.note)

        for note in due_notes:
            self._send_note_off(note)

    def _send_note_off(self, note: int) -> None:
        with self._output_lock:
            self._output.write_short(0x80 | self._channel, int(note), 0)

    def _sleep_tick(self) -> None:
        self._stop_event.wait(self._WAIT_GRANULARITY)

    def _score_index_to_target_time(self, score_index: int) -> float:
        tempo_tracker = self._dispatcher.tempo_tracker
        position = int(tempo_tracker.index_to_position[int(score_index)])
        return float(tempo_tracker.nominal_onsets[position])

    def _score_index_to_next_target_time(self, score_index: int) -> float | None:
        tempo_tracker = self._dispatcher.tempo_tracker
        position = int(tempo_tracker.index_to_position[int(score_index)])
        next_position = position + 1
        if next_position >= len(tempo_tracker.nominal_onsets):
            return None
        return float(tempo_tracker.nominal_onsets[next_position])

    def _projected_master_time_locked(self) -> float | None:
        if self._master_target_time is None:
            return None
        target_time = float(self._master_target_time)
        anchor_time = self._master_anchor_clock_time
        if anchor_time is None:
            return target_time

        projected = target_time + (
            max(0.0, self._clock() - anchor_time)
            * max(self._MIN_TEMPO_RATIO, self._tempo_ratio)
        )
        # Continue moving on the accompaniment MIDI timeline while waiting for
        # the next solo dispatch so sustained notes are not truncated.
        return max(target_time, projected)


class LocalHandPracticeAccompaniment:
    """Local-sample accompaniment for two-hand practice mode."""

    _MIN_TEMPO_RATIO = 0.25
    _WAIT_GRANULARITY = 0.002
    _SEEK_TIME_THRESHOLD = 0.75

    def __init__(
        self,
        midi_path: str | Path,
        dispatcher: ScoreEventDispatcher,
        *,
        note_on_callback: Callable[[int, int], None],
    ) -> None:
        self._midi_path = Path(midi_path)
        self._dispatcher = dispatcher
        self._note_on_callback = note_on_callback
        self._events = load_hand_practice_note_events(self._midi_path)
        self._source_times = np.asarray(
            [event.source_time for event in self._events],
            dtype=np.float64,
        )
        self._event_index = 0
        self._master_target_time: float | None = None
        self._master_next_target_time: float | None = None
        self._master_anchor_clock_time: float | None = None
        self._tempo_ratio = 1.0
        self._transport_paused = True
        self._seek_request_time: float | None = None
        self._last_source_time: float | None = None
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.status_label = "Local piano samples"

        self._dispatcher.subscribe(self.handle_dispatch)

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def source_pitches(self) -> list[int]:
        return sorted({int(event.note) for event in self._events})

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._play_loop,
            name="LocalHandPracticeAccompaniment",
            daemon=True,
        )
        self._thread.start()

    def close(self, timeout: float = 1.0) -> None:
        self._dispatcher.unsubscribe(self.handle_dispatch)
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def resume(self) -> None:
        with self._state_lock:
            self._transport_paused = False

    def halt(self) -> None:
        with self._state_lock:
            self._transport_paused = True
            self._master_target_time = None
            self._master_next_target_time = None
            self._master_anchor_clock_time = None
            self._seek_request_time = None
            self._event_index = 0
            self._last_source_time = None

    def panic(self) -> None:
        return

    def reset_to_start(self) -> None:
        self.seek(0.0)

    def seek(self, target_time: float) -> None:
        with self._state_lock:
            self._seek_request_time = max(0.0, float(target_time))
            self._transport_paused = False

    def handle_dispatch(self, index: int, tempo_ratio: float) -> None:
        target_time = self._score_index_to_target_time(int(index))
        next_target_time = self._score_index_to_next_target_time(int(index))
        anchor_clock_time = dispatcher_event_anchor_time(self._dispatcher, time.monotonic)
        with self._state_lock:
            previous_target_time = self._master_target_time
            self._master_target_time = target_time
            self._master_next_target_time = next_target_time
            self._master_anchor_clock_time = anchor_clock_time
            self._tempo_ratio = max(self._MIN_TEMPO_RATIO, float(tempo_ratio))
            self._transport_paused = False
            if previous_target_time is None:
                self._seek_request_time = target_time
            elif abs(target_time - previous_target_time) > self._SEEK_TIME_THRESHOLD:
                self._seek_request_time = target_time
            elif self._last_source_time is not None and target_time < (
                self._last_source_time - self._SEEK_TIME_THRESHOLD
            ):
                self._seek_request_time = target_time

    def _play_loop(self) -> None:
        while not self._stop_event.is_set():
            seek_target = None
            with self._state_lock:
                if self._seek_request_time is not None:
                    seek_target = self._seek_request_time
                    self._seek_request_time = None

            if seek_target is not None:
                self._perform_seek(seek_target)
                continue

            with self._state_lock:
                paused = self._transport_paused
                playhead_time = self._projected_master_time_locked()
                event_index = self._event_index

            if paused or playhead_time is None or event_index >= len(self._events):
                self._sleep_tick()
                continue

            event = self._events[event_index]
            if event.source_time > playhead_time:
                self._sleep_tick()
                continue

            self._note_on_callback(int(event.note), int(max(1, min(127, event.velocity))))
            with self._state_lock:
                self._event_index = event_index + 1
                self._last_source_time = event.source_time

    def _perform_seek(self, target_time: float) -> None:
        with self._state_lock:
            self._event_index = int(np.searchsorted(self._source_times, target_time, side="left"))
            self._last_source_time = float(target_time)

    def _sleep_tick(self) -> None:
        self._stop_event.wait(self._WAIT_GRANULARITY)

    def _score_index_to_target_time(self, score_index: int) -> float:
        tempo_tracker = self._dispatcher.tempo_tracker
        position = int(tempo_tracker.index_to_position[int(score_index)])
        return float(tempo_tracker.nominal_onsets[position])

    def _score_index_to_next_target_time(self, score_index: int) -> float | None:
        tempo_tracker = self._dispatcher.tempo_tracker
        position = int(tempo_tracker.index_to_position[int(score_index)])
        next_position = position + 1
        if next_position >= len(tempo_tracker.nominal_onsets):
            return None
        return float(tempo_tracker.nominal_onsets[next_position])

    def _projected_master_time_locked(self) -> float | None:
        if self._master_target_time is None:
            return None
        target_time = float(self._master_target_time)
        anchor_time = self._master_anchor_clock_time
        if anchor_time is None:
            return target_time

        projected = target_time + (
            max(0.0, time.monotonic() - anchor_time)
            * max(self._MIN_TEMPO_RATIO, self._tempo_ratio)
        )
        # Keep local accompaniment progressing by MIDI time between solo events,
        # so note tails and phrase continuations are preserved on pauses.
        return max(target_time, projected)


LOCAL_ORCHESTRA_LENGTH_SECONDS = {
    "025": 0.25,
    "05": 0.5,
    "1": 1.0,
    "15": 1.5,
    "long": 3.0,
    "very-long": 5.0,
    "phrase": 1.5,
}
LOCAL_ORCHESTRA_DYNAMIC_ORDER = {
    "pianissimo": 0,
    "piano": 1,
    "mezzo-piano": 2,
    "mezzo-forte": 3,
    "forte": 4,
    "fortissimo": 5,
}
LOCAL_ORCHESTRA_GM_PROGRAMS = {
    47: "bass drum",
    48: "violin",
    49: "violin",
    50: "violin",
    56: "trumpet",
    57: "trombone",
    58: "tuba",
    60: "french horn",
    68: "oboe",
    70: "bassoon",
    71: "clarinet",
    73: "flute",
}
LOCAL_ORCHESTRA_TRACK_HINTS = (
    ("violoncell", "cello"),
    ("cello", "cello"),
    ("contrabb", "double bass"),
    ("double bass", "double bass"),
    ("bassi", "double bass"),
    ("viole", "viola"),
    ("viola", "viola"),
    ("violini", "violin"),
    ("violin", "violin"),
    ("flauti", "flute"),
    ("flute", "flute"),
    ("oboi", "oboe"),
    ("oboe", "oboe"),
    ("clarinet", "clarinet"),
    ("fagotti", "bassoon"),
    ("bassoon", "bassoon"),
    ("corni", "french horn"),
    ("horn", "french horn"),
    ("trombe", "trumpet"),
    ("trumpet", "trumpet"),
    ("tromboni", "trombone"),
    ("trombone", "trombone"),
    ("tuba", "tuba"),
    ("timpani", "bass drum"),
)
LOCAL_SAMPLE_NOTE_RE = re.compile(r"^[A-G]s?-?\d+$")


@dataclass(frozen=True)
class LocalOrchestraSample:
    zip_path: Path
    member_name: str
    instrument: str
    pitch: int | None
    length_code: str
    dynamic: str
    articulation: str


@dataclass(frozen=True)
class LocalOrchestraNoteEvent:
    source_time: float
    note: int
    velocity: int
    duration: float
    instrument: str


def normalize_local_instrument_name(name: str) -> str:
    normalized = name.lower().replace("-", " ").replace("_", " ").strip()
    return " ".join(normalized.split())


def local_sample_note_to_midi(note_name: str) -> int:
    match = re.fullmatch(r"([A-G]s?)(-?\d+)", note_name)
    if match is None:
        raise ValueError(f"Unsupported local orchestra sample note: {note_name}")
    note, octave_text = match.groups()
    semitone = {
        "C": 0,
        "Cs": 1,
        "D": 2,
        "Ds": 3,
        "E": 4,
        "F": 5,
        "Fs": 6,
        "G": 7,
        "Gs": 8,
        "A": 9,
        "As": 10,
        "B": 11,
    }[note]
    return ((int(octave_text) + 1) * 12) + semitone


def infer_local_orchestra_instrument(track_name: str, program: int | None, midi_pitch: int) -> str:
    normalized_track = normalize_local_instrument_name(track_name)
    for needle, instrument in LOCAL_ORCHESTRA_TRACK_HINTS:
        if needle in normalized_track:
            return instrument

    if program is not None and int(program) in LOCAL_ORCHESTRA_GM_PROGRAMS:
        return LOCAL_ORCHESTRA_GM_PROGRAMS[int(program)]

    if midi_pitch < 40:
        return "double bass"
    if midi_pitch < 56:
        return "cello"
    if midi_pitch < 67:
        return "viola"
    return "violin"


class LocalOrchestraSampleBank:
    """Lazy local sample bank for Philharmonia-style orchestra archives."""

    def __init__(
        self,
        sample_root: Path = ORCHESTRA_SAMPLE_DIR,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self.sample_root = sample_root
        self.cache_dir = cache_dir or (sample_root / "cache")
        self._zip_files: dict[Path, zipfile.ZipFile] = {}
        self._sample_index: dict[str, list[LocalOrchestraSample]] = {}
        self._sound_cache: dict[str, pygame.mixer.Sound] = {}
        self._lock = threading.RLock()
        self._build_index()

    @property
    def instrument_names(self) -> set[str]:
        return set(self._sample_index)

    def get_sound(
        self,
        instrument: str,
        midi_pitch: int,
        velocity: int,
        *,
        duration: float,
    ) -> pygame.mixer.Sound:
        instrument_key = normalize_local_instrument_name(instrument)
        with self._lock:
            candidates = self._sample_index.get(instrument_key)
            if not candidates:
                raise RuntimeError(f"No local orchestra samples for instrument: {instrument_key}")

            target_pitch = int(midi_pitch)
            target_duration = float(max(0.08, min(5.0, duration)))
            velocity_value = int(max(1, min(127, velocity)))
            sample = min(
                candidates,
                key=lambda candidate: self._sample_score(
                    candidate,
                    target_pitch,
                    velocity_value,
                    target_duration,
                ),
            )
            cache_key = f"{sample.zip_path}:{sample.member_name}"
            cached = self._sound_cache.get(cache_key)
            if cached is not None:
                return cached

            extracted_path = self._extract_sample(sample)
            sound = pygame.mixer.Sound(str(extracted_path))
            self._sound_cache[cache_key] = sound
            return sound

    def close(self) -> None:
        for zip_file in self._zip_files.values():
            zip_file.close()
        self._zip_files.clear()

    def _build_index(self) -> None:
        for zip_path in sorted(self.sample_root.glob("*.zip")):
            zip_file = zipfile.ZipFile(zip_path)
            self._zip_files[zip_path] = zip_file
            for member_name in zip_file.namelist():
                sample = self._parse_member(zip_path, member_name)
                if sample is None:
                    continue
                self._sample_index.setdefault(sample.instrument, []).append(sample)

        if not self._sample_index:
            raise RuntimeError(f"No local orchestra samples found in: {self.sample_root}")

    def _parse_member(self, zip_path: Path, member_name: str) -> LocalOrchestraSample | None:
        if not member_name.lower().endswith(".mp3"):
            return None

        parts = member_name.split("/")
        if len(parts) < 3:
            return None

        instrument = normalize_local_instrument_name(parts[1])
        stem = Path(member_name).stem
        tokens = stem.split("_")
        length_index = next(
            (
                index
                for index, token in enumerate(tokens)
                if token in LOCAL_ORCHESTRA_LENGTH_SECONDS
            ),
            None,
        )
        if length_index is None:
            return None

        note_token = tokens[length_index - 1] if length_index > 0 else ""
        sample_pitch = (
            local_sample_note_to_midi(note_token)
            if LOCAL_SAMPLE_NOTE_RE.match(note_token)
            else None
        )
        dynamic = tokens[length_index + 1] if length_index + 1 < len(tokens) else "mezzo-forte"
        articulation = tokens[length_index + 2] if length_index + 2 < len(tokens) else "normal"
        return LocalOrchestraSample(
            zip_path=zip_path,
            member_name=member_name,
            instrument=instrument,
            pitch=sample_pitch,
            length_code=tokens[length_index],
            dynamic=dynamic,
            articulation=articulation,
        )

    def _extract_sample(self, sample: LocalOrchestraSample) -> Path:
        zip_file = self._zip_files[sample.zip_path]
        destination = self.cache_dir / sample.zip_path.stem / sample.member_name
        if destination.exists():
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        with zip_file.open(sample.member_name) as source, destination.open("wb") as target:
            target.write(source.read())
        return destination

    def _sample_score(
        self,
        sample: LocalOrchestraSample,
        target_pitch: int,
        velocity: int,
        target_duration: float,
    ) -> tuple[float, float, float, int]:
        pitch_distance = 0.0 if sample.pitch is None else abs(float(sample.pitch - target_pitch))
        dynamic_distance = abs(self._dynamic_rank(sample.dynamic) - self._velocity_rank(velocity))
        sample_length = LOCAL_ORCHESTRA_LENGTH_SECONDS.get(sample.length_code, 1.0)
        undershoot_penalty = max(0.0, target_duration - sample_length) * 2.0
        length_distance = abs(sample_length - target_duration) + undershoot_penalty
        articulation_penalty = 0
        if target_duration >= 0.45 and "stacc" in sample.articulation:
            articulation_penalty += 4
        if target_duration < 0.35 and "stacc" not in sample.articulation and sample.length_code not in {"025", "05"}:
            articulation_penalty += 2
        return (pitch_distance, length_distance, float(dynamic_distance), articulation_penalty)

    @staticmethod
    def _dynamic_rank(dynamic: str) -> int:
        return LOCAL_ORCHESTRA_DYNAMIC_ORDER.get(dynamic, LOCAL_ORCHESTRA_DYNAMIC_ORDER["mezzo-forte"])

    @staticmethod
    def _velocity_rank(velocity: int) -> int:
        if velocity >= 112:
            return LOCAL_ORCHESTRA_DYNAMIC_ORDER["fortissimo"]
        if velocity >= 88:
            return LOCAL_ORCHESTRA_DYNAMIC_ORDER["forte"]
        if velocity >= 64:
            return LOCAL_ORCHESTRA_DYNAMIC_ORDER["mezzo-forte"]
        if velocity >= 42:
            return LOCAL_ORCHESTRA_DYNAMIC_ORDER["piano"]
        return LOCAL_ORCHESTRA_DYNAMIC_ORDER["pianissimo"]


def build_midi_tick_to_second(midi_file: mido.MidiFile) -> Callable[[int], float]:
    tempo_events: list[tuple[int, int]] = [(0, DEFAULT_MIDI_TEMPO)]
    for track in midi_file.tracks:
        abs_tick = 0
        for message in track:
            abs_tick += int(getattr(message, "time", 0))
            if getattr(message, "type", None) == "set_tempo":
                tempo_events.append((abs_tick, int(message.tempo)))

    tempo_events.sort(key=lambda item: item[0])
    compact_events: list[tuple[int, int]] = []
    for tick, tempo in tempo_events:
        if compact_events and compact_events[-1][0] == tick:
            compact_events[-1] = (tick, tempo)
        else:
            compact_events.append((tick, tempo))

    cache: dict[int, float] = {}

    def tick_to_second(tick: int) -> float:
        abs_tick = int(max(0, tick))
        cached = cache.get(abs_tick)
        if cached is not None:
            return cached

        elapsed = 0.0
        previous_tick = 0
        active_tempo = DEFAULT_MIDI_TEMPO
        for tempo_tick, tempo in compact_events:
            if tempo_tick <= 0:
                active_tempo = int(tempo)
                continue
            if abs_tick <= tempo_tick:
                break
            elapsed += mido.tick2second(tempo_tick - previous_tick, midi_file.ticks_per_beat, active_tempo)
            previous_tick = tempo_tick
            active_tempo = int(tempo)
        elapsed += mido.tick2second(abs_tick - previous_tick, midi_file.ticks_per_beat, active_tempo)
        cache[abs_tick] = float(elapsed)
        return float(elapsed)

    return tick_to_second


def load_local_orchestra_note_events(midi_path: str | Path) -> list[LocalOrchestraNoteEvent]:
    midi_file = mido.MidiFile(midi_path)
    tick_to_second = build_midi_tick_to_second(midi_file)
    events: list[LocalOrchestraNoteEvent] = []

    for track in midi_file.tracks:
        abs_tick = 0
        track_name = ""
        program_by_channel: dict[int, int] = {}
        open_notes: dict[tuple[int, int], list[tuple[int, int, str]]] = {}

        for message in track:
            abs_tick += int(getattr(message, "time", 0))
            message_type = getattr(message, "type", None)
            if message_type == "track_name":
                track_name = str(getattr(message, "name", ""))
                continue
            if getattr(message, "is_meta", False):
                continue

            channel = getattr(message, "channel", None)
            if channel is None:
                continue
            channel = int(channel)

            if message_type == "program_change":
                program_by_channel[channel] = int(getattr(message, "program", 0))
                continue

            if message_type == "note_on" and int(getattr(message, "velocity", 0)) > 0:
                note = int(getattr(message, "note", 0))
                instrument = infer_local_orchestra_instrument(
                    track_name,
                    program_by_channel.get(channel),
                    note,
                )
                open_notes.setdefault((channel, note), []).append(
                    (abs_tick, int(getattr(message, "velocity", 0)), instrument)
                )
                continue

            if message_type == "note_off" or (
                message_type == "note_on" and int(getattr(message, "velocity", 0)) <= 0
            ):
                note = int(getattr(message, "note", 0))
                note_stack = open_notes.get((channel, note))
                if not note_stack:
                    continue
                start_tick, velocity, instrument = note_stack.pop()
                if not note_stack:
                    open_notes.pop((channel, note), None)
                start_time = tick_to_second(start_tick)
                end_time = tick_to_second(abs_tick)
                events.append(
                    LocalOrchestraNoteEvent(
                        source_time=start_time,
                        note=note,
                        velocity=max(1, min(127, velocity)),
                        duration=max(0.05, end_time - start_time),
                        instrument=instrument,
                    )
                )

        track_end_time = tick_to_second(abs_tick)
        for (_channel, note), note_stack in open_notes.items():
            for start_tick, velocity, instrument in note_stack:
                start_time = tick_to_second(start_tick)
                events.append(
                    LocalOrchestraNoteEvent(
                        source_time=start_time,
                        note=int(note),
                        velocity=max(1, min(127, velocity)),
                        duration=max(0.05, track_end_time - start_time),
                        instrument=instrument,
                    )
                )

    events.sort(key=lambda event: (event.source_time, event.instrument, event.note))
    return events


class LocalOrchestraAccompaniment:
    """Local-sample orchestra player driven by the same dispatcher timeline as MIDI orchestra."""

    _MIN_TEMPO_RATIO = 0.25
    _WAIT_GRANULARITY = 0.002
    _SEEK_TIME_THRESHOLD = 1.0

    def __init__(
        self,
        midi_path: str | Path,
        dispatcher: ScoreEventDispatcher,
        *,
        volume_scale: float = ORCHESTRA_VOLUME_DEFAULT,
        channel_start: int = 64,
        channel_count: int = 64,
    ) -> None:
        self._midi_path = Path(midi_path)
        self._dispatcher = dispatcher
        self._volume_scale = clamp_orchestra_mix(volume_scale)
        self._sample_bank = LocalOrchestraSampleBank()
        self._events = load_local_orchestra_note_events(self._midi_path)
        if not self._events:
            raise RuntimeError(f"No local orchestra note events found in: {self._midi_path}")

        self._source_times = np.asarray([event.source_time for event in self._events], dtype=np.float64)
        self._event_index = 0
        self._master_target_time: float | None = None
        self._master_next_target_time: float | None = None
        self._master_anchor_clock_time: float | None = None
        self._tempo_ratio = 1.0
        self._transport_paused = True
        self._seek_request_time: float | None = None
        self._last_source_time: float | None = None
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._channel_indices = list(range(channel_start, channel_start + channel_count))
        self._channel_cursor = 0
        self._fallback_sound_cache: dict[tuple[int, int], pygame.mixer.Sound] = {}
        self.status_label = "Local orchestra samples"

        self._dispatcher.subscribe(self.handle_dispatch)

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def source_pitches(self) -> list[int]:
        return sorted({int(event.note) for event in self._events})

    def set_volume_scale(self, volume_scale: float) -> None:
        with self._state_lock:
            self._volume_scale = clamp_orchestra_mix(volume_scale)

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._play_loop,
            name="LocalOrchestraAccompaniment",
            daemon=True,
        )
        self._thread.start()

    def close(self, timeout: float = 1.0) -> None:
        self._dispatcher.unsubscribe(self.handle_dispatch)
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        self.panic()
        self._sample_bank.close()

    def resume(self) -> None:
        with self._state_lock:
            self._transport_paused = False

    def halt(self) -> None:
        with self._state_lock:
            self._transport_paused = True
            self._master_target_time = None
            self._master_next_target_time = None
            self._master_anchor_clock_time = None
            self._seek_request_time = None
            self._event_index = 0
            self._last_source_time = None
        self.panic()

    def panic(self) -> None:
        for channel_index in self._channel_indices:
            pygame.mixer.Channel(channel_index).stop()

    def reset_to_start(self) -> None:
        self.seek(0.0)

    def seek(self, target_time: float) -> None:
        with self._state_lock:
            self._seek_request_time = max(0.0, float(target_time))
            self._transport_paused = False

    def handle_dispatch(self, index: int, tempo_ratio: float) -> None:
        target_time = self._score_index_to_target_time(int(index))
        next_target_time = self._score_index_to_next_target_time(int(index))
        anchor_clock_time = dispatcher_event_anchor_time(self._dispatcher, time.monotonic)
        with self._state_lock:
            previous_target_time = self._master_target_time
            self._master_target_time = target_time
            self._master_next_target_time = next_target_time
            self._master_anchor_clock_time = anchor_clock_time
            self._tempo_ratio = max(self._MIN_TEMPO_RATIO, float(tempo_ratio))
            self._transport_paused = False
            if previous_target_time is None:
                self._seek_request_time = target_time
            elif abs(target_time - previous_target_time) > self._SEEK_TIME_THRESHOLD:
                self._seek_request_time = target_time
            elif self._last_source_time is not None and target_time < (
                self._last_source_time - self._SEEK_TIME_THRESHOLD
            ):
                self._seek_request_time = target_time

    def _play_loop(self) -> None:
        while not self._stop_event.is_set():
            seek_target = None
            with self._state_lock:
                if self._seek_request_time is not None:
                    seek_target = self._seek_request_time
                    self._seek_request_time = None

            if seek_target is not None:
                self._perform_seek(seek_target)
                continue

            with self._state_lock:
                paused = self._transport_paused
                playhead_time = self._projected_master_time_locked()
                tempo_ratio = self._tempo_ratio
                event_index = self._event_index

            if paused or playhead_time is None or event_index >= len(self._events):
                self._sleep_tick()
                continue

            event = self._events[event_index]
            if event.source_time > playhead_time:
                self._sleep_tick()
                continue

            self._emit_event(event, tempo_ratio)
            with self._state_lock:
                self._event_index = event_index + 1
                self._last_source_time = event.source_time

    def _perform_seek(self, target_time: float) -> None:
        self.panic()
        with self._state_lock:
            self._event_index = int(np.searchsorted(self._source_times, target_time, side="left"))
            self._last_source_time = float(target_time)

    def _emit_event(self, event: LocalOrchestraNoteEvent, tempo_ratio: float) -> None:
        scaled_duration = max(0.05, float(event.duration) / max(self._MIN_TEMPO_RATIO, float(tempo_ratio)))
        try:
            sound = self._sample_bank.get_sound(
                event.instrument,
                event.note,
                event.velocity,
                duration=scaled_duration,
            )
        except RuntimeError:
            sound = self._fallback_sound(event.note, scaled_duration)

        channel = pygame.mixer.Channel(self._channel_indices[self._channel_cursor])
        self._channel_cursor = (self._channel_cursor + 1) % len(self._channel_indices)
        velocity_scale = max(0.05, min(1.0, float(event.velocity) / 127.0))
        channel.set_volume(apply_orchestra_mix_level(velocity_scale, self._volume_scale))
        sample_length = max(0.05, float(sound.get_length()))
        loops = max(0, int(scaled_duration // sample_length))
        channel.play(sound, loops=loops, maxtime=max(1, int(scaled_duration * 1000)))

    def _fallback_sound(self, midi_pitch: int, duration: float) -> pygame.mixer.Sound:
        duration_bucket = int(round(max(0.35, min(1.6, duration)) * 10))
        key = (int(midi_pitch), duration_bucket)
        cached = self._fallback_sound_cache.get(key)
        if cached is not None:
            return cached
        sound = make_piano_sound(int(midi_pitch), duration=duration_bucket / 10.0)
        self._fallback_sound_cache[key] = sound
        return sound

    def _sleep_tick(self) -> None:
        self._stop_event.wait(self._WAIT_GRANULARITY)

    def _score_index_to_target_time(self, score_index: int) -> float:
        tempo_tracker = self._dispatcher.tempo_tracker
        position = int(tempo_tracker.index_to_position[int(score_index)])
        return float(tempo_tracker.nominal_onsets[position])

    def _score_index_to_next_target_time(self, score_index: int) -> float | None:
        tempo_tracker = self._dispatcher.tempo_tracker
        position = int(tempo_tracker.index_to_position[int(score_index)])
        next_position = position + 1
        if next_position >= len(tempo_tracker.nominal_onsets):
            return None
        return float(tempo_tracker.nominal_onsets[next_position])

    def _projected_master_time_locked(self) -> float | None:
        if self._master_target_time is None:
            return None
        target_time = float(self._master_target_time)
        anchor_time = self._master_anchor_clock_time
        if anchor_time is None:
            return target_time
        projected = target_time + (
            max(0.0, time.monotonic() - anchor_time)
            * max(self._MIN_TEMPO_RATIO, self._tempo_ratio)
        )
        return max(target_time, projected)


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    position: tuple[int, int],
) -> None:
    rendered = font.render(text, True, color)
    surface.blit(rendered, position)


def fit_text(font: pygame.font.Font, text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if font.size(text)[0] <= max_width:
        return text

    ellipsis = "..."
    if font.size(ellipsis)[0] > max_width:
        return ""

    trimmed = text.rstrip()
    while trimmed and font.size(trimmed + ellipsis)[0] > max_width:
        trimmed = trimmed[:-1].rstrip(" ,;:/|-")
    return (trimmed + ellipsis) if trimmed else ellipsis


def draw_fitted_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    position: tuple[int, int],
    max_width: int,
) -> None:
    draw_text(surface, font, fit_text(font, text, max_width), color, position)


def draw_card(surface: pygame.Surface, rect: pygame.Rect, color: tuple[int, int, int]) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=22)


def draw_button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    font: pygame.font.Font,
    *,
    active: bool,
    hovered: bool,
) -> None:
    if active and hovered:
        fill = BUTTON_ACTIVE_HOVER
    elif active:
        fill = BUTTON_ACTIVE
    elif hovered:
        fill = BUTTON_HOVER
    else:
        fill = BUTTON_IDLE

    text_color = (245, 248, 252) if active else TEXT_COLOR
    pygame.draw.rect(surface, fill, rect, border_radius=14)
    pygame.draw.rect(surface, (18, 24, 33), rect, width=2, border_radius=14)
    rendered = font.render(text, True, text_color)
    label_rect = rendered.get_rect(center=rect.center)
    surface.blit(rendered, label_rect)


def draw_input_box(
    surface: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    font: pygame.font.Font,
    *,
    active: bool,
    hovered: bool,
) -> None:
    if active:
        fill = (255, 255, 255)
        border = ACCENT
    elif hovered:
        fill = BUTTON_HOVER
        border = (18, 24, 33)
    else:
        fill = BUTTON_IDLE
        border = (18, 24, 33)

    pygame.draw.rect(surface, fill, rect, border_radius=12)
    pygame.draw.rect(surface, border, rect, width=2, border_radius=12)

    rendered = font.render(text, True, TEXT_COLOR)
    label_rect = rendered.get_rect(midleft=(rect.x + 12, rect.centery))
    surface.blit(rendered, label_rect)


def score_note_pitches(note: dict[str, object]) -> list[int]:
    raw_pitches = note.get("pitches")
    if raw_pitches is None:
        raw_pitch = note.get("pitch")
        if raw_pitch is None:
            raise ValueError("score note is missing 'pitch'/'pitches'")
        return [int(raw_pitch)]

    if not isinstance(raw_pitches, list) or not raw_pitches:
        raise ValueError("score note 'pitches' must be a non-empty list")
    return sorted({int(pitch) for pitch in raw_pitches})


class SequentialPracticeFollower:
    """Local-only follower for one-hand practice parts with repeated patterns."""

    def __init__(self, score_json: str | Path, *, lookahead: int = 4) -> None:
        score_path = Path(score_json)
        self.score_data = json.loads(score_path.read_text(encoding="utf-8"))
        self.notes = list(self.score_data.get("notes", []))
        if not self.notes:
            raise ValueError(f"Practice score is empty: {score_path}")

        self.N = len(self.notes)
        self.current_index = 0
        self.lookahead = max(1, int(lookahead))
        self.confidence = 1.0
        self.last_selected_model = "study"
        self._note_pitch_sets = [set(score_note_pitches(note)) for note in self.notes]
        self._last_progress_timestamp: float | None = None
        self._repeat_advance_seconds = 0.10

    @property
    def mode_label(self) -> str:
        return "Study Local"

    def reset_to_start(self) -> int:
        self.current_index = 0
        self.confidence = 1.0
        self.last_selected_model = "study"
        self._last_progress_timestamp = None
        return self.current_index

    def process_event(self, pitches: Any, timestamp: float) -> int:
        event_time = float(timestamp)
        observed = set(score_note_pitches({"pitches": list(np.atleast_1d(pitches))}))
        if not observed:
            return self.current_index

        start = int(self.current_index)
        if start + 1 < self.N:
            current_target = self._note_pitch_sets[start]
            next_target = self._note_pitch_sets[start + 1]
            elapsed_since_progress = (
                float("inf")
                if self._last_progress_timestamp is None
                else event_time - self._last_progress_timestamp
            )
            if (
                current_target
                and current_target == next_target
                and bool(observed & current_target)
                and elapsed_since_progress >= self._repeat_advance_seconds
            ):
                self.current_index = start + 1
                self.confidence = min(1.0, len(observed & next_target) / max(1, len(observed)))
                self._last_progress_timestamp = event_time
                return self.current_index

        stop = min(self.N, start + self.lookahead + 1)
        best_index = start
        best_score = -1.0

        for index in range(start, stop):
            target = self._note_pitch_sets[index]
            if not target:
                continue

            overlap = len(observed & target)
            if overlap <= 0:
                score = 0.0
            else:
                # Jaccard handles both single live keydowns and full autoplay chords.
                score = overlap / max(1, len(observed | target))

            distance = index - start
            if score > best_score + 1e-9 or (
                abs(score - best_score) <= 1e-9 and distance < (best_index - start)
            ):
                best_score = score
                best_index = index

        if best_score > 0.0:
            previous_index = self.current_index
            self.current_index = max(self.current_index, int(best_index))
            current_target = self._note_pitch_sets[self.current_index]
            self.confidence = min(1.0, len(observed & current_target) / max(1, len(observed)))
            if self._last_progress_timestamp is None or self.current_index != previous_index:
                self._last_progress_timestamp = event_time
        else:
            self.confidence = 0.0

        return self.current_index


def representative_score_pitch(note: dict[str, object]) -> int:
    return max(score_note_pitches(note))


def format_chord_label(pitches: list[int]) -> str:
    return ", ".join(f"{pitch_to_note_name(pitch)} ({pitch})" for pitch in pitches)


def build_autoplay_events(score_notes: list[dict[str, object]]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    previous_onset: float | None = None
    onset_cursor = 0.0

    for score_position, note in enumerate(score_notes):
        duration = max(0.0, float(note.get("nominal_duration", 0.25)))
        onset = float(note.get("nominal_onset", onset_cursor))
        chord_pitches = score_note_pitches(note)
        if previous_onset is None:
            first_delay = max(0.14, onset)
        else:
            first_delay = max(MIN_AUTOPLAY_GAP, onset - previous_onset)

        events.append(
            {
                "pitches": [int(pitch) for pitch in chord_pitches],
                "delay": first_delay,
                "nominal_duration": duration,
                "nominal_onset": onset,
                "score_position": score_position,
                "chord_size": len(chord_pitches),
            }
        )
        previous_onset = onset
        onset_cursor = onset + duration

    return events


def build_autoplay_events_from_note_events(
    note_events: list[HandPracticeNoteEvent],
    *,
    chord_epsilon: float = 0.03,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    if not note_events:
        return events

    grouped: list[list[HandPracticeNoteEvent]] = []
    current_group: list[HandPracticeNoteEvent] = []
    current_onset = 0.0
    for note_event in sorted(note_events, key=lambda event: (event.source_time, event.note)):
        if not current_group:
            current_group = [note_event]
            current_onset = float(note_event.source_time)
            continue
        if float(note_event.source_time) - current_onset <= chord_epsilon:
            current_group.append(note_event)
            continue
        grouped.append(current_group)
        current_group = [note_event]
        current_onset = float(note_event.source_time)
    if current_group:
        grouped.append(current_group)

    previous_onset: float | None = None
    for score_position, group in enumerate(grouped):
        onset = min(float(event.source_time) for event in group)
        if previous_onset is None:
            delay = max(0.14, onset)
        else:
            delay = max(MIN_AUTOPLAY_GAP, onset - previous_onset)

        pitches = sorted({int(event.note) for event in group})
        durations_by_pitch: dict[int, float] = {}
        velocities_by_pitch: dict[int, int] = {}
        for event in group:
            pitch = int(event.note)
            durations_by_pitch[pitch] = max(durations_by_pitch.get(pitch, 0.0), float(event.duration))
            velocities_by_pitch[pitch] = max(velocities_by_pitch.get(pitch, 0), int(event.velocity))

        events.append(
            {
                "pitches": pitches,
                "delay": delay,
                "nominal_duration": max(durations_by_pitch.values(), default=0.25),
                "durations": durations_by_pitch,
                "velocities": velocities_by_pitch,
                "nominal_onset": onset,
                "score_position": score_position,
                "chord_size": len(pitches),
            }
        )
        previous_onset = onset

    return events


def autoplay_event_start_index(
    autoplay_events: list[dict[str, object]],
    score_start_index: int,
) -> int:
    target_score_position = max(0, int(score_start_index))
    for event_index, event in enumerate(autoplay_events):
        if int(event.get("score_position", 0)) >= target_score_position:
            return event_index
    return len(autoplay_events)


def autoplay_cache_pitches(
    autoplay_events: list[dict[str, object]],
    *,
    include_mistake_variants: bool,
    start_index: int = 0,
) -> list[int]:
    cached_pitches: set[int] = set()

    for note in autoplay_events[max(0, start_index):]:
        chord_pitches = [int(pitch) for pitch in note.get("pitches", [])]
        if not chord_pitches:
            continue

        for target_pitch in chord_pitches:
            cached_pitches.add(target_pitch)

            if not include_mistake_variants:
                continue

            for delta in (-12, -5, -2, -1, 1, 2, 5, 12):
                cached_pitches.add(max(SOUND_START, min(SOUND_END, target_pitch + delta)))

    return sorted(cached_pitches)


def build_workspace_import_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        midi_file=None,
        orchestra_midi_file=None,
        require_orchestra=False,
        library_root=WORKSPACE_DEFAULT_LIBRARY_ROOT,
        title=None,
        skip_study_mode=False,
        skip_calibration=False,
        calibration_level="fast",
        full_chord_policy="chord",
        full_chord_epsilon=0.03,
        split_points=list(WORKSPACE_DEFAULT_SPLIT_POINTS),
        study_chord_epsilons=list(WORKSPACE_DEFAULT_CHORD_EPSILONS),
        force=False,
        list=False,
    )


def workspace_launch_command(workspace: dict[str, Any], mode: str) -> list[str]:
    commands = workspace.get("commands")
    if not isinstance(commands, dict):
        raise RuntimeError("Workspace is missing launch commands.")

    command: list[Any] | None
    if mode == "hands":
        command = commands.get("practice_left")
        if command is None:
            raise RuntimeError("This workspace has no left/right-hand practice mode.")
    else:
        command = commands.get("full_score_json") or commands.get("full_score_midi")
        if command is None:
            raise RuntimeError("This workspace has no full-score launch command.")

    normalized_command = [sys.executable]
    for index, part in enumerate([str(item) for item in command]):
        if index == 0:
            continue
        if part.startswith("--") or part in {"left", "right"}:
            normalized_command.append(part)
            continue

        candidate = Path(part)
        if candidate.is_absolute() or "/" in part or "\\" in part or candidate.suffix:
            normalized_command.append(str(resolve_project_path(candidate)))
            continue

        normalized_command.append(part)
    return normalized_command


def most_recent_workspace_entry() -> dict[str, Any] | None:
    entries = load_library_entries(WORKSPACE_DEFAULT_LIBRARY_ROOT)
    if not entries:
        return None
    return max(entries, key=lambda entry: str(entry.get("created_at", "")))


def run_startup_launcher(*, setup_expected: bool) -> int:
    pygame.init()
    pygame.font.init()

    launcher_size = (1240, 720)
    screen = pygame.display.set_mode(launcher_size)
    pygame.display.set_caption("Virtual AI Orchestra")
    clock = pygame.time.Clock()
    fonts = {
        "title": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 38, bold=True),
        "body": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 24),
        "small": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 18),
        "tiny": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 14),
    }

    orchestra_mode_rect = pygame.Rect(84, 190, 498, 180)
    hands_mode_rect = pygame.Rect(656, 190, 498, 180)
    choose_midi_rect = pygame.Rect(84, 416, 240, 46)
    open_last_rect = pygame.Rect(342, 416, 240, 46)
    quit_rect = pygame.Rect(84, 480, 160, 40)
    selected_mode = "orchestra"
    status_message = "Choose a mode, then load a MIDI file."
    error_message: str | None = None
    processing = False
    processing_stage = ""
    processing_log: list[str] = []
    processing_started_at = 0.0
    pending_launch_command: list[str] | None = None
    worker_queue: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
    import_thread: threading.Thread | None = None
    last_workspace = most_recent_workspace_entry()

    def refresh_last_workspace() -> None:
        nonlocal last_workspace
        last_workspace = most_recent_workspace_entry()

    def draw_mode_card(
        rect: pygame.Rect,
        title: str,
        description: str,
        *,
        active: bool,
        hovered: bool,
    ) -> None:
        fill = ACCENT_SOFT if active else SURFACE
        if hovered and not active:
            fill = BUTTON_HOVER
        pygame.draw.rect(screen, fill, rect, border_radius=24)
        pygame.draw.rect(screen, (18, 24, 33), rect, width=3 if active else 2, border_radius=24)
        draw_text(screen, fonts["body"], title, TEXT_COLOR, (rect.x + 24, rect.y + 22))
        draw_fitted_text(
            screen,
            fonts["small"],
            description,
            SUBTLE_TEXT,
            (rect.x + 24, rect.y + 74),
            rect.width - 48,
        )
        draw_fitted_text(
            screen,
            fonts["tiny"],
            "Algorithm tuning will be calibrated automatically for this piece.",
            SUCCESS,
            (rect.x + 24, rect.y + 130),
            rect.width - 48,
        )

    running = True
    while running:
        while True:
            try:
                message_kind, payload = worker_queue.get_nowait()
            except queue.Empty:
                break

            if message_kind == "progress":
                stage, detail = payload
                processing_stage = str(stage)
                detail_text = str(detail).strip()
                if detail_text:
                    processing_log.append(detail_text)
                    processing_log = processing_log[-8:]
                continue

            if message_kind == "done":
                workspace = payload
                refresh_last_workspace()
                try:
                    pending_launch_command = workspace_launch_command(workspace, selected_mode)
                except Exception as exc:
                    error_message = str(exc)
                    status_message = "Workspace was prepared, but launch failed."
                    processing = False
                    pending_launch_command = None
                    continue
                processing_stage = "Launching Tester"
                processing_log.append("Workspace finished. Switching into the tester.")
                processing_log = processing_log[-8:]
                processing = False
                continue

            if message_kind == "system_exit":
                error_text = str(payload).strip()
                if error_text:
                    error_message = error_text
                status_message = "MIDI import was canceled."
                processing = False
                processing_stage = ""
                pending_launch_command = None
                continue

            if message_kind == "error":
                error_message = f"Import failed: {payload}"
                status_message = "Import failed."
                processing = False
                processing_stage = ""
                pending_launch_command = None
                continue

        if pending_launch_command is not None:
            pygame.quit()
            os.execv(sys.executable, pending_launch_command)

        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
                break

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not processing:
                if orchestra_mode_rect.collidepoint(event.pos):
                    selected_mode = "orchestra"
                    error_message = None
                    continue
                if hands_mode_rect.collidepoint(event.pos):
                    selected_mode = "hands"
                    error_message = None
                    continue
                if quit_rect.collidepoint(event.pos):
                    running = False
                    break
                if open_last_rect.collidepoint(event.pos) and last_workspace is not None:
                    try:
                        command = workspace_launch_command(last_workspace, selected_mode)
                    except Exception as exc:
                        error_message = str(exc)
                        continue
                    pygame.quit()
                    os.execv(sys.executable, command)
                if choose_midi_rect.collidepoint(event.pos):
                    try:
                        selected_midi = pick_midi_file(
                            prompt=(
                                "Select piano MIDI file to import"
                                if selected_mode == "orchestra"
                                else "Select MIDI file to import"
                            )
                        )
                    except SystemExit as exc:
                        error_text = str(exc).strip()
                        if error_text and error_text != "No MIDI file selected.":
                            error_message = error_text
                        continue

                    selected_orchestra_midi: Path | None = None
                    if selected_mode == "orchestra":
                        try:
                            selected_orchestra_midi = pick_midi_file(
                                prompt="Select orchestra MIDI file to import"
                            )
                        except SystemExit as exc:
                            error_text = str(exc).strip()
                            if not error_text or error_text == "No MIDI file selected.":
                                error_message = "Orchestra MIDI is required for Orchestra / Full Score import."
                            else:
                                error_message = error_text
                            continue

                    processing = True
                    status_message = (
                        "Importing piano and orchestra MIDI files and preparing the workspace. "
                        "This can take some time."
                        if selected_mode == "orchestra"
                        else "Importing MIDI and preparing the workspace. This can take some time."
                    )
                    processing_stage = "Loading MIDI"
                    processing_log = [f"Selected piano MIDI: {selected_midi.name}"]
                    if selected_orchestra_midi is not None:
                        processing_log.append(f"Selected orchestra MIDI: {selected_orchestra_midi.name}")
                    processing_started_at = time.time()
                    error_message = None
                    pending_launch_command = None

                    workspace_args = build_workspace_import_namespace()
                    workspace_args.midi_file = selected_midi
                    workspace_args.orchestra_midi_file = selected_orchestra_midi
                    workspace_args.require_orchestra = selected_mode == "orchestra"

                    def progress_callback(stage: str, detail: str) -> None:
                        worker_queue.put(("progress", (stage, detail)))

                    def import_worker() -> None:
                        try:
                            workspace = import_piece_workspace_with_progress(
                                workspace_args,
                                progress_callback=progress_callback,
                            )
                        except SystemExit as exc:
                            worker_queue.put(("system_exit", str(exc)))
                            return
                        except Exception as exc:
                            worker_queue.put(("error", str(exc)))
                            return
                        worker_queue.put(("done", workspace))

                    import_thread = threading.Thread(target=import_worker, daemon=True)
                    import_thread.start()

        screen.fill(BACKGROUND)
        draw_card(screen, pygame.Rect(44, 36, launcher_size[0] - 88, launcher_size[1] - 72), SURFACE_ALT)
        if processing:
            draw_text(screen, fonts["title"], "Preparing MIDI Workspace", TEXT_COLOR, (84, 72))
            draw_fitted_text(
                screen,
                fonts["body"],
                processing_stage or status_message,
                TEXT_COLOR,
                (84, 126),
                1040,
            )
            elapsed_seconds = max(0.0, time.time() - processing_started_at)
            draw_fitted_text(
                screen,
                fonts["small"],
                f"Elapsed: {elapsed_seconds:.1f}s. The app stays responsive while preprocessing runs in the background.",
                SUBTLE_TEXT,
                (84, 168),
                1040,
            )
            draw_fitted_text(
                screen,
                fonts["small"],
                "Current activity log:",
                TEXT_COLOR,
                (84, 214),
                320,
            )
            for log_index, log_line in enumerate(processing_log[-8:]):
                draw_fitted_text(
                    screen,
                    fonts["tiny"],
                    f"{log_index + 1}. {log_line}",
                    SUBTLE_TEXT,
                    (84, 252 + (log_index * 26)),
                    1060,
                )
        else:
            draw_text(screen, fonts["title"], "Virtual AI Orchestra", TEXT_COLOR, (84, 72))
            draw_fitted_text(
                screen,
                fonts["body"],
                "Load the source MIDI, preprocess it, run first-time MIDI setup if needed, and open the tester.",
                TEXT_COLOR,
                (84, 126),
                980,
            )
            draw_fitted_text(screen, fonts["small"], "Import Mode", TEXT_COLOR, (84, 160), 240)

            draw_mode_card(
                orchestra_mode_rect,
                "Orchestra / Full Score",
                "Import a piano MIDI together with a separate orchestra MIDI. The launcher now requires both files explicitly for this mode.",
                active=selected_mode == "orchestra",
                hovered=orchestra_mode_rect.collidepoint(mouse_pos),
            )
            draw_mode_card(
                hands_mode_rect,
                "Left / Right Hand",
                "Prepare study-mode assets with the smart hand splitter and open the tester in hand-practice mode. You can switch hands inside the tester.",
                active=selected_mode == "hands",
                hovered=hands_mode_rect.collidepoint(mouse_pos),
            )

            draw_button(
                screen,
                choose_midi_rect,
                "Load MIDI Pair" if selected_mode == "orchestra" else "Load MIDI",
                fonts["small"],
                active=True,
                hovered=choose_midi_rect.collidepoint(mouse_pos),
            )
            draw_button(
                screen,
                open_last_rect,
                "Open Last Imported",
                fonts["small"],
                active=last_workspace is not None,
                hovered=last_workspace is not None and open_last_rect.collidepoint(mouse_pos),
            )
            draw_button(
                screen,
                quit_rect,
                "Quit",
                fonts["tiny"],
                active=False,
                hovered=quit_rect.collidepoint(mouse_pos),
            )

            draw_fitted_text(screen, fonts["small"], status_message, TEXT_COLOR, (84, 554), 920)
            if setup_expected:
                draw_fitted_text(
                    screen,
                    fonts["small"],
                    "First launch detected: the MIDI setup wizard will open automatically before the tester starts.",
                    WARNING,
                    (84, 586),
                    980,
                )
            else:
                draw_fitted_text(
                    screen,
                    fonts["small"],
                    "MIDI setup can be changed later from Advanced inside the tester.",
                    SUBTLE_TEXT,
                    (84, 586),
                    980,
                )

            if last_workspace is not None:
                last_title = str(last_workspace.get("title", "Untitled"))
                draw_fitted_text(
                    screen,
                    fonts["small"],
                    f"Last imported piece: {last_title}",
                    SUCCESS,
                    (84, 618),
                    980,
                )

            if error_message:
                draw_fitted_text(
                    screen,
                    fonts["small"],
                    error_message,
                    WARNING,
                    (84, 650),
                    1040,
                )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


def draw_piano(
    surface: pygame.Surface,
    white_keys: list[PianoKey],
    black_keys: list[PianoKey],
    pitch_labels: dict[int, str],
    fonts: dict[str, pygame.font.Font],
    active_pitches: set[int],
    flashing_pitches: set[int],
    current_score_pitches: set[int],
) -> None:
    for key in white_keys:
        fill = (254, 252, 248)
        border = (45, 50, 58)

        if key.midi_pitch in current_score_pitches:
            fill = (227, 245, 233)
        if key.midi_pitch in flashing_pitches:
            fill = (219, 236, 255)
        if key.midi_pitch in active_pitches:
            fill = (153, 209, 255)

        pygame.draw.rect(surface, fill, key.rect, border_radius=6)
        pygame.draw.rect(surface, border, key.rect, width=2, border_radius=6)

        label = pitch_labels.get(key.midi_pitch)
        if label:
            rendered = fonts["small"].render(label, True, TEXT_COLOR)
            label_rect = rendered.get_rect(center=(key.rect.centerx, key.rect.bottom - 30))
            surface.blit(rendered, label_rect)

        if key.midi_pitch % 12 == 0:
            note_text = fonts["tiny"].render(
                pitch_to_note_name(key.midi_pitch), True, SUBTLE_TEXT
            )
            note_rect = note_text.get_rect(center=(key.rect.centerx, key.rect.bottom - 14))
            surface.blit(note_text, note_rect)

    for key in black_keys:
        fill = (19, 22, 28)
        border = (6, 8, 10)

        if key.midi_pitch in current_score_pitches:
            fill = (45, 86, 60)
        if key.midi_pitch in flashing_pitches:
            fill = (53, 92, 133)
        if key.midi_pitch in active_pitches:
            fill = (69, 139, 214)

        pygame.draw.rect(surface, fill, key.rect, border_radius=6)
        pygame.draw.rect(surface, border, key.rect, width=2, border_radius=6)


def main() -> int:
    args = build_parser().parse_args()
    settings_exist = SETTINGS_PATH.exists()
    loaded_settings = apply_saved_settings(args)
    launcher_requested = bool(args.launcher) or (
        len(sys.argv) == 1 and not args.midi_routing_test and not args.setup_wizard
    )
    if launcher_requested:
        return run_startup_launcher(setup_expected=not settings_exist)
    if args.midi_routing_test:
        return run_midi_routing_test(args)

    practice_mode_label: str | None = None
    accompaniment_label: str | None = None
    piece_title_reference: Path | None = None
    practice_midi_path: Path | None = None
    if args.practice_hand is not None:
        if args.orchestra_midi is not None:
            print(
                "[INFO] Ignoring --orchestra-midi because --practice-hand selects the "
                "opposite hand accompaniment automatically."
            )
        (
            score_path,
            orchestra_midi_path,
            practice_mode_label,
            accompaniment_label,
            piece_title_reference,
        ) = resolve_practice_materials(args.score_json, args.practice_hand)
        candidate_practice_midi_path = score_path.with_suffix(".mid")
        if candidate_practice_midi_path.exists():
            practice_midi_path = candidate_practice_midi_path
    else:
        score_path = resolve_score_path(args.score_json)
        orchestra_midi_path = resolve_optional_midi_path(args.orchestra_midi)
        piece_title_reference = score_path
    direct_hand_study_pair = (
        args.practice_hand is None and is_hand_study_pair(score_path, orchestra_midi_path)
    )

    pygame.mixer.pre_init(SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.init()
    pygame.font.init()
    pygame.mixer.set_num_channels(128)

    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Virtual AI Orchestra")
    clock = pygame.time.Clock()

    fonts = {
        "title": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 34, bold=True),
        "body": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 24),
        "small": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 18),
        "tiny": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 14),
    }

    tester_initialized_midi = False
    if not pygame.midi.get_init():
        pygame.midi.init()
        tester_initialized_midi = True

    def run_first_run_setup_wizard() -> bool:
        wizard_output_devices = available_midi_output_devices()
        wizard_input_ports = available_midi_input_ports()

        def autodetect_wizard_settings() -> dict[str, Any]:
            detected_input = preferred_auto_midi_input_port(wizard_input_ports)
            detected_output = choose_default_midi_output_id(wizard_output_devices)
            return {
                "midi_out": detected_output,
                "piano_midi_out": None,
                "piano_midi_channel": 1,
                "orchestra_midi_channel": 2,
                "orchestra_volume": clamp_orchestra_mix(float(args.orchestra_volume)),
                "local_orchestra": bool(args.local_orchestra),
                "live_midi_in": detected_input is not None,
                "midi_in_port": detected_input,
                "mute_local_piano": detected_input is not None,
            }

        wizard_settings = autodetect_wizard_settings()
        if bool(loaded_settings) and bool(args.setup_wizard):
            wizard_settings.update(
                {
                    "midi_out": args.midi_out if int(args.midi_out) >= 0 else wizard_settings["midi_out"],
                    "piano_midi_out": args.piano_midi_out,
                    "piano_midi_channel": int(max(1, min(16, int(args.piano_midi_channel)))),
                    "orchestra_midi_channel": int(resolved_orchestra_midi_channel(args)),
                    "orchestra_volume": clamp_orchestra_mix(float(args.orchestra_volume)),
                    "local_orchestra": bool(args.local_orchestra),
                    "live_midi_in": bool(args.live_midi_in),
                    "midi_in_port": args.midi_in_port,
                    "mute_local_piano": bool(args.mute_local_piano),
                }
            )

        panel_rect = pygame.Rect(300, 118, 900, 470)
        auto_button_rect = pygame.Rect(panel_rect.x + 32, panel_rect.bottom - 62, 150, 36)
        skip_button_rect = pygame.Rect(panel_rect.right - 322, panel_rect.bottom - 62, 130, 36)
        save_button_rect = pygame.Rect(panel_rect.right - 170, panel_rect.bottom - 62, 138, 36)
        row_y_start = panel_rect.y + 130
        row_gap = 48

        def wizard_row_rects(row_index: int) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
            row_y = row_y_start + (row_index * row_gap)
            left_rect = pygame.Rect(panel_rect.x + 536, row_y, 36, 32)
            value_rect = pygame.Rect(panel_rect.x + 584, row_y, 250, 32)
            right_rect = pygame.Rect(panel_rect.x + 846, row_y, 36, 32)
            return left_rect, value_rect, right_rect

        def wizard_format_output(output_id: int | None) -> str:
            if output_id is None:
                return "None"
            for device_id, device_name in wizard_output_devices:
                if device_id == output_id:
                    return f"{device_id}: {device_name}"
            return f"{output_id}: unavailable"

        def wizard_cycle_output(delta: int) -> None:
            options = [device_id for device_id, _name in wizard_output_devices]
            if not options:
                wizard_settings["midi_out"] = None
                return
            current = wizard_settings.get("midi_out")
            if current not in options:
                current = options[0]
            current_index = options.index(current)
            wizard_settings["midi_out"] = options[(current_index + delta) % len(options)]

        def wizard_cycle_input(delta: int) -> None:
            options: list[str | None] = [None, *wizard_input_ports]
            current = wizard_settings.get("midi_in_port")
            if current not in options:
                current = options[0]
            current_index = options.index(current)
            wizard_settings["midi_in_port"] = options[(current_index + delta) % len(options)]
            wizard_settings["live_midi_in"] = wizard_settings["midi_in_port"] is not None

        def wizard_adjust_channel(delta: int) -> None:
            current = int(max(1, min(16, int(wizard_settings.get("orchestra_midi_channel", 2)))))
            wizard_settings["orchestra_midi_channel"] = int(max(1, min(16, current + delta)))

        def apply_wizard_settings(*, persist: bool) -> bool:
            args.midi_out = -1 if wizard_settings["midi_out"] is None else int(wizard_settings["midi_out"])
            args.piano_midi_out = wizard_settings["piano_midi_out"]
            args.piano_midi_channel = int(wizard_settings["piano_midi_channel"])
            args.orchestra_midi_channel = int(wizard_settings["orchestra_midi_channel"])
            args.orchestra_volume = float(wizard_settings["orchestra_volume"])
            args.local_orchestra = bool(wizard_settings["local_orchestra"])
            args.live_midi_in = bool(wizard_settings["live_midi_in"])
            args.midi_in_port = wizard_settings["midi_in_port"]
            args.mute_local_piano = bool(wizard_settings["mute_local_piano"])
            if persist:
                save_settings(wizard_settings)
            return True

        running_wizard = True
        while running_wizard:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    engine_left_rect, engine_value_rect, engine_right_rect = wizard_row_rects(0)
                    output_left_rect, _output_value_rect, output_right_rect = wizard_row_rects(1)
                    input_left_rect, input_value_rect, input_right_rect = wizard_row_rects(2)
                    channel_left_rect, channel_value_rect, channel_right_rect = wizard_row_rects(3)
                    piano_left_rect, piano_value_rect, piano_right_rect = wizard_row_rects(4)

                    if auto_button_rect.collidepoint(event.pos):
                        wizard_settings = autodetect_wizard_settings()
                        continue
                    if skip_button_rect.collidepoint(event.pos):
                        return apply_wizard_settings(persist=False)
                    if save_button_rect.collidepoint(event.pos):
                        return apply_wizard_settings(persist=True)
                    if (
                        engine_left_rect.collidepoint(event.pos)
                        or engine_value_rect.collidepoint(event.pos)
                        or engine_right_rect.collidepoint(event.pos)
                    ):
                        wizard_settings["local_orchestra"] = not bool(wizard_settings["local_orchestra"])
                        continue
                    if output_left_rect.collidepoint(event.pos):
                        wizard_cycle_output(-1)
                        continue
                    if output_right_rect.collidepoint(event.pos):
                        wizard_cycle_output(1)
                        continue
                    if input_left_rect.collidepoint(event.pos):
                        wizard_cycle_input(-1)
                        continue
                    if input_right_rect.collidepoint(event.pos):
                        wizard_cycle_input(1)
                        continue
                    if channel_left_rect.collidepoint(event.pos):
                        wizard_adjust_channel(-1)
                        continue
                    if channel_right_rect.collidepoint(event.pos):
                        wizard_adjust_channel(1)
                        continue
                    if (
                        piano_left_rect.collidepoint(event.pos)
                        or piano_value_rect.collidepoint(event.pos)
                        or piano_right_rect.collidepoint(event.pos)
                    ):
                        wizard_settings["mute_local_piano"] = not bool(wizard_settings["mute_local_piano"])
                        continue
                    if (
                        input_left_rect.collidepoint(event.pos)
                        or input_value_rect.collidepoint(event.pos)
                        or input_right_rect.collidepoint(event.pos)
                    ) and not wizard_input_ports:
                        wizard_settings["live_midi_in"] = False

            mouse_pos = pygame.mouse.get_pos()
            screen.fill(BACKGROUND)
            draw_card(screen, panel_rect, SURFACE)
            draw_text(screen, fonts["title"], "First-Run Setup Wizard", TEXT_COLOR, (panel_rect.x + 32, panel_rect.y + 24))
            draw_fitted_text(
                screen,
                fonts["small"],
                "The app auto-detected your MIDI devices. Confirm them once and the config will be saved.",
                SUBTLE_TEXT,
                (panel_rect.x + 32, panel_rect.y + 76),
                panel_rect.width - 64,
            )

            rows = [
                (
                    "Orchestra Engine",
                    "Local samples" if wizard_settings.get("local_orchestra", True) else "Logic / MIDI",
                ),
                ("Orchestra Output", wizard_format_output(wizard_settings.get("midi_out"))),
                ("Digital Piano Input", wizard_settings.get("midi_in_port") or "Off"),
                ("Orchestra Channel", f"Ch {int(wizard_settings.get('orchestra_midi_channel', 2))}"),
                ("Local Piano At Start", "Muted" if wizard_settings.get("mute_local_piano") else "On"),
            ]
            for row_index, (label_text, value_text) in enumerate(rows):
                left_rect, value_rect, right_rect = wizard_row_rects(row_index)
                draw_fitted_text(
                    screen,
                    fonts["body"],
                    label_text,
                    TEXT_COLOR,
                    (panel_rect.x + 32, left_rect.y + 2),
                    470,
                )
                draw_button(
                    screen,
                    left_rect,
                    "<",
                    fonts["tiny"],
                    active=False,
                    hovered=left_rect.collidepoint(mouse_pos),
                )
                draw_input_box(
                    screen,
                    value_rect,
                    str(value_text),
                    fonts["tiny"],
                    active=False,
                    hovered=value_rect.collidepoint(mouse_pos),
                )
                draw_button(
                    screen,
                    right_rect,
                    ">",
                    fonts["tiny"],
                    active=False,
                    hovered=right_rect.collidepoint(mouse_pos),
                )

            status_message = (
                "No MIDI outputs detected. Local orchestra will still work."
                if not wizard_output_devices and wizard_settings.get("local_orchestra", True)
                else "No MIDI outputs detected. Create an IAC bus or connect a synth first."
                if not wizard_output_devices
                else "No external MIDI input found. You can still use the on-screen piano."
                if not wizard_input_ports
                else "You can change technical routing later in Advanced."
            )
            draw_fitted_text(
                screen,
                fonts["small"],
                status_message,
                WARNING if not wizard_output_devices else SUBTLE_TEXT,
                (panel_rect.x + 32, panel_rect.bottom - 112),
                panel_rect.width - 64,
            )
            draw_button(
                screen,
                auto_button_rect,
                "Auto Detect",
                fonts["tiny"],
                active=False,
                hovered=auto_button_rect.collidepoint(mouse_pos),
            )
            draw_button(
                screen,
                skip_button_rect,
                "Use Once",
                fonts["tiny"],
                active=False,
                hovered=skip_button_rect.collidepoint(mouse_pos),
            )
            draw_button(
                screen,
                save_button_rect,
                "Save & Continue",
                fonts["tiny"],
                active=True,
                hovered=save_button_rect.collidepoint(mouse_pos),
            )

            pygame.display.flip()
            clock.tick(60)

        return True

    initial_detected_outputs = available_midi_output_devices()
    initial_detected_inputs = available_midi_input_ports()
    if bool(args.live_midi_in) and not cli_option_present("--midi-in-port"):
        sanitized_port, repair_message = sanitize_auto_selected_midi_input_port(
            args.midi_in_port,
            initial_detected_inputs,
        )
        args.midi_in_port = sanitized_port
        if sanitized_port is None:
            args.live_midi_in = False
        if repair_message:
            print(f"[INFO] {repair_message}")
    if settings_require_setup(
        args,
        settings_exist=settings_exist,
        detected_outputs=initial_detected_outputs,
        detected_inputs=initial_detected_inputs,
    ):
        if not run_first_run_setup_wizard():
            pygame.quit()
            if tester_initialized_midi and pygame.midi.get_init():
                pygame.midi.quit()
            return 0

    def create_score_follower() -> Any:
        if args.practice_hand is None:
            return HybridScoreFollower(score_path, sigma=args.sigma)

        return SequentialPracticeFollower(score_path)

    follower = create_score_follower()
    score_notes = list(follower.score_data.get("notes", []))
    practice_autoplay_note_events = (
        load_hand_practice_note_events(practice_midi_path)
        if practice_midi_path is not None
        else None
    )
    autoplay_events = (
        build_autoplay_events_from_note_events(practice_autoplay_note_events)
        if practice_autoplay_note_events is not None
        else build_autoplay_events(score_notes)
    )
    keyboard_map, pitch_labels = build_keyboard_map()
    white_keys, black_keys, _ = build_piano_layout()
    tempo_tracker = TempoTracker(score_path)
    dispatcher = ScoreEventDispatcher(score_path, tempo_tracker=tempo_tracker)
    orchestra: Any | None = None
    piano_midi: PygameMidiPianoOutput | None = None

    note_sounds: dict[int, pygame.mixer.Sound] = {}
    note_channels = [pygame.mixer.Channel(index) for index in range(1, 64)]
    note_channel_index = 0
    clock_now = time.monotonic
    audio_engine_label = "Salamander Grand Piano samples"
    orchestra_engine_label = "Off"
    real_orchestra_active = False
    local_piano_enabled = True
    local_piano_synth_fallback = False
    tempo_tracking_enabled = True
    orchestra_volume_scale = clamp_orchestra_mix(float(args.orchestra_volume))
    use_local_practice_audio = (bool(args.local_practice_audio) or bool(args.local_orchestra)) and (
        args.practice_hand is not None or direct_hand_study_pair
    )
    use_local_orchestra_audio = (
        bool(args.local_orchestra)
        and orchestra_midi_path is not None
        and not use_local_practice_audio
    )
    if args.local_practice_audio and not use_local_practice_audio:
        print(
            "[WARN] Ignoring --local-practice-audio because the score/accompaniment "
            "is not a left_hand/right_hand study pair."
        )
    duet_study_mode_label = (
        "MODE: ACCOMPANIMENT (Chasing)"
        if direct_hand_study_pair
        else None
    )

    def stop_local_piano_audio() -> None:
        for channel in note_channels:
            channel.stop()

    def refresh_audio_engine_label() -> None:
        nonlocal audio_engine_label
        local_engine_label = "Salamander Grand Piano samples"
        if local_piano_synth_fallback:
            local_engine_label += " + synth fallback"

        if local_piano_enabled:
            audio_engine_label = (
                f"{local_engine_label} + {piano_midi.status_label}"
                if piano_midi is not None
                else local_engine_label
            )
            return

        audio_engine_label = (
            f"Local piano muted + {piano_midi.status_label}"
            if piano_midi is not None
            else "Local piano muted"
        )

    def live_piano_mix_scale() -> float:
        if orchestra is None or not real_orchestra_active:
            return 1.0
        mix_scale = 1.0 - (0.35 * min(orchestra_volume_scale, 1.0)) - (0.10 * max(0.0, orchestra_volume_scale - 1.0))
        return float(np.clip(mix_scale, 0.50, 1.0))

    if orchestra_midi_path is not None and not use_local_practice_audio and not use_local_orchestra_audio:
        try:
            orchestra_start_channel = resolved_orchestra_midi_channel(args)
            if args.practice_hand is not None:
                hand_program = (
                    args.force_orchestra_instrument
                    if args.force_orchestra_instrument is not None
                    else args.force_instrument
                )
                orchestra = HandPracticeMidiAccompaniment(
                    orchestra_midi_path,
                    dispatcher,
                    midi_output_id=args.midi_out,
                    channel=max(0, min(15, orchestra_start_channel - 1)),
                    program=hand_program,
                    volume_scale=orchestra_volume_scale,
                )
                orchestra_engine_label = (
                    f"{orchestra.status_label}: {orchestra_midi_path.name}"
                )
            elif duet_study_mode_label is not None:
                hand_program = (
                    args.force_orchestra_instrument
                    if args.force_orchestra_instrument is not None
                    else args.force_instrument
                    if args.force_instrument is not None
                    else 0
                )
                orchestra = HandPracticeMidiAccompaniment(
                    orchestra_midi_path,
                    dispatcher,
                    midi_output_id=args.midi_out,
                    channel=max(0, min(15, orchestra_start_channel - 1)),
                    program=hand_program,
                    volume_scale=orchestra_volume_scale,
                )
                orchestra_engine_label = (
                    f"{orchestra.status_label}, piano Program {hand_program}: "
                    f"{orchestra_midi_path.name}"
                )
            else:
                merge_orchestra_to_channel = (
                    bool(args.merge_orchestra_to_channel) or not bool(args.preserve_orchestra_channels)
                )
                merged_orchestra_channel = (
                    max(0, min(15, orchestra_start_channel - 1))
                    if merge_orchestra_to_channel
                    else None
                )
                orchestra_force_instrument = (
                    args.force_orchestra_instrument
                    if args.force_orchestra_instrument is not None
                    else args.force_instrument
                )
                if orchestra_force_instrument is None and merge_orchestra_to_channel:
                    orchestra_force_instrument = 48
                orchestra = DynamicOrchestraPlayer(
                    orchestra_midi_path,
                    dispatcher,
                    midi_output_id=args.midi_out,
                    output_channel=merged_orchestra_channel,
                    channel_offset=(
                        0
                        if merge_orchestra_to_channel
                        else max(0, min(15, orchestra_start_channel - 1))
                    ),
                    force_instrument=orchestra_force_instrument,
                    volume_scale=orchestra_volume_scale,
                )
                if merge_orchestra_to_channel:
                    orchestra_engine_label = (
                        f"{orchestra.status_label} (merged ch {orchestra_start_channel})"
                        f": {orchestra_midi_path.name}"
                    )
                else:
                    orchestra_engine_label = (
                        f"{orchestra.status_label} (ch {orchestra_start_channel}+)"
                        f": {orchestra_midi_path.name}"
                    )
            real_orchestra_active = True
            # Warm the transport immediately so the first played note does not
            # pay thread startup cost.
            orchestra.start()
        except RuntimeError as exc:
            print(f"[WARN] Real orchestra disabled: {exc}")
            orchestra = None
            orchestra_engine_label = "Real MIDI unavailable"
            try:
                orchestra = LocalOrchestraAccompaniment(
                    orchestra_midi_path,
                    dispatcher,
                    volume_scale=orchestra_volume_scale,
                )
                orchestra_engine_label = f"{orchestra.status_label} fallback: {orchestra_midi_path.name}"
                real_orchestra_active = True
                orchestra.start()
                print("[INFO] Falling back to local orchestra samples.")
            except RuntimeError as fallback_exc:
                print(f"[WARN] Local orchestra fallback disabled: {fallback_exc}")
                orchestra = None
                orchestra_engine_label = "Orchestra unavailable"
    elif args.fallback_midi_orchestra:
        orchestra = PygameMidiOrchestra(dispatcher, score_path)
        orchestra_engine_label = orchestra.status_label

    if args.piano_midi_out is not None and not use_local_practice_audio:
        try:
            shared_piano_output = (
                orchestra is not None
                and real_orchestra_active
                and args.piano_midi_out == orchestra.midi_output_id
                and orchestra.shared_output is not None
            )
            if shared_piano_output:
                piano_midi = PygameMidiPianoOutput(
                    orchestra.midi_output_id,
                    channel=max(0, min(15, int(args.piano_midi_channel) - 1)),
                    program=None if args.force_instrument is None else int(args.force_instrument),
                    midi_output=orchestra.shared_output,
                    write_lock=orchestra.output_lock,
                )
                piano_midi.status_label = (
                    f"Shared MIDI piano via output #{orchestra.midi_output_id} "
                    f"(ch {max(1, min(16, int(args.piano_midi_channel)))})"
                )
            else:
                piano_midi = PygameMidiPianoOutput(
                    args.piano_midi_out,
                    channel=max(0, min(15, int(args.piano_midi_channel) - 1)),
                    program=None if args.force_instrument is None else int(args.force_instrument),
                )
                if (
                    orchestra is not None
                    and real_orchestra_active
                    and args.piano_midi_out == getattr(orchestra, "midi_output_id", None)
                ):
                    print(
                        "[WARN] Piano and orchestra share the same MIDI output port. "
                        "Dense orchestra playback can degrade piano timing/timbre in Logic. "
                        "Prefer separate IAC buses/ports for piano and orchestra."
                    )
            local_piano_enabled = not args.mute_local_piano
            refresh_audio_engine_label()
        except (RuntimeError, ValueError) as exc:
            print(f"[WARN] Live piano MIDI disabled: {exc}")
            piano_midi = None
            local_piano_enabled = True
            refresh_audio_engine_label()

    def get_note_sound(midi_pitch: int) -> pygame.mixer.Sound:
        nonlocal local_piano_synth_fallback
        if midi_pitch in note_sounds:
            return note_sounds[midi_pitch]

        sound = load_real_piano_sound(midi_pitch)
        if sound is None:
            sound = make_piano_sound(midi_pitch)
            local_piano_synth_fallback = True
            refresh_audio_engine_label()
        note_sounds[midi_pitch] = sound
        return sound

    def play_note_sound(
        midi_pitch: int,
        *,
        send_midi: bool = False,
        velocity: int | None = None,
        volume_scale: float = 1.0,
        mix_role: str = "piano",
    ) -> None:
        nonlocal note_channel_index
        midi_pitch = clamp_midi_pitch(midi_pitch)
        if local_piano_enabled:
            channel = note_channels[note_channel_index]
            note_channel_index = (note_channel_index + 1) % len(note_channels)
            velocity_scale = 1.0 if velocity is None else max(0.05, min(1.0, float(velocity) / 127.0))
            if mix_role == "orchestra":
                channel_gain = apply_orchestra_mix_level(velocity_scale, float(volume_scale))
            else:
                channel_gain = max(0.0, min(1.0, velocity_scale * float(volume_scale) * live_piano_mix_scale()))
            channel.set_volume(channel_gain)
            channel.play(get_note_sound(midi_pitch))
        if send_midi and piano_midi is not None:
            piano_midi.press_note(midi_pitch, velocity=velocity)

    def stop_note_sound(midi_pitch: int) -> None:
        midi_pitch = clamp_midi_pitch(midi_pitch)
        if piano_midi is not None:
            piano_midi.release_note(midi_pitch)

    if use_local_practice_audio and orchestra_midi_path is not None:
        orchestra = LocalHandPracticeAccompaniment(
            orchestra_midi_path,
            dispatcher,
            note_on_callback=lambda midi_pitch, velocity: play_note_sound(
                midi_pitch,
                send_midi=False,
                velocity=velocity,
                volume_scale=orchestra_volume_scale,
                mix_role="orchestra",
            ),
        )
        orchestra_engine_label = f"Local piano samples: {orchestra_midi_path.name}"
        real_orchestra_active = True
        local_piano_enabled = True
        local_practice_pitches = set(orchestra.source_pitches)
        for score_note in score_notes:
            local_practice_pitches.update(score_note_pitches(score_note))
        for note_event in practice_autoplay_note_events or []:
            local_practice_pitches.add(int(note_event.note))
        for midi_pitch in sorted(local_practice_pitches):
            get_note_sound(midi_pitch)
        orchestra.start()
    elif use_local_orchestra_audio and orchestra_midi_path is not None:
        try:
            orchestra = LocalOrchestraAccompaniment(
                orchestra_midi_path,
                dispatcher,
                volume_scale=orchestra_volume_scale,
            )
            orchestra_engine_label = f"{orchestra.status_label}: {orchestra_midi_path.name}"
            real_orchestra_active = True
            orchestra.start()
        except RuntimeError as exc:
            print(f"[WARN] Local orchestra disabled: {exc}")
            orchestra = None
            orchestra_engine_label = "Local orchestra unavailable"

    refresh_audio_engine_label()

    live_midi_input_port: Any | None = None
    live_midi_input_name: str | None = None
    if args.live_midi_in:
        try:
            live_midi_input_device_id, live_midi_input_name = resolve_midi_input_device(args.midi_in_port)
            live_midi_input_port = PygameLiveMidiInputPort(live_midi_input_device_id)
            print(
                f"[INFO] Live MIDI input enabled: {live_midi_input_name} "
                f"(device #{live_midi_input_device_id})"
            )
        except Exception as exc:
            print(f"[WARN] Live MIDI input disabled: {exc}")
            live_midi_input_port = None
            live_midi_input_name = None

    detected_midi_outputs = available_midi_output_devices()
    detected_midi_input_ports = available_midi_input_ports()
    resolved_orchestra_output_id = (
        int(getattr(orchestra, "midi_output_id"))
        if orchestra is not None and hasattr(orchestra, "midi_output_id")
        else int(args.midi_out)
    )
    if resolved_orchestra_output_id < 0:
        try:
            resolved_orchestra_output_id = resolve_midi_output_id(args.midi_out)
        except RuntimeError:
            resolved_orchestra_output_id = -1

    def clamp_orchestra_volume(value: float) -> float:
        return clamp_orchestra_mix(value)

    settings_state: dict[str, Any] = {
        "midi_out": resolved_orchestra_output_id,
        "piano_midi_out": (
            int(piano_midi.midi_output_id)
            if piano_midi is not None
            else None if args.piano_midi_out is None else int(args.piano_midi_out)
        ),
        "piano_midi_channel": int(max(1, min(16, int(args.piano_midi_channel)))),
        "orchestra_midi_channel": int(resolved_orchestra_midi_channel(args)),
        "orchestra_volume": clamp_orchestra_volume(orchestra_volume_scale),
        "local_orchestra": bool(args.local_orchestra),
        "live_midi_in": bool(args.live_midi_in),
        "midi_in_port": live_midi_input_name if live_midi_input_name is not None else args.midi_in_port,
        "mute_local_piano": bool(args.mute_local_piano),
    }
    if settings_state["live_midi_in"] and settings_state["midi_in_port"] is None and detected_midi_input_ports:
        settings_state["midi_in_port"] = preferred_auto_midi_input_port(detected_midi_input_ports)
        settings_state["live_midi_in"] = settings_state["midi_in_port"] is not None
    stored_settings = dict(settings_state)
    advanced_settings = dict(settings_state)
    settings_panel_open = False
    startup_settings_flags = (
        "--midi-out",
        "--piano-midi-out",
        "--piano-midi-channel",
        "--orchestra-midi-channel",
        "--orchestra-volume",
        "--local-orchestra",
        "--midi-orchestra",
        "--live-midi-in",
        "--midi-in-port",
        "--mute-local-piano",
    )
    should_persist_startup_settings = bool(loaded_settings) or any(
        cli_option_present(flag) for flag in startup_settings_flags
    )
    if should_persist_startup_settings:
        save_settings(stored_settings)

    def format_output_option(output_id: int | None) -> str:
        if output_id is None:
            return "Off"
        for candidate_id, candidate_name in detected_midi_outputs:
            if candidate_id == output_id:
                return f"{candidate_id}: {candidate_name}"
        return f"{output_id}: unavailable"

    def format_input_option(port_name: str | None) -> str:
        return "Off" if port_name is None else str(port_name)

    def cycle_output_setting(key: str, delta: int, *, allow_none: bool = False) -> None:
        options: list[int | None] = [device_id for device_id, _ in detected_midi_outputs]
        if allow_none:
            options = [None, *options]
        if not options:
            return
        current_value = advanced_settings.get(key)
        if current_value not in options:
            current_value = options[0]
        current_index = options.index(current_value)
        advanced_settings[key] = options[(current_index + delta) % len(options)]

    def cycle_input_port_setting(delta: int) -> None:
        options: list[str | None] = [None, *detected_midi_input_ports]
        current_value = advanced_settings.get("midi_in_port")
        if current_value not in options:
            current_value = options[0]
        current_index = options.index(current_value)
        advanced_settings["midi_in_port"] = options[(current_index + delta) % len(options)]

    def adjust_advanced_channel(key: str, delta: int) -> None:
        current_value = int(max(1, min(16, int(advanced_settings.get(key, 1)))))
        advanced_settings[key] = int(max(1, min(16, current_value + delta)))

    def apply_orchestra_volume(value: float, *, persist: bool = False) -> None:
        nonlocal orchestra_volume_scale
        orchestra_volume_scale = clamp_orchestra_volume(value)
        args.orchestra_volume = orchestra_volume_scale
        settings_state["orchestra_volume"] = orchestra_volume_scale
        stored_settings["orchestra_volume"] = orchestra_volume_scale
        advanced_settings["orchestra_volume"] = orchestra_volume_scale
        if orchestra is not None and hasattr(orchestra, "set_volume_scale"):
            orchestra.set_volume_scale(orchestra_volume_scale)
        if persist:
            save_settings(stored_settings)

    def ensure_real_orchestra_started() -> None:
        if not real_orchestra_active or orchestra is None:
            return
        if orchestra.is_running:
            return
        orchestra.start()

    def warm_note_cache(mode: str | None, *, start_index: int = 0) -> None:
        if mode is None:
            return

        for midi_pitch in autoplay_cache_pitches(
            autoplay_events,
            include_mistake_variants=mode == "mistakes",
            start_index=start_index,
        ):
            get_note_sound(midi_pitch)

    pressed_keys: set[int] = set()
    pressed_mouse_pitch: int | None = None
    live_active_pitch_counts: dict[int, int] = {}
    pending_live_follow_pitches: set[int] = set()
    pending_live_follow_deadline = 0.0
    pending_live_follow_timestamp = 0.0
    pending_live_follow_source = "keyboard"
    pending_live_follow_tempo_update = True
    flashing_pitches: dict[int, float] = {}
    last_event_pitches: list[int] | None = None
    last_event_timestamp: float | None = None
    last_input_source: str | None = None
    last_advance_at: float | None = None
    session_started_at = clock_now()
    autoplay_mode: str | None = None
    autoplay_index = 0
    autoplay_next_at = 0.0
    autoplay_active_pitches: dict[int, float] = {}
    autoplay_pending_note_offs: list[tuple[float, int]] = []
    autoplay_reference_timestamp = 0.0
    autoplay_reference_nominal_onset = 0.0
    autoplay_rng = np.random.default_rng(20260419)
    autoplay_clean_button_rect = pygame.Rect(WINDOW_SIZE[0] - 596, 52, 248, 42)
    autoplay_mistakes_button_rect = pygame.Rect(WINDOW_SIZE[0] - 332, 52, 248, 42)
    autoplay_start_decrease_rect = pygame.Rect(WINDOW_SIZE[0] - 596, 110, 38, 36)
    autoplay_start_input_rect = pygame.Rect(WINDOW_SIZE[0] - 548, 108, 116, 40)
    autoplay_start_increase_rect = pygame.Rect(WINDOW_SIZE[0] - 422, 110, 38, 36)
    tempo_toggle_button_rect = pygame.Rect(334, 146, 100, 30)
    piano_toggle_button_rect = pygame.Rect(444, 146, 106, 30)
    advanced_button_rect = pygame.Rect(562, 146, 118, 30)
    orchestra_volume_decrease_rect = pygame.Rect(694, 146, 30, 30)
    orchestra_volume_value_rect = pygame.Rect(732, 146, 76, 30)
    orchestra_volume_increase_rect = pygame.Rect(816, 146, 30, 30)
    practice_left_button_rect = pygame.Rect(WINDOW_SIZE[0] - 596, 156, 120, 36)
    practice_right_button_rect = pygame.Rect(WINDOW_SIZE[0] - 464, 156, 120, 36)
    settings_panel_rect = pygame.Rect(430, 156, 640, 548)
    settings_close_button_rect = pygame.Rect(settings_panel_rect.right - 108, settings_panel_rect.y + 16, 78, 32)
    settings_cancel_button_rect = pygame.Rect(settings_panel_rect.right - 264, settings_panel_rect.bottom - 54, 108, 34)
    settings_apply_button_rect = pygame.Rect(settings_panel_rect.right - 144, settings_panel_rect.bottom - 54, 108, 34)
    autoplay_start_index = 0
    autoplay_start_text = "1"
    autoplay_start_input_active = False

    def advanced_row_rects(row_index: int) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        row_y = settings_panel_rect.y + 92 + (row_index * 44)
        left_rect = pygame.Rect(settings_panel_rect.x + 338, row_y, 34, 30)
        value_rect = pygame.Rect(settings_panel_rect.x + 382, row_y, 172, 30)
        right_rect = pygame.Rect(settings_panel_rect.x + 564, row_y, 34, 30)
        return left_rect, value_rect, right_rect

    def remap_pointer_position(
        position: tuple[int, int],
        *,
        rects: list[pygame.Rect] | tuple[pygame.Rect, ...],
    ) -> tuple[int, int]:
        candidates: list[tuple[int, int]] = []

        def add_candidate(point: tuple[int, int] | tuple[float, float]) -> None:
            normalized = (int(round(point[0])), int(round(point[1])))
            if normalized not in candidates:
                candidates.append(normalized)

        add_candidate(position)
        add_candidate(pygame.mouse.get_pos())

        surface = pygame.display.get_surface()
        if surface is not None and hasattr(pygame.display, "get_window_size"):
            surface_width, surface_height = surface.get_size()
            window_width, window_height = pygame.display.get_window_size()
            if (
                surface_width > 0
                and surface_height > 0
                and window_width > 0
                and window_height > 0
                and (surface_width, surface_height) != (window_width, window_height)
            ):
                scale_x = surface_width / window_width
                scale_y = surface_height / window_height
                add_candidate((position[0] * scale_x, position[1] * scale_y))
                mouse_x, mouse_y = pygame.mouse.get_pos()
                add_candidate((mouse_x * scale_x, mouse_y * scale_y))

        for candidate in candidates:
            if any(rect.collidepoint(candidate) for rect in rects):
                return candidate

        return candidates[0]

    def clamped_autoplay_start_index(index: int) -> int:
        if not score_notes:
            return 0
        return max(0, min(index, len(score_notes) - 1))

    def set_autoplay_start_index(index: int) -> None:
        nonlocal autoplay_start_index, autoplay_start_text
        autoplay_start_index = clamped_autoplay_start_index(index)
        autoplay_start_text = str(autoplay_start_index + 1)

    def commit_autoplay_start_text() -> None:
        text = autoplay_start_text.strip()
        if not text:
            set_autoplay_start_index(autoplay_start_index)
            return

        entered_value = max(1, int(text))
        set_autoplay_start_index(entered_value - 1)

    def clear_dispatch_queue() -> None:
        if hasattr(dispatcher, "clear_pending"):
            dispatcher.clear_pending()

    def register_live_pitch_on(midi_pitch: int) -> int:
        normalized_pitch = clamp_midi_pitch(midi_pitch)
        live_active_pitch_counts[normalized_pitch] = live_active_pitch_counts.get(normalized_pitch, 0) + 1
        return normalized_pitch

    def register_live_pitch_off(midi_pitch: int) -> int:
        normalized_pitch = clamp_midi_pitch(midi_pitch)
        current_count = live_active_pitch_counts.get(normalized_pitch, 0)
        if current_count <= 1:
            live_active_pitch_counts.pop(normalized_pitch, None)
        else:
            live_active_pitch_counts[normalized_pitch] = current_count - 1
        return normalized_pitch

    def flush_autoplay_note_offs(now: float | None = None, *, force: bool = False) -> None:
        nonlocal autoplay_pending_note_offs
        if not autoplay_pending_note_offs:
            return

        if force:
            due_pitches = [midi_pitch for _, midi_pitch in autoplay_pending_note_offs]
            autoplay_pending_note_offs = []
        else:
            current_time = clock_now() if now is None else float(now)
            remaining: list[tuple[float, int]] = []
            due_pitches: list[int] = []
            for release_at, midi_pitch in autoplay_pending_note_offs:
                if release_at <= current_time:
                    due_pitches.append(midi_pitch)
                else:
                    remaining.append((release_at, midi_pitch))
            autoplay_pending_note_offs = remaining

        for midi_pitch in due_pitches:
            stop_note_sound(midi_pitch)

    def schedule_autoplay_note_offs(
        midi_pitches: list[int],
        nominal_duration: float | list[float],
        *,
        onset_time: float,
    ) -> None:
        nonlocal autoplay_pending_note_offs
        if isinstance(nominal_duration, list):
            durations = [max(0.05, float(duration)) for duration in nominal_duration]
        else:
            durations = [max(0.05, float(nominal_duration)) for _ in midi_pitches]

        for midi_pitch, duration in compat_zip(midi_pitches, durations):
            release_at = float(onset_time) + duration
            autoplay_pending_note_offs.append((release_at, int(midi_pitch)))

    def apply_tempo_tracking_state() -> None:
        tempo_tracker.reset()
        dispatcher.current_tempo_ratio = 1.0
        if dispatcher.current_index is not None:
            dispatcher.broadcast(dispatcher.current_index, clock_now(), tempo_update=False)

    def build_relaunch_argv(
        *,
        practice_hand_override: str | None = None,
        settings_override: dict[str, Any] | None = None,
    ) -> list[str]:
        value_flags: dict[str, Any] = {}
        bool_flags: dict[str, bool] = {}
        if settings_override is not None:
            value_flags.update(
                {
                    "--midi-out": settings_override.get("midi_out"),
                    "--piano-midi-out": settings_override.get("piano_midi_out"),
                    "--piano-midi-channel": settings_override.get("piano_midi_channel"),
                    "--orchestra-midi-channel": settings_override.get("orchestra_midi_channel"),
                    "--orchestra-volume": settings_override.get("orchestra_volume"),
                    "--midi-in-port": settings_override.get("midi_in_port"),
                }
            )
            bool_flags.update(
                {
                    "--local-orchestra": bool(settings_override.get("local_orchestra", True)),
                    "--midi-orchestra": not bool(settings_override.get("local_orchestra", True)),
                    "--live-midi-in": bool(settings_override.get("live_midi_in")),
                    "--mute-local-piano": bool(settings_override.get("mute_local_piano")),
                }
            )
        if practice_hand_override is not None:
            value_flags["--practice-hand"] = practice_hand_override
        rebuilt_args: list[str] = []
        argv = list(sys.argv[1:])
        index = 0
        while index < len(argv):
            token = argv[index]
            matched_flag = next(
                (
                    flag
                    for flag in (*value_flags.keys(), *bool_flags.keys())
                    if token == flag or token.startswith(f"{flag}=")
                ),
                None,
            )
            if matched_flag is not None:
                if matched_flag in value_flags and token == matched_flag:
                    index += 2
                else:
                    index += 1
                continue
            rebuilt_args.append(token)
            index += 1

        for flag, value in value_flags.items():
            if value is None:
                continue
            rebuilt_args.extend([flag, str(value)])

        for flag, enabled in bool_flags.items():
            if enabled:
                rebuilt_args.append(flag)

        return [sys.executable, str(Path(__file__).resolve()), *rebuilt_args]

    def build_relaunch_argv_with_practice_hand(new_hand: str) -> list[str]:
        return build_relaunch_argv(practice_hand_override=new_hand)

    def apply_advanced_settings_and_restart() -> None:
        nonlocal settings_panel_open
        settings_panel_open = False
        stored_settings.update(advanced_settings)
        stored_settings["orchestra_volume"] = settings_state["orchestra_volume"]
        save_settings(stored_settings)

        restart_settings = dict(stored_settings)
        restart_settings["orchestra_volume"] = settings_state["orchestra_volume"]
        restart_settings["midi_in_port"] = (
            restart_settings["midi_in_port"] if restart_settings.get("live_midi_in") else None
        )

        stop_autoplay()
        release_live_inputs()
        if piano_midi is not None:
            piano_midi.close()
        if orchestra is not None:
            clear_dispatch_queue()
            halt_orchestra_transport()
            orchestra.close()
        dispatcher.close()
        pygame.quit()
        os.execv(
            sys.executable,
            build_relaunch_argv(settings_override=restart_settings),
        )

    def resume_orchestra_transport() -> None:
        if orchestra is not None and hasattr(orchestra, "resume"):
            orchestra.resume()

    def halt_orchestra_transport() -> None:
        if orchestra is not None and hasattr(orchestra, "halt"):
            orchestra.halt()
        elif orchestra is not None and hasattr(orchestra, "panic"):
            orchestra.panic()

    def reset_tracker_state() -> None:
        nonlocal last_event_pitches
        nonlocal last_event_timestamp
        nonlocal last_input_source
        nonlocal last_advance_at
        nonlocal session_started_at
        nonlocal autoplay_active_pitches
        nonlocal autoplay_pending_note_offs

        follower.reset_to_start()
        tempo_tracker.reset()
        clear_dispatch_queue()
        dispatcher.current_index = None
        dispatcher.current_tempo_ratio = tempo_tracker.tempo_ratio
        halt_orchestra_transport()
        release_live_inputs()
        pending_live_follow_pitches.clear()
        if piano_midi is not None:
            piano_midi.panic()
        flashing_pitches.clear()
        last_event_pitches = None
        last_event_timestamp = None
        last_input_source = None
        last_advance_at = None
        session_started_at = clock_now()
        autoplay_active_pitches = {}
        autoplay_pending_note_offs = []

    def set_autoplay(mode: str | None) -> None:
        nonlocal autoplay_mode
        nonlocal autoplay_index
        nonlocal autoplay_next_at
        nonlocal autoplay_active_pitches
        nonlocal autoplay_pending_note_offs
        nonlocal autoplay_reference_timestamp
        nonlocal autoplay_reference_nominal_onset
        nonlocal pressed_mouse_pitch

        autoplay_mode = mode
        pressed_mouse_pitch = None
        if mode is not None:
            # Start autoplay from a cold tracker state so score-following
            # must recover from the played notes instead of teleporting.
            reset_tracker_state()
            autoplay_mode = mode
            start_score_index = clamped_autoplay_start_index(autoplay_start_index)
            autoplay_index = autoplay_event_start_index(autoplay_events, start_score_index)
            warm_note_cache(mode, start_index=autoplay_index)
            first_delay = 0.14
            autoplay_next_at = clock_now() + first_delay
            autoplay_reference_timestamp = autoplay_next_at
            if autoplay_index < len(autoplay_events):
                autoplay_reference_nominal_onset = float(
                    autoplay_events[autoplay_index].get("nominal_onset", 0.0)
                )
            else:
                autoplay_reference_nominal_onset = 0.0
        else:
            autoplay_index = 0
            autoplay_active_pitches = {}
            flush_autoplay_note_offs(force=True)
            autoplay_pending_note_offs = []
            autoplay_reference_timestamp = 0.0
            autoplay_reference_nominal_onset = 0.0

    def shift_autoplay_start_index(delta: int) -> None:
        set_autoplay_start_index(autoplay_start_index + delta)

    def choose_autoplay_pitch(target_pitch: int, mode: str | None) -> int:
        if mode != "mistakes":
            return int(target_pitch)

        roll = float(autoplay_rng.random())
        played_pitch = int(target_pitch)

        if roll < 0.11:
            played_pitch += int(autoplay_rng.choice([-2, -1, 1, 2]))
        elif roll < 0.155:
            played_pitch += int(autoplay_rng.choice([-12, 12]))
        elif roll < 0.19:
            played_pitch += int(autoplay_rng.choice([-5, 5]))

        return max(SOUND_START, min(SOUND_END, played_pitch))

    def dispatch_input(
        pitches: list[int],
        source: str,
        *,
        event_timestamp: float | None = None,
        tempo_update: bool = True,
        dispatch_index: int | None = None,
    ) -> None:
        nonlocal last_event_pitches, last_event_timestamp, last_input_source, last_advance_at
        normalized_pitches = sorted({clamp_midi_pitch(pitch) for pitch in pitches})
        if not normalized_pitches:
            return

        event_timestamp = clock_now() if event_timestamp is None else float(event_timestamp)
        previous_index = follower.current_index

        ensure_real_orchestra_started()
        resume_orchestra_transport()
        predicted_index = follower.process_event(normalized_pitches, event_timestamp)
        effective_tempo_update = bool(tempo_tracking_enabled and tempo_update)
        if not effective_tempo_update:
            dispatcher.current_tempo_ratio = 1.0
        broadcast_index = predicted_index if dispatch_index is None else int(dispatch_index)
        dispatcher.broadcast(broadcast_index, event_timestamp, tempo_update=effective_tempo_update)

        if follower.current_index > previous_index:
            last_advance_at = event_timestamp

        last_event_pitches = normalized_pitches
        last_event_timestamp = event_timestamp
        last_input_source = source
        for midi_pitch in normalized_pitches:
            flashing_pitches[midi_pitch] = event_timestamp + 0.18

    def queue_live_follow_input(
        midi_pitch: int,
        source: str,
        *,
        event_timestamp: float,
        tempo_update: bool,
    ) -> None:
        nonlocal pending_live_follow_deadline
        nonlocal pending_live_follow_timestamp
        nonlocal pending_live_follow_source
        nonlocal pending_live_follow_tempo_update

        if args.practice_hand is None:
            dispatch_input([midi_pitch], source, event_timestamp=event_timestamp, tempo_update=tempo_update)
            return

        if not pending_live_follow_pitches:
            pending_live_follow_timestamp = event_timestamp
            pending_live_follow_source = source
            pending_live_follow_tempo_update = bool(tempo_update)
        else:
            pending_live_follow_timestamp = min(pending_live_follow_timestamp, event_timestamp)
            pending_live_follow_tempo_update = pending_live_follow_tempo_update and bool(tempo_update)

        pending_live_follow_pitches.add(int(midi_pitch))
        pending_live_follow_deadline = clock_now() + LIVE_FOLLOW_BATCH_SECONDS

    def flush_live_follow_input(now: float | None = None, *, force: bool = False) -> None:
        nonlocal pending_live_follow_deadline
        nonlocal pending_live_follow_timestamp
        nonlocal pending_live_follow_source
        nonlocal pending_live_follow_tempo_update

        if not pending_live_follow_pitches:
            return

        now = clock_now() if now is None else float(now)
        if not force and now < pending_live_follow_deadline:
            return

        pitches = sorted(pending_live_follow_pitches)
        source = pending_live_follow_source
        timestamp = pending_live_follow_timestamp
        tempo_update = pending_live_follow_tempo_update
        pending_live_follow_pitches.clear()
        pending_live_follow_deadline = 0.0
        pending_live_follow_timestamp = 0.0
        pending_live_follow_source = "keyboard"
        pending_live_follow_tempo_update = True
        dispatch_input(pitches, source, event_timestamp=timestamp, tempo_update=tempo_update)

    def trigger_note(
        midi_pitch: int,
        source: str,
        *,
        event_timestamp: float | None = None,
        tempo_update: bool = True,
        send_midi: bool = False,
    ) -> None:
        midi_pitch = clamp_midi_pitch(midi_pitch)
        dispatch_timestamp = clock_now() if event_timestamp is None else float(event_timestamp)
        play_note_sound(midi_pitch, send_midi=send_midi)
        queue_live_follow_input(
            midi_pitch,
            source,
            event_timestamp=dispatch_timestamp,
            tempo_update=tempo_update,
        )

    def trigger_chord(
        midi_pitches: list[int],
        source: str,
        *,
        event_timestamp: float | None = None,
        tempo_update: bool = True,
        send_midi: bool = False,
        velocities: list[int] | None = None,
        dispatch_index: int | None = None,
    ) -> None:
        dispatch_timestamp = clock_now() if event_timestamp is None else float(event_timestamp)
        normalized_pitches = sorted({clamp_midi_pitch(pitch) for pitch in midi_pitches})
        if not normalized_pitches:
            return

        velocity_by_pitch = {
            clamp_midi_pitch(pitch): int(max(1, min(127, velocity)))
            for pitch, velocity in compat_zip(midi_pitches, velocities or [])
        }
        for midi_pitch in normalized_pitches:
            play_note_sound(
                midi_pitch,
                send_midi=send_midi,
                velocity=velocity_by_pitch.get(midi_pitch),
            )
        dispatch_input(
            normalized_pitches,
            source,
            event_timestamp=dispatch_timestamp,
            tempo_update=tempo_update,
            dispatch_index=dispatch_index,
        )

    def update_autoplay(now: float) -> None:
        nonlocal autoplay_mode
        nonlocal autoplay_index
        nonlocal autoplay_next_at
        nonlocal autoplay_active_pitches

        if autoplay_mode is None or now < autoplay_next_at:
            return

        if autoplay_index >= len(autoplay_events):
            autoplay_mode = None
            autoplay_active_pitches = {}
            return

        note = autoplay_events[autoplay_index]
        target_pitches = [int(pitch) for pitch in note.get("pitches", [])]
        played_pitches = [
            choose_autoplay_pitch(target_pitch, autoplay_mode)
            for target_pitch in target_pitches
        ]
        raw_durations = note.get("durations", {})
        raw_velocities = note.get("velocities", {})
        durations_by_pitch = raw_durations if isinstance(raw_durations, dict) else {}
        velocities_by_pitch = raw_velocities if isinstance(raw_velocities, dict) else {}
        played_durations = [
            float(durations_by_pitch.get(target_pitch, note.get("nominal_duration", 0.25)))
            for target_pitch in target_pitches
        ]
        played_velocities = [
            int(max(108, int(velocities_by_pitch.get(target_pitch, 108))))
            for target_pitch in target_pitches
        ]
        note_nominal_onset = float(note.get("nominal_onset", 0.0))
        autoplay_event_timestamp = autoplay_reference_timestamp + max(
            0.0,
            note_nominal_onset - autoplay_reference_nominal_onset,
        )

        trigger_chord(
            played_pitches,
            f"autoplay-{autoplay_mode}",
            event_timestamp=autoplay_event_timestamp,
            send_midi=True,
            velocities=played_velocities,
            dispatch_index=(
                int(note.get("index", autoplay_index))
                if use_local_practice_audio
                else None
            ),
        )
        dispatch_finished_at = clock_now()
        schedule_autoplay_note_offs(
            played_pitches,
            played_durations,
            onset_time=dispatch_finished_at,
        )
        autoplay_active_pitches = {
            midi_pitch: dispatch_finished_at + 0.15 for midi_pitch in sorted(set(played_pitches))
        }
        autoplay_index += 1

        if autoplay_index < len(autoplay_events):
            autoplay_next_at = dispatch_finished_at + float(autoplay_events[autoplay_index]["delay"])
        else:
            autoplay_mode = None

    def stop_autoplay() -> None:
        was_autoplay_active = (
            autoplay_mode is not None
            or bool(autoplay_active_pitches)
            or bool(autoplay_pending_note_offs)
        )
        if not was_autoplay_active:
            return

        set_autoplay(None)
        flush_autoplay_note_offs(force=True)
        clear_dispatch_queue()
        halt_orchestra_transport()

    def restart_with_practice_hand(new_hand: str) -> None:
        stop_autoplay()
        release_live_inputs()
        if piano_midi is not None:
            piano_midi.close()
        if orchestra is not None:
            clear_dispatch_queue()
            halt_orchestra_transport()
            orchestra.close()
        dispatcher.close()
        pygame.quit()
        os.execv(sys.executable, build_relaunch_argv_with_practice_hand(new_hand))

    def release_live_inputs() -> None:
        nonlocal pressed_mouse_pitch
        pending_live_follow_pitches.clear()
        if pressed_mouse_pitch is not None:
            stop_note_sound(pressed_mouse_pitch)
            pressed_mouse_pitch = None
        for key_code in tuple(pressed_keys):
            midi_pitch = keyboard_map.get(key_code)
            if midi_pitch is not None:
                stop_note_sound(midi_pitch)
        pressed_keys.clear()
        for midi_pitch in tuple(live_active_pitch_counts):
            stop_note_sound(midi_pitch)
        live_active_pitch_counts.clear()

    def manual_reset_to_start() -> None:
        nonlocal last_event_pitches
        nonlocal last_event_timestamp
        nonlocal last_input_source
        nonlocal last_advance_at
        nonlocal session_started_at
        nonlocal autoplay_active_pitches

        event_timestamp = clock_now()
        stop_autoplay()
        follower.reset_to_start()
        tempo_tracker.reset()
        clear_dispatch_queue()
        dispatcher.current_index = 0
        dispatcher.current_tempo_ratio = tempo_tracker.tempo_ratio
        resume_orchestra_transport()
        dispatcher.broadcast(0, event_timestamp)

        release_live_inputs()
        if piano_midi is not None:
            piano_midi.panic()
        flashing_pitches.clear()
        last_event_pitches = None
        last_event_timestamp = None
        last_input_source = "reset"
        last_advance_at = None
        session_started_at = event_timestamp
        autoplay_active_pitches = {}

    running = True
    window_focus_lost_event = getattr(pygame, "WINDOWFOCUSLOST", None)
    while running:
        if live_midi_input_port is not None:
            try:
                incoming_messages = list(live_midi_input_port.iter_pending())
            except Exception as exc:
                print(f"[WARN] Live MIDI input disconnected ({live_midi_input_name}): {exc}")
                try:
                    live_midi_input_port.close()
                except Exception:
                    pass
                live_midi_input_port = None
                live_midi_input_name = None
                incoming_messages = []

            if incoming_messages:
                live_note_ons: list[int] = []
                live_velocities: list[int] = []
                live_timestamp = clock_now()

                for message in incoming_messages:
                    message_type = getattr(message, "type", None)
                    if message_type == "note_on" and int(getattr(message, "velocity", 0)) > 0:
                        live_note_ons.append(register_live_pitch_on(int(getattr(message, "note", 0))))
                        live_velocities.append(int(getattr(message, "velocity", 0)))
                        continue

                    if message_type == "note_off" or (
                        message_type == "note_on" and int(getattr(message, "velocity", 0)) == 0
                    ):
                        stop_note_sound(register_live_pitch_off(int(getattr(message, "note", 0))))

                if live_note_ons:
                    stop_autoplay()
                    trigger_chord(
                        live_note_ons,
                        "midi-live",
                        event_timestamp=live_timestamp,
                        send_midi=False,
                        velocities=live_velocities,
                    )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_autoplay()
                release_live_inputs()
                if piano_midi is not None:
                    piano_midi.panic()
                running = False
                break

            if window_focus_lost_event is not None and event.type == window_focus_lost_event:
                release_live_inputs()
                continue

            if event.type == pygame.KEYDOWN:
                if settings_panel_open:
                    if event.key == pygame.K_ESCAPE:
                        settings_panel_open = False
                    continue

                if autoplay_start_input_active:
                    if event.key == pygame.K_ESCAPE:
                        autoplay_start_input_active = False
                        set_autoplay_start_index(autoplay_start_index)
                        continue

                    if event.key in {pygame.K_RETURN, pygame.K_KP_ENTER}:
                        commit_autoplay_start_text()
                        autoplay_start_input_active = False
                        continue

                    if event.key == pygame.K_BACKSPACE:
                        autoplay_start_text = autoplay_start_text[:-1]
                        continue

                    if event.unicode.isdigit() and len(autoplay_start_text) < 4:
                        autoplay_start_text += event.unicode
                        autoplay_start_text = autoplay_start_text.lstrip("0")
                        continue

                    continue

                if event.key == pygame.K_ESCAPE:
                    stop_autoplay()
                    release_live_inputs()
                    if piano_midi is not None:
                        piano_midi.panic()
                    running = False
                    break

                if event.key == pygame.K_r:
                    manual_reset_to_start()
                    continue

                midi_pitch = keyboard_map.get(event.key)
                if midi_pitch is None or event.key in pressed_keys:
                    continue

                stop_autoplay()
                pressed_keys.add(event.key)
                trigger_note(midi_pitch, "keyboard", send_midi=True)

            if event.type == pygame.KEYUP:
                midi_pitch = keyboard_map.get(event.key)
                if event.key in pressed_keys:
                    pressed_keys.remove(event.key)
                if midi_pitch is not None:
                    stop_note_sound(midi_pitch)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                interactive_rects = [
                    autoplay_clean_button_rect,
                    autoplay_mistakes_button_rect,
                    autoplay_start_decrease_rect,
                    autoplay_start_input_rect,
                    autoplay_start_increase_rect,
                    tempo_toggle_button_rect,
                    piano_toggle_button_rect,
                    advanced_button_rect,
                    orchestra_volume_decrease_rect,
                    orchestra_volume_value_rect,
                    orchestra_volume_increase_rect,
                ]
                if args.practice_hand is not None:
                    interactive_rects.extend([practice_left_button_rect, practice_right_button_rect])
                if settings_panel_open:
                    interactive_rects.extend(
                        [
                            settings_panel_rect,
                            settings_close_button_rect,
                            settings_cancel_button_rect,
                            settings_apply_button_rect,
                        ]
                    )
                    for row_index in range(8):
                        interactive_rects.extend(advanced_row_rects(row_index))
                click_pos = remap_pointer_position(event.pos, rects=interactive_rects)

                if autoplay_start_input_active and not autoplay_start_input_rect.collidepoint(click_pos):
                    commit_autoplay_start_text()
                    autoplay_start_input_active = False

                if advanced_button_rect.collidepoint(click_pos):
                    settings_panel_open = not settings_panel_open
                    if settings_panel_open:
                        advanced_settings = dict(stored_settings)
                        advanced_settings["orchestra_volume"] = settings_state["orchestra_volume"]
                    continue

                if settings_panel_open:
                    if settings_close_button_rect.collidepoint(click_pos) or settings_cancel_button_rect.collidepoint(click_pos):
                        settings_panel_open = False
                        continue

                    if settings_apply_button_rect.collidepoint(click_pos):
                        apply_advanced_settings_and_restart()
                        continue

                    if not settings_panel_rect.collidepoint(click_pos):
                        settings_panel_open = False
                        continue

                    engine_left_rect, engine_value_rect, engine_right_rect = advanced_row_rects(0)
                    orchestra_out_left_rect, orchestra_out_value_rect, orchestra_out_right_rect = advanced_row_rects(1)
                    orchestra_channel_left_rect, orchestra_channel_value_rect, orchestra_channel_right_rect = advanced_row_rects(2)
                    piano_out_left_rect, piano_out_value_rect, piano_out_right_rect = advanced_row_rects(3)
                    piano_channel_left_rect, piano_channel_value_rect, piano_channel_right_rect = advanced_row_rects(4)
                    live_input_left_rect, live_input_value_rect, live_input_right_rect = advanced_row_rects(5)
                    input_port_left_rect, input_port_value_rect, input_port_right_rect = advanced_row_rects(6)
                    startup_piano_left_rect, startup_piano_value_rect, startup_piano_right_rect = advanced_row_rects(7)

                    if (
                        engine_left_rect.collidepoint(click_pos)
                        or engine_value_rect.collidepoint(click_pos)
                        or engine_right_rect.collidepoint(click_pos)
                    ):
                        advanced_settings["local_orchestra"] = not bool(advanced_settings.get("local_orchestra", True))
                        continue
                    if orchestra_out_left_rect.collidepoint(click_pos):
                        cycle_output_setting("midi_out", -1)
                        continue
                    if orchestra_out_right_rect.collidepoint(click_pos):
                        cycle_output_setting("midi_out", 1)
                        continue
                    if orchestra_channel_left_rect.collidepoint(click_pos):
                        adjust_advanced_channel("orchestra_midi_channel", -1)
                        continue
                    if orchestra_channel_right_rect.collidepoint(click_pos):
                        adjust_advanced_channel("orchestra_midi_channel", 1)
                        continue
                    if piano_out_left_rect.collidepoint(click_pos):
                        cycle_output_setting("piano_midi_out", -1, allow_none=True)
                        continue
                    if piano_out_right_rect.collidepoint(click_pos):
                        cycle_output_setting("piano_midi_out", 1, allow_none=True)
                        continue
                    if piano_channel_left_rect.collidepoint(click_pos):
                        adjust_advanced_channel("piano_midi_channel", -1)
                        continue
                    if piano_channel_right_rect.collidepoint(click_pos):
                        adjust_advanced_channel("piano_midi_channel", 1)
                        continue
                    if (
                        live_input_left_rect.collidepoint(click_pos)
                        or live_input_value_rect.collidepoint(click_pos)
                        or live_input_right_rect.collidepoint(click_pos)
                    ):
                        advanced_settings["live_midi_in"] = not bool(advanced_settings.get("live_midi_in"))
                        if advanced_settings["live_midi_in"] and advanced_settings.get("midi_in_port") is None:
                            advanced_settings["midi_in_port"] = (
                                preferred_auto_midi_input_port(detected_midi_input_ports)
                                if detected_midi_input_ports
                                else None
                            )
                            advanced_settings["live_midi_in"] = advanced_settings["midi_in_port"] is not None
                        continue
                    if input_port_left_rect.collidepoint(click_pos):
                        cycle_input_port_setting(-1)
                        continue
                    if input_port_right_rect.collidepoint(click_pos):
                        cycle_input_port_setting(1)
                        continue
                    if (
                        startup_piano_left_rect.collidepoint(click_pos)
                        or startup_piano_value_rect.collidepoint(click_pos)
                        or startup_piano_right_rect.collidepoint(click_pos)
                    ):
                        advanced_settings["mute_local_piano"] = not bool(advanced_settings.get("mute_local_piano"))
                        continue
                    continue

                if orchestra_volume_decrease_rect.collidepoint(click_pos):
                    apply_orchestra_volume(settings_state["orchestra_volume"] - ORCHESTRA_VOLUME_STEP, persist=True)
                    continue

                if orchestra_volume_increase_rect.collidepoint(click_pos):
                    apply_orchestra_volume(settings_state["orchestra_volume"] + ORCHESTRA_VOLUME_STEP, persist=True)
                    continue

                if tempo_toggle_button_rect.collidepoint(click_pos):
                    tempo_tracking_enabled = not tempo_tracking_enabled
                    apply_tempo_tracking_state()
                    continue

                if piano_toggle_button_rect.collidepoint(click_pos):
                    local_piano_enabled = not local_piano_enabled
                    if not local_piano_enabled:
                        stop_local_piano_audio()
                    refresh_audio_engine_label()
                    continue

                if autoplay_clean_button_rect.collidepoint(click_pos):
                    next_mode = None if autoplay_mode == "clean" else "clean"
                    stop_autoplay()
                    if next_mode is not None:
                        set_autoplay(next_mode)
                    continue

                if autoplay_mistakes_button_rect.collidepoint(click_pos):
                    next_mode = None if autoplay_mode == "mistakes" else "mistakes"
                    stop_autoplay()
                    if next_mode is not None:
                        set_autoplay(next_mode)
                    continue

                if args.practice_hand is not None and practice_left_button_rect.collidepoint(click_pos):
                    if args.practice_hand != "left":
                        restart_with_practice_hand("left")
                    continue

                if args.practice_hand is not None and practice_right_button_rect.collidepoint(click_pos):
                    if args.practice_hand != "right":
                        restart_with_practice_hand("right")
                    continue

                if autoplay_start_decrease_rect.collidepoint(click_pos):
                    shift_autoplay_start_index(-1)
                    continue

                if autoplay_start_input_rect.collidepoint(click_pos):
                    autoplay_start_input_active = True
                    autoplay_start_text = "" if autoplay_start_index == 0 else str(autoplay_start_index + 1)
                    continue

                if autoplay_start_increase_rect.collidepoint(click_pos):
                    shift_autoplay_start_index(1)
                    continue

                midi_pitch = pitch_at_position(click_pos, white_keys, black_keys)
                if midi_pitch is not None:
                    stop_autoplay()
                    pressed_mouse_pitch = midi_pitch
                    trigger_note(midi_pitch, "mouse", send_midi=True)

            if event.type == pygame.MOUSEMOTION and event.buttons[0]:
                if settings_panel_open:
                    continue
                midi_pitch = pitch_at_position(event.pos, white_keys, black_keys)
                if midi_pitch != pressed_mouse_pitch:
                    if pressed_mouse_pitch is not None:
                        stop_note_sound(pressed_mouse_pitch)
                    pressed_mouse_pitch = midi_pitch
                    if midi_pitch is not None:
                        stop_autoplay()
                        trigger_note(midi_pitch, "mouse", send_midi=True)

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if pressed_mouse_pitch is not None:
                    stop_note_sound(pressed_mouse_pitch)
                pressed_mouse_pitch = None

        now = clock_now()
        flush_live_follow_input(now)
        flush_autoplay_note_offs(now)
        update_autoplay(now)
        flashing_pitches = {
            pitch: expires_at for pitch, expires_at in flashing_pitches.items() if expires_at > now
        }
        autoplay_active_pitches = {
            pitch: expires_at
            for pitch, expires_at in autoplay_active_pitches.items()
            if expires_at > now
        }
        flashed_pitch_set = set(flashing_pitches)
        active_pitches = {
            keyboard_map[key_code]
            for key_code in pressed_keys
            if key_code in keyboard_map
        }
        if pressed_mouse_pitch is not None:
            active_pitches.add(pressed_mouse_pitch)
        active_pitches.update(
            midi_pitch for midi_pitch in live_active_pitch_counts if is_visible_piano_pitch(midi_pitch)
        )
        active_pitches.update(autoplay_active_pitches)

        current_index = follower.current_index
        current_score_note = score_notes[current_index]
        current_score_pitches = score_note_pitches(current_score_note)
        next_index = min(current_index + 1, follower.N - 1)
        next_score_note = score_notes[next_index]
        next_score_pitches = score_note_pitches(next_score_note)
        autoplay_start_index = clamped_autoplay_start_index(autoplay_start_index)
        autoplay_start_note = score_notes[autoplay_start_index] if score_notes else current_score_note
        autoplay_start_pitches = score_note_pitches(autoplay_start_note)
        progress_ratio = (current_index + 1) / follower.N
        elapsed_session = now - session_started_at
        piece_name = str(follower.score_data.get("piece_name", score_path.stem))
        piece_title = display_piece_title(piece_title_reference, piece_name)
        header_mid_x = 448
        right_panel_x = autoplay_clean_button_rect.x
        header_mid_width = max(220, right_panel_x - header_mid_x - 28)
        footer_width = WINDOW_SIZE[0] - 128
        autoplay_mode_label = (
            "Mistakes" if autoplay_mode == "mistakes" else "Clean" if autoplay_mode == "clean" else "Off"
        )
        last_input_label = format_chord_label(last_event_pitches) if last_event_pitches else "n/a"
        mouse_pos = pygame.mouse.get_pos()

        screen.fill(BACKGROUND)
        draw_card(screen, pygame.Rect(34, 28, WINDOW_SIZE[0] - 68, HEADER_HEIGHT), SURFACE)
        draw_card(
            screen,
            pygame.Rect(34, HEADER_HEIGHT + 44, WINDOW_SIZE[0] - 68, WINDOW_SIZE[1] - HEADER_HEIGHT - 72),
            SURFACE_ALT,
        )

        draw_text(screen, fonts["title"], "Virtual AI Orchestra", TEXT_COLOR, (64, 50))
        draw_button(
            screen,
            autoplay_clean_button_rect,
            "Autoplay Clean",
            fonts["small"],
            active=autoplay_mode == "clean",
            hovered=autoplay_clean_button_rect.collidepoint(mouse_pos),
        )
        draw_button(
            screen,
            autoplay_mistakes_button_rect,
            "Autoplay Mistakes",
            fonts["small"],
            active=autoplay_mode == "mistakes",
            hovered=autoplay_mistakes_button_rect.collidepoint(mouse_pos),
        )
        draw_button(
            screen,
            autoplay_start_decrease_rect,
            "-",
            fonts["small"],
            active=False,
            hovered=autoplay_start_decrease_rect.collidepoint(mouse_pos),
        )
        draw_input_box(
            screen,
            autoplay_start_input_rect,
            autoplay_start_text if autoplay_start_input_active else str(autoplay_start_index + 1),
            fonts["small"],
            active=autoplay_start_input_active,
            hovered=autoplay_start_input_rect.collidepoint(mouse_pos),
        )
        draw_button(
            screen,
            autoplay_start_increase_rect,
            "+",
            fonts["small"],
            active=False,
            hovered=autoplay_start_increase_rect.collidepoint(mouse_pos),
        )
        draw_fitted_text(screen, fonts["body"], f"Piece: {piece_title}", TEXT_COLOR, (64, 94), 340)
        draw_fitted_text(
            screen,
            fonts["body"],
            f"Progress: {current_index + 1} / {follower.N}",
            SUCCESS if last_advance_at and now - last_advance_at < 0.18 else ACCENT,
            (64, 124),
            340,
        )
        draw_fitted_text(
            screen,
            fonts["small"],
            f"Mode: {follower.mode_label}",
            WARNING if follower.last_selected_model == "oltw" else SUCCESS,
            (64, 154),
            150,
        )
        if tempo_tracking_enabled:
            draw_fitted_text(
                screen,
                fonts["small"],
                f"Tempo: {dispatcher.current_tempo_ratio:.2f}x",
                ACCENT if abs(dispatcher.current_tempo_ratio - 1.0) <= 0.05 else WARNING,
                (214, 154),
                118,
            )
        else:
            draw_fitted_text(
                screen,
                fonts["small"],
                "Tempo: OFF",
                SUBTLE_TEXT,
                (214, 154),
                118,
            )
        draw_button(
            screen,
            tempo_toggle_button_rect,
            "Tempo On" if tempo_tracking_enabled else "Tempo Off",
            fonts["tiny"],
            active=tempo_tracking_enabled,
            hovered=tempo_toggle_button_rect.collidepoint(mouse_pos),
        )
        draw_button(
            screen,
            piano_toggle_button_rect,
            "Piano On" if local_piano_enabled else "Piano Off",
            fonts["tiny"],
            active=local_piano_enabled,
            hovered=piano_toggle_button_rect.collidepoint(mouse_pos),
        )
        draw_button(
            screen,
            advanced_button_rect,
            "Advanced",
            fonts["tiny"],
            active=settings_panel_open,
            hovered=advanced_button_rect.collidepoint(mouse_pos),
        )
        draw_fitted_text(
            screen,
            fonts["tiny"],
            "Orch Vol",
            TEXT_COLOR,
            (690, 128),
            64,
        )
        draw_button(
            screen,
            orchestra_volume_decrease_rect,
            "-",
            fonts["tiny"],
            active=False,
            hovered=orchestra_volume_decrease_rect.collidepoint(mouse_pos),
        )
        draw_input_box(
            screen,
            orchestra_volume_value_rect,
            f"{settings_state['orchestra_volume']:.2f}",
            fonts["tiny"],
            active=False,
            hovered=orchestra_volume_value_rect.collidepoint(mouse_pos),
        )
        draw_button(
            screen,
            orchestra_volume_increase_rect,
            "+",
            fonts["tiny"],
            active=False,
            hovered=orchestra_volume_increase_rect.collidepoint(mouse_pos),
        )

        progress_bar_rect = pygame.Rect(64, 176, 320, 10)
        pygame.draw.rect(screen, (209, 213, 219), progress_bar_rect, border_radius=11)
        filled_width = max(12, int(progress_bar_rect.width * progress_ratio))
        pygame.draw.rect(
            screen,
            ACCENT,
            pygame.Rect(progress_bar_rect.x, progress_bar_rect.y, filled_width, progress_bar_rect.height),
            border_radius=11,
        )

        draw_fitted_text(
            screen,
            fonts["small"],
            f"Current chord: {format_chord_label(current_score_pitches)}",
            TEXT_COLOR,
            (header_mid_x, 56),
            header_mid_width,
        )
        draw_fitted_text(
            screen,
            fonts["small"],
            f"Next chord: {format_chord_label(next_score_pitches)}",
            TEXT_COLOR,
            (header_mid_x, 82),
            header_mid_width,
        )
        draw_fitted_text(
            screen,
            fonts["small"],
            f"Selected chord: {format_chord_label(autoplay_start_pitches)}",
            TEXT_COLOR,
            (header_mid_x, 108),
            header_mid_width,
        )
        if args.practice_hand is not None:
            draw_button(
                screen,
                practice_left_button_rect,
                "Left Hand",
                fonts["tiny"],
                active=args.practice_hand == "left",
                hovered=practice_left_button_rect.collidepoint(mouse_pos),
            )
            draw_button(
                screen,
                practice_right_button_rect,
                "Right Hand",
                fonts["tiny"],
                active=args.practice_hand == "right",
                hovered=practice_right_button_rect.collidepoint(mouse_pos),
            )
        draw_fitted_text(
            screen,
            fonts["small"],
            "Autoplay start",
            TEXT_COLOR,
            (right_panel_x, 90),
            180,
        )

        draw_piano(
            screen,
            white_keys,
            black_keys,
            pitch_labels,
            fonts,
            active_pitches,
            flashed_pitch_set,
            set(current_score_pitches),
        )

        footer_first_line_y = WINDOW_SIZE[1] - 92
        footer_second_line_y = WINDOW_SIZE[1] - 62
        footer_status_parts = [
            f"Audio: {audio_engine_label}",
            f"Orchestra: {orchestra_engine_label}",
        ]
        if practice_mode_label is not None and accompaniment_label is not None:
            footer_status_parts.extend(
                [
                    f"Practice: {practice_mode_label}",
                    f"Accompaniment: {accompaniment_label}",
                ]
            )
        elif duet_study_mode_label is not None:
            footer_status_parts.append(duet_study_mode_label)
        else:
            footer_status_parts.extend(
                [
                    f"Autoplay: {autoplay_mode_label}",
                    f"Start: {autoplay_start_index + 1}/{follower.N}",
                    "Esc: quit",
                    "R: reset",
                ]
            )
        draw_fitted_text(
            screen,
            fonts["small"],
            "    |    ".join(footer_status_parts),
            SUBTLE_TEXT,
            (64, footer_first_line_y),
            footer_width,
        )
        last_input_summary = (
            f"Last input: {last_input_label}    |    source={last_input_source or 'n/a'}    |    "
            f"event t={(last_event_timestamp - session_started_at):0.3f}s    |    session t={elapsed_session:0.1f}s"
            if last_event_timestamp is not None
            else f"Last input: n/a    |    session t={elapsed_session:0.1f}s"
        )
        draw_fitted_text(
            screen,
            fonts["tiny"],
            last_input_summary,
            SUBTLE_TEXT,
            (64, footer_second_line_y),
            footer_width,
        )

        if settings_panel_open:
            overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            overlay.fill((20, 24, 33, 96))
            screen.blit(overlay, (0, 0))
            draw_card(screen, settings_panel_rect, SURFACE)
            draw_text(
                screen,
                fonts["body"],
                "Advanced Settings",
                TEXT_COLOR,
                (settings_panel_rect.x + 28, settings_panel_rect.y + 20),
            )
            draw_fitted_text(
                screen,
                fonts["tiny"],
                "Technical MIDI routing is saved and applied after restart.",
                SUBTLE_TEXT,
                (settings_panel_rect.x + 28, settings_panel_rect.y + 52),
                settings_panel_rect.width - 170,
            )
            draw_button(
                screen,
                settings_close_button_rect,
                "Close",
                fonts["tiny"],
                active=False,
                hovered=settings_close_button_rect.collidepoint(mouse_pos),
            )

            row_specs = [
                (
                    "Orchestra Engine",
                    "Local samples" if advanced_settings.get("local_orchestra", True) else "Logic / MIDI",
                ),
                ("Orchestra MIDI Out", format_output_option(advanced_settings.get("midi_out"))),
                ("Orchestra Channel", f"Ch {int(advanced_settings.get('orchestra_midi_channel', 1))}"),
                ("Piano MIDI Out", format_output_option(advanced_settings.get("piano_midi_out"))),
                ("Piano Channel", f"Ch {int(advanced_settings.get('piano_midi_channel', 1))}"),
                ("Live MIDI Input", "Enabled" if advanced_settings.get("live_midi_in") else "Disabled"),
                ("MIDI Input Port", format_input_option(advanced_settings.get("midi_in_port"))),
                ("Local Piano At Start", "Muted" if advanced_settings.get("mute_local_piano") else "On"),
            ]

            for row_index, (label_text, value_text) in enumerate(row_specs):
                left_rect, value_rect, right_rect = advanced_row_rects(row_index)
                label_y = left_rect.y + 6
                draw_fitted_text(
                    screen,
                    fonts["small"],
                    label_text,
                    TEXT_COLOR,
                    (settings_panel_rect.x + 28, label_y),
                    292,
                )
                draw_button(
                    screen,
                    left_rect,
                    "<",
                    fonts["tiny"],
                    active=False,
                    hovered=left_rect.collidepoint(mouse_pos),
                )
                draw_input_box(
                    screen,
                    value_rect,
                    value_text,
                    fonts["tiny"],
                    active=False,
                    hovered=value_rect.collidepoint(mouse_pos),
                )
                draw_button(
                    screen,
                    right_rect,
                    ">",
                    fonts["tiny"],
                    active=False,
                    hovered=right_rect.collidepoint(mouse_pos),
                )

            draw_button(
                screen,
                settings_cancel_button_rect,
                "Cancel",
                fonts["tiny"],
                active=False,
                hovered=settings_cancel_button_rect.collidepoint(mouse_pos),
            )
            draw_button(
                screen,
                settings_apply_button_rect,
                "Apply",
                fonts["tiny"],
                active=True,
                hovered=settings_apply_button_rect.collidepoint(mouse_pos),
            )

        pygame.display.flip()
        clock.tick(60)

    if piano_midi is not None:
        piano_midi.close()
    if orchestra is not None:
        clear_dispatch_queue()
        halt_orchestra_transport()
        orchestra.close()
    if live_midi_input_port is not None:
        try:
            live_midi_input_port.close()
        except Exception:
            pass
    if tester_initialized_midi and pygame.midi.get_init():
        pygame.midi.quit()
    dispatcher.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
