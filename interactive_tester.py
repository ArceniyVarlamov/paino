from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_VENDOR_DIR = Path(__file__).resolve().parent / ".vendor"
if _VENDOR_DIR.exists():
    vendor_path = str(_VENDOR_DIR)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

import numpy as np

try:
    import pygame
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pygame is not installed. Install it into the local .vendor directory first."
    ) from exc

from hybrid_fusion import HybridScoreFollower
from midi.real_orchestra_player import DynamicOrchestraPlayer
from output_dispatcher import PygameMidiOrchestra, ScoreEventDispatcher, TempoTracker

DEFAULT_SCORE_PATH = Path(__file__).resolve().parent / "generated_dataset" / "ideal.json"
SAMPLE_RATE = 44_100
PIANO_START = 36
PIANO_END = 96
SOUND_START = 0
SOUND_END = 127
REAL_PIANO_SAMPLE_DIR = (
    Path(__file__).resolve().parent / "assets" / "piano_samples" / "salamander_mp3"
)
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
MIN_AUTOPLAY_GAP = 0.012

WINDOW_PADDING = 48
HEADER_HEIGHT = 188
PIANO_TOP = 246
WHITE_KEY_WIDTH = 38
WHITE_KEY_HEIGHT = 370
BLACK_KEY_WIDTH = 24
BLACK_KEY_HEIGHT = 220


@dataclass(frozen=True)
class PianoKey:
    midi_pitch: int
    rect: pygame.Rect
    is_black: bool


WHITE_KEY_COUNT = sum(
    1 for pitch in range(PIANO_START, PIANO_END + 1) if pitch % 12 in WHITE_PITCH_CLASSES
)
WINDOW_SIZE = (
    max(1500, (WINDOW_PADDING * 2) + (WHITE_KEY_COUNT * WHITE_KEY_WIDTH)),
    720,
)
PIANO_LEFT = (WINDOW_SIZE[0] - (WHITE_KEY_COUNT * WHITE_KEY_WIDTH)) // 2

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
        "--midi-out",
        type=int,
        default=-1,
        help=(
            "MIDI output device ID for the real orchestra player. "
            "Use -1 to select the system default output automatically."
        ),
    )
    return parser


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


def resolve_optional_midi_path(path: Path | None) -> Path | None:
    if path is None:
        return None

    resolved = path.expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix not in {".mid", ".midi"}:
        raise SystemExit(f"Unsupported orchestra MIDI input: {resolved}")
    if not resolved.exists():
        raise SystemExit(f"Orchestra MIDI file not found: {resolved}")
    return resolved


def is_white_key(midi_pitch: int) -> bool:
    return midi_pitch % 12 in WHITE_PITCH_CLASSES


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
    white_pitches = white_pitches_in_range(PIANO_START, PIANO_END)
    labels = [label for row in KEYBOARD_ROWS for label in row]
    keyboard_map: dict[int, int] = {}
    pitch_labels: dict[int, str] = {}

    for label, pitch in zip(labels, white_pitches, strict=False):
        key_code = KEY_LABEL_TO_CODE[label]
        keyboard_map[key_code] = pitch
        pitch_labels[pitch] = label

    return keyboard_map, pitch_labels


def build_piano_layout() -> tuple[list[PianoKey], list[PianoKey], dict[int, PianoKey]]:
    white_keys: list[PianoKey] = []
    black_keys: list[PianoKey] = []
    pitch_to_key: dict[int, PianoKey] = {}
    white_index = 0

    for pitch in range(PIANO_START, PIANO_END + 1):
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


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    position: tuple[int, int],
) -> None:
    rendered = font.render(text, True, color)
    surface.blit(rendered, position)


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


def representative_score_pitch(note: dict[str, object]) -> int:
    return max(score_note_pitches(note))


def format_chord_label(pitches: list[int]) -> str:
    return ", ".join(f"{pitch_to_note_name(pitch)} ({pitch})" for pitch in pitches)


def build_autoplay_events(score_notes: list[dict[str, object]]) -> list[dict[str, float | int]]:
    events: list[dict[str, float | int]] = []
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

        for pitch_index, pitch in enumerate(chord_pitches):
            delay = first_delay if pitch_index == 0 else MIN_AUTOPLAY_GAP
            events.append(
                {
                    "pitch": int(pitch),
                    "delay": delay,
                    "nominal_duration": duration,
                    "nominal_onset": onset,
                    "score_position": score_position,
                    "chord_size": len(chord_pitches),
                    "chord_position": pitch_index,
                }
            )
        previous_onset = onset
        onset_cursor = onset + duration

    return events


def autoplay_event_start_index(
    autoplay_events: list[dict[str, float | int]],
    score_start_index: int,
) -> int:
    target_score_position = max(0, int(score_start_index))
    for event_index, event in enumerate(autoplay_events):
        if int(event.get("score_position", 0)) >= target_score_position:
            return event_index
    return len(autoplay_events)


def autoplay_cache_pitches(
    autoplay_events: list[dict[str, float | int]],
    *,
    include_mistake_variants: bool,
    start_index: int = 0,
) -> list[int]:
    cached_pitches: set[int] = set()

    for note in autoplay_events[max(0, start_index):]:
        target_pitch = int(note["pitch"])
        cached_pitches.add(target_pitch)

        if not include_mistake_variants:
            continue

        for delta in (-12, -5, -2, -1, 1, 2, 5, 12):
            cached_pitches.add(max(SOUND_START, min(SOUND_END, target_pitch + delta)))

    return sorted(cached_pitches)


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
    score_path = resolve_score_path(args.score_json)
    orchestra_midi_path = resolve_optional_midi_path(args.orchestra_midi)

    pygame.mixer.pre_init(SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.init()
    pygame.font.init()
    pygame.mixer.set_num_channels(64)

    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Score-Following Vibe Tester")
    clock = pygame.time.Clock()

    fonts = {
        "title": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 34, bold=True),
        "body": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 24),
        "small": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 18),
        "tiny": pygame.font.SysFont("Avenir Next,Helvetica,Arial", 14),
    }

    follower = HybridScoreFollower(score_path, sigma=args.sigma)
    score_notes = list(follower.score_data.get("notes", []))
    autoplay_events = build_autoplay_events(score_notes)
    keyboard_map, pitch_labels = build_keyboard_map()
    white_keys, black_keys, _ = build_piano_layout()
    tempo_tracker = TempoTracker(score_path)
    dispatcher = ScoreEventDispatcher(score_path, tempo_tracker=tempo_tracker)
    orchestra: Any | None = None

    note_sounds: dict[int, pygame.mixer.Sound] = {}
    note_channels = [pygame.mixer.Channel(index) for index in range(1, 64)]
    note_channel_index = 0
    clock_now = time.monotonic
    audio_engine_label = "Salamander Grand Piano samples"
    orchestra_engine_label = "Off"
    real_orchestra_active = False

    if orchestra_midi_path is not None:
        try:
            orchestra = DynamicOrchestraPlayer(
                orchestra_midi_path,
                dispatcher,
                midi_output_id=args.midi_out,
            )
            orchestra_engine_label = f"{orchestra.status_label}: {orchestra_midi_path.name}"
            real_orchestra_active = True
        except RuntimeError as exc:
            print(f"[WARN] Real orchestra disabled: {exc}")
            orchestra = None
            orchestra_engine_label = "Real MIDI unavailable"
    else:
        orchestra = PygameMidiOrchestra(dispatcher, score_path)
        orchestra_engine_label = orchestra.status_label

    def get_note_sound(midi_pitch: int) -> pygame.mixer.Sound:
        nonlocal audio_engine_label
        if midi_pitch in note_sounds:
            return note_sounds[midi_pitch]

        sound = load_real_piano_sound(midi_pitch)
        if sound is None:
            sound = make_piano_sound(midi_pitch)
            audio_engine_label = "Salamander Grand Piano samples + synth fallback"
        note_sounds[midi_pitch] = sound
        return sound

    def play_note_sound(midi_pitch: int) -> None:
        nonlocal note_channel_index
        channel = note_channels[note_channel_index]
        note_channel_index = (note_channel_index + 1) % len(note_channels)
        channel.play(get_note_sound(midi_pitch))

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

    pressed_keyboard_codes: set[int] = set()
    pressed_mouse_pitch: int | None = None
    flashing_pitches: dict[int, float] = {}
    last_event_pitch: int | None = None
    last_event_timestamp: float | None = None
    last_input_source: str | None = None
    last_advance_at: float | None = None
    session_started_at = clock_now()
    autoplay_mode: str | None = None
    autoplay_index = 0
    autoplay_next_at = 0.0
    autoplay_pitch: int | None = None
    autoplay_pitch_until = 0.0
    autoplay_rng = np.random.default_rng(20260419)
    autoplay_clean_button_rect = pygame.Rect(WINDOW_SIZE[0] - 672, 54, 276, 46)
    autoplay_mistakes_button_rect = pygame.Rect(WINDOW_SIZE[0] - 376, 54, 312, 46)
    autoplay_start_decrease_rect = pygame.Rect(WINDOW_SIZE[0] - 672, 110, 44, 36)
    autoplay_start_input_rect = pygame.Rect(WINDOW_SIZE[0] - 620, 108, 140, 40)
    autoplay_start_increase_rect = pygame.Rect(WINDOW_SIZE[0] - 112, 110, 44, 36)
    autoplay_start_index = 0
    autoplay_start_text = "1"
    autoplay_start_input_active = False

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

    def reset_tracker_state() -> None:
        nonlocal follower
        nonlocal score_notes
        nonlocal autoplay_events
        nonlocal last_event_pitch
        nonlocal last_event_timestamp
        nonlocal last_input_source
        nonlocal last_advance_at
        nonlocal session_started_at
        nonlocal autoplay_pitch
        nonlocal autoplay_pitch_until

        follower = HybridScoreFollower(score_path, sigma=args.sigma)
        score_notes = list(follower.score_data.get("notes", []))
        autoplay_events = build_autoplay_events(score_notes)
        tempo_tracker.reset()
        dispatcher.current_index = None
        dispatcher.current_tempo_ratio = tempo_tracker.tempo_ratio
        if orchestra is not None and hasattr(orchestra, "panic"):
            orchestra.panic()
        pressed_keyboard_codes.clear()
        flashing_pitches.clear()
        last_event_pitch = None
        last_event_timestamp = None
        last_input_source = None
        last_advance_at = None
        session_started_at = clock_now()
        autoplay_pitch = None
        autoplay_pitch_until = 0.0

    def set_autoplay(mode: str | None) -> None:
        nonlocal autoplay_mode
        nonlocal autoplay_index
        nonlocal autoplay_next_at
        nonlocal autoplay_pitch
        nonlocal autoplay_pitch_until
        nonlocal pressed_mouse_pitch

        autoplay_mode = mode
        pressed_mouse_pitch = None
        if mode is not None:
            start_score_index = clamped_autoplay_start_index(autoplay_start_index)
            autoplay_index = autoplay_event_start_index(autoplay_events, start_score_index)
            warm_note_cache(mode, start_index=autoplay_index)
            first_delay = 0.14
            autoplay_next_at = clock_now() + first_delay
        else:
            autoplay_index = 0
            autoplay_pitch = None
            autoplay_pitch_until = 0.0

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

    def trigger_note(midi_pitch: int, source: str) -> None:
        nonlocal last_event_pitch, last_event_timestamp, last_input_source, last_advance_at
        event_timestamp = clock_now()
        previous_index = follower.current_index

        ensure_real_orchestra_started()
        play_note_sound(midi_pitch)
        predicted_index = follower.process_event(midi_pitch, event_timestamp)
        dispatcher.broadcast(predicted_index, event_timestamp)

        if follower.current_index > previous_index:
            last_advance_at = event_timestamp

        last_event_pitch = midi_pitch
        last_event_timestamp = event_timestamp
        last_input_source = source
        flashing_pitches[midi_pitch] = event_timestamp + 0.18

    def update_autoplay(now: float) -> None:
        nonlocal autoplay_mode
        nonlocal autoplay_index
        nonlocal autoplay_next_at
        nonlocal autoplay_pitch
        nonlocal autoplay_pitch_until

        if autoplay_mode is None or now < autoplay_next_at:
            return

        if autoplay_index >= len(autoplay_events):
            autoplay_mode = None
            autoplay_pitch = None
            autoplay_pitch_until = 0.0
            return

        note = autoplay_events[autoplay_index]
        target_pitch = int(note["pitch"])
        played_pitch = choose_autoplay_pitch(target_pitch, autoplay_mode)

        trigger_note(played_pitch, f"autoplay-{autoplay_mode}")
        dispatch_finished_at = clock_now()
        autoplay_pitch = played_pitch
        autoplay_pitch_until = dispatch_finished_at + 0.15
        autoplay_index += 1

        if autoplay_index < len(autoplay_events):
            autoplay_next_at = dispatch_finished_at + float(autoplay_events[autoplay_index]["delay"])
        else:
            autoplay_mode = None

    def stop_autoplay() -> None:
        set_autoplay(None)
        if orchestra is not None and hasattr(orchestra, "panic"):
            orchestra.panic()

    def manual_reset_to_start() -> None:
        nonlocal last_event_pitch
        nonlocal last_event_timestamp
        nonlocal last_input_source
        nonlocal last_advance_at
        nonlocal session_started_at
        nonlocal autoplay_pitch
        nonlocal autoplay_pitch_until
        nonlocal pressed_mouse_pitch

        event_timestamp = clock_now()
        stop_autoplay()
        follower.reset_to_start()
        tempo_tracker.reset()
        dispatcher.current_index = 0
        dispatcher.current_tempo_ratio = tempo_tracker.tempo_ratio
        dispatcher.broadcast(0, event_timestamp)

        pressed_keyboard_codes.clear()
        pressed_mouse_pitch = None
        flashing_pitches.clear()
        last_event_pitch = None
        last_event_timestamp = None
        last_input_source = "reset"
        last_advance_at = None
        session_started_at = event_timestamp
        autoplay_pitch = None
        autoplay_pitch_until = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
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
                    running = False
                    break

                if event.key == pygame.K_r:
                    manual_reset_to_start()
                    continue

                midi_pitch = keyboard_map.get(event.key)
                if midi_pitch is None or event.key in pressed_keyboard_codes:
                    continue

                stop_autoplay()
                pressed_keyboard_codes.add(event.key)
                trigger_note(midi_pitch, "keyboard")

            if event.type == pygame.KEYUP:
                pressed_keyboard_codes.discard(event.key)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if autoplay_start_input_active and not autoplay_start_input_rect.collidepoint(event.pos):
                    commit_autoplay_start_text()
                    autoplay_start_input_active = False

                if autoplay_clean_button_rect.collidepoint(event.pos):
                    set_autoplay(None if autoplay_mode == "clean" else "clean")
                    continue

                if autoplay_mistakes_button_rect.collidepoint(event.pos):
                    set_autoplay(None if autoplay_mode == "mistakes" else "mistakes")
                    continue

                if autoplay_start_decrease_rect.collidepoint(event.pos):
                    shift_autoplay_start_index(-1)
                    continue

                if autoplay_start_input_rect.collidepoint(event.pos):
                    autoplay_start_input_active = True
                    autoplay_start_text = "" if autoplay_start_index == 0 else str(autoplay_start_index + 1)
                    continue

                if autoplay_start_increase_rect.collidepoint(event.pos):
                    shift_autoplay_start_index(1)
                    continue

                midi_pitch = pitch_at_position(event.pos, white_keys, black_keys)
                if midi_pitch is not None:
                    stop_autoplay()
                    pressed_mouse_pitch = midi_pitch
                    trigger_note(midi_pitch, "mouse")

            if event.type == pygame.MOUSEMOTION and event.buttons[0]:
                midi_pitch = pitch_at_position(event.pos, white_keys, black_keys)
                if midi_pitch != pressed_mouse_pitch:
                    pressed_mouse_pitch = midi_pitch
                    if midi_pitch is not None:
                        stop_autoplay()
                        trigger_note(midi_pitch, "mouse")

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                pressed_mouse_pitch = None

        now = clock_now()
        update_autoplay(now)
        flashing_pitches = {
            pitch: expires_at for pitch, expires_at in flashing_pitches.items() if expires_at > now
        }
        flashed_pitch_set = set(flashing_pitches)
        active_pitches = {
            keyboard_map[key_code]
            for key_code in pressed_keyboard_codes
            if key_code in keyboard_map
        }
        if pressed_mouse_pitch is not None:
            active_pitches.add(pressed_mouse_pitch)
        if autoplay_pitch is not None and autoplay_pitch_until > now:
            active_pitches.add(autoplay_pitch)
        elif autoplay_pitch_until <= now:
            autoplay_pitch = None

        current_index = follower.current_index
        current_score_note = score_notes[current_index]
        current_score_pitches = score_note_pitches(current_score_note)
        next_index = min(current_index + 1, follower.N - 1)
        next_score_note = score_notes[next_index]
        next_score_pitches = score_note_pitches(next_score_note)
        current_tempo_ratio = dispatcher.current_tempo_ratio
        autoplay_start_index = clamped_autoplay_start_index(autoplay_start_index)
        autoplay_start_note = score_notes[autoplay_start_index] if score_notes else current_score_note
        autoplay_start_pitches = score_note_pitches(autoplay_start_note)
        progress_ratio = (current_index + 1) / follower.N
        elapsed_session = now - session_started_at
        piece_name = str(follower.score_data.get("piece_name", score_path.stem))

        screen.fill(BACKGROUND)
        draw_card(screen, pygame.Rect(34, 28, WINDOW_SIZE[0] - 68, HEADER_HEIGHT), SURFACE)
        draw_card(
            screen,
            pygame.Rect(34, HEADER_HEIGHT + 44, WINDOW_SIZE[0] - 68, WINDOW_SIZE[1] - HEADER_HEIGHT - 72),
            SURFACE_ALT,
        )

        draw_text(screen, fonts["title"], "Score-Following Vibe Tester", TEXT_COLOR, (64, 54))
        draw_button(
            screen,
            autoplay_clean_button_rect,
            "Autoplay without mistakes",
            fonts["small"],
            active=autoplay_mode == "clean",
            hovered=autoplay_clean_button_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_button(
            screen,
            autoplay_mistakes_button_rect,
            "Autoplay with mistakes",
            fonts["small"],
            active=autoplay_mode == "mistakes",
            hovered=autoplay_mistakes_button_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_button(
            screen,
            autoplay_start_decrease_rect,
            "-",
            fonts["small"],
            active=False,
            hovered=autoplay_start_decrease_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_input_box(
            screen,
            autoplay_start_input_rect,
            autoplay_start_text if autoplay_start_input_active else str(autoplay_start_index + 1),
            fonts["small"],
            active=autoplay_start_input_active,
            hovered=autoplay_start_input_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_button(
            screen,
            autoplay_start_increase_rect,
            "+",
            fonts["small"],
            active=False,
            hovered=autoplay_start_increase_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_text(
            screen,
            fonts["body"],
            f"Piece: {piece_name}",
            TEXT_COLOR,
            (64, 98),
        )
        draw_text(
            screen,
            fonts["body"],
            f"Progress: {current_index + 1} / {follower.N}",
            SUCCESS if last_advance_at and now - last_advance_at < 0.18 else ACCENT,
            (64, 132),
        )

        progress_bar_rect = pygame.Rect(480, 160, WINDOW_SIZE[0] - 560, 22)
        pygame.draw.rect(screen, (209, 213, 219), progress_bar_rect, border_radius=11)
        filled_width = max(12, int(progress_bar_rect.width * progress_ratio))
        pygame.draw.rect(
            screen,
            ACCENT,
            pygame.Rect(progress_bar_rect.x, progress_bar_rect.y, filled_width, progress_bar_rect.height),
            border_radius=11,
        )

        draw_text(
            screen,
            fonts["small"],
            f"Current score chord: {format_chord_label(current_score_pitches)}",
            TEXT_COLOR,
            (480, 56),
        )
        draw_text(
            screen,
            fonts["small"],
            f"Autoplay start # / {follower.N}",
            TEXT_COLOR,
            (WINDOW_SIZE[0] - 620, 88),
        )
        draw_text(
            screen,
            fonts["small"],
            f"Selected chord: {format_chord_label(autoplay_start_pitches)}",
            TEXT_COLOR,
            (WINDOW_SIZE[0] - 320, 117),
        )
        draw_text(
            screen,
            fonts["small"],
            f"MODE: {follower.mode_label}",
            WARNING if follower.last_selected_model == "oltw" else SUCCESS,
            (480, 112),
        )
        draw_text(
            screen,
            fonts["small"],
            f"Tempo Ratio: {current_tempo_ratio:.2f}",
            ACCENT if abs(current_tempo_ratio - 1.0) <= 0.05 else WARNING,
            (680, 112),
        )
        draw_text(
            screen,
            fonts["small"],
            f"Next score chord: {format_chord_label(next_score_pitches)}",
            TEXT_COLOR,
            (480, 84),
        )
        draw_text(
            screen,
            fonts["small"],
            "Mouse: any key on screen    Keyboard: rows Z / A / Q cover wide white-note zones    R: reset",
            SUBTLE_TEXT,
            (480, 138),
        )
        draw_text(
            screen,
            fonts["small"],
            f"Audio: {audio_engine_label}    Orchestra: {orchestra_engine_label}",
            SUBTLE_TEXT,
            (480, 164),
        )

        info_lines = [
            "Bottom row is low register, middle row is mid register, top row is high register.",
            "Mouse clicks and drag-to-glissando work across the full 5-octave keyboard, including black keys.",
            "Top-right buttons run either perfect autoplay or autoplay with deliberate pitch mistakes.",
            "Use +/- or click the number box, type a score note number, and press Enter.",
            "Autoplay now keeps the current tracker state instead of resetting progress.",
            "Play QWERTY piano live while the orchestra follows the dispatcher tempo updates.",
            "Press Esc to quit.",
        ]
        for index, text in enumerate(info_lines):
            draw_text(screen, fonts["small"], text, TEXT_COLOR, (64, 192 + (index * 24)))

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

        if last_event_pitch is not None and last_event_timestamp is not None:
            draw_text(
                screen,
                fonts["small"],
                (
                    f"Last input: {pitch_to_note_name(last_event_pitch)} ({last_event_pitch})    "
                    f"source={last_input_source or 'n/a'}    "
                    f"event t={last_event_timestamp - session_started_at:0.3f}s    "
                    f"session t={elapsed_session:0.1f}s"
                ),
                SUBTLE_TEXT,
                (64, WINDOW_SIZE[1] - 42),
            )

        pygame.display.flip()
        clock.tick(60)

    if orchestra is not None:
        orchestra.close()
    dispatcher.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
