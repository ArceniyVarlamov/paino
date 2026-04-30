from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / ".vendor"
for candidate in (PROJECT_ROOT, VENDOR_DIR):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

try:
    import mido
except ModuleNotFoundError as exc:
    raise SystemExit(
        "mido is not installed. Install it into the local .vendor directory first."
    ) from exc

from midi.midi_splitter import build_split_files
from midi_to_score import convert_to_score
from prepare_study_mode_batch import (
    CALIBRATION_LEVELS,
    DEFAULT_CHORD_EPSILONS,
    DEFAULT_SPLIT_POINTS,
    calibrate_command,
    choose_chord_epsilon,
    slugify,
    write_score_json,
)
from portable_paths import portable_command, project_relative_path
from smart_hand_splitter import split_midi_file

DEFAULT_LIBRARY_ROOT = PROJECT_ROOT / "midi" / "library"
LIBRARY_MANIFEST_NAME = "manifest.json"
WORKSPACE_FILE_NAME = "workspace.json"
PAIR_SOLO_TOKENS = frozenset({"solo", "piano"})
PAIR_ORCHESTRA_TOKENS = frozenset({"orchestra", "orch"})
PIANO_TRACK_TOKENS = ("piano", "pianoforte", "fortepiano", "klavier")
ProgressCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class PreparedSourceBundle:
    selected_input: Path
    tracking_source_origin: Path
    imported_midi_path: Path
    imported_json_path: Path
    orchestra_midi_path: Path | None
    orchestra_origin: Path | None
    orchestra_source_kind: str | None
    piano_track_index: int | None = None
    piano_track_name: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import a MIDI file into the project library, run the standard preprocessing "
            "pipeline, and create a structured workspace."
        ),
    )
    parser.add_argument(
        "midi_file",
        nargs="?",
        type=Path,
        help="Source MIDI file. If omitted, a native file picker will be opened.",
    )
    parser.add_argument(
        "--orchestra-midi-file",
        type=Path,
        default=None,
        help="Optional separate orchestra MIDI file to attach explicitly during import.",
    )
    parser.add_argument(
        "--require-orchestra",
        action="store_true",
        help="Require orchestra accompaniment for the import instead of allowing piano-only input.",
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=DEFAULT_LIBRARY_ROOT,
        help="Root folder where imported piece workspaces are stored.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional display title override for the imported piece.",
    )
    parser.add_argument(
        "--skip-study-mode",
        action="store_true",
        help="Only build the full score JSON and skip left/right-hand study-mode assets.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip hybrid profile calibration for all generated score JSON files.",
    )
    parser.add_argument(
        "--calibration-level",
        choices=CALIBRATION_LEVELS,
        default="fast",
        help="Calibration depth for the generated score profiles.",
    )
    parser.add_argument(
        "--full-chord-policy",
        choices=("chord", "highest", "lowest", "first", "flatten"),
        default="chord",
        help="Chord grouping policy for the full imported score JSON.",
    )
    parser.add_argument(
        "--full-chord-epsilon",
        type=float,
        default=0.03,
        help="Onset grouping window in seconds for the full imported score JSON.",
    )
    parser.add_argument(
        "--split-points",
        type=int,
        nargs="+",
        default=list(DEFAULT_SPLIT_POINTS),
        help="Legacy compatibility option. The smart hand splitter ignores manual split points.",
    )
    parser.add_argument(
        "--study-chord-epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_CHORD_EPSILONS),
        help="Candidate onset grouping windows used for study-mode JSON generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reuse the target workspace folder if it already exists instead of allocating a new suffix.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the currently indexed piece workspaces and exit.",
    )
    return parser


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    detail: str,
) -> None:
    if progress_callback is not None:
        progress_callback(str(stage), str(detail))


def normalized_stem_tokens(stem: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", stem)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return [token for token in re.split(r"[^A-Za-z0-9]+", ascii_value.lower()) if token]


def midi_track_name(track: mido.MidiTrack, index: int) -> str:
    explicit_name = str(getattr(track, "name", "") or "").strip()
    if explicit_name:
        return explicit_name
    for message in track:
        if getattr(message, "type", None) == "track_name":
            track_name = str(getattr(message, "name", "") or "").strip()
            if track_name:
                return track_name
    return f"<unnamed track {index}>"


def track_note_count(track: mido.MidiTrack) -> int:
    return sum(
        1
        for message in track
        if getattr(message, "type", None) == "note_on"
        and int(getattr(message, "velocity", 0)) > 0
    )


def detect_linked_solo_orchestra_pair(selected_input_path: Path) -> tuple[Path, Path] | None:
    selected_input_path = selected_input_path.expanduser().resolve()
    selected_tokens = normalized_stem_tokens(selected_input_path.stem)
    if not selected_tokens:
        return None

    selected_has_solo = any(token in PAIR_SOLO_TOKENS for token in selected_tokens)
    selected_has_orchestra = any(token in PAIR_ORCHESTRA_TOKENS for token in selected_tokens)
    if not (selected_has_solo or selected_has_orchestra):
        return None

    selected_base = [
        token
        for token in selected_tokens
        if token not in PAIR_SOLO_TOKENS and token not in PAIR_ORCHESTRA_TOKENS
    ]

    sibling_candidates = sorted(selected_input_path.parent.glob("*.mid")) + sorted(
        selected_input_path.parent.glob("*.midi")
    )
    for candidate in sibling_candidates:
        resolved_candidate = candidate.expanduser().resolve()
        if resolved_candidate == selected_input_path:
            continue

        candidate_tokens = normalized_stem_tokens(candidate.stem)
        candidate_base = [
            token
            for token in candidate_tokens
            if token not in PAIR_SOLO_TOKENS and token not in PAIR_ORCHESTRA_TOKENS
        ]
        if candidate_base != selected_base:
            continue

        candidate_has_solo = any(token in PAIR_SOLO_TOKENS for token in candidate_tokens)
        candidate_has_orchestra = any(token in PAIR_ORCHESTRA_TOKENS for token in candidate_tokens)
        if selected_has_solo and candidate_has_orchestra:
            return selected_input_path, resolved_candidate
        if selected_has_orchestra and candidate_has_solo:
            return resolved_candidate, selected_input_path

    return None


def detect_piano_track_index(midi_file: mido.MidiFile) -> tuple[int | None, str | None]:
    candidates: list[tuple[int, int, int, str]] = []
    for track_index, track in enumerate(midi_file.tracks):
        note_count = track_note_count(track)
        if note_count <= 0:
            continue

        track_name = midi_track_name(track, track_index)
        track_name_lower = track_name.lower()
        programs = {
            int(message.program)
            for message in track
            if getattr(message, "type", None) == "program_change"
        }
        channels = {
            int(message.channel)
            for message in track
            if hasattr(message, "channel")
        }

        if any(token in track_name_lower for token in PIANO_TRACK_TOKENS):
            candidates.append((0, -note_count, track_index, track_name))
            continue
        if any(0 <= program <= 7 for program in programs):
            candidates.append((1, -note_count, track_index, track_name))
            continue
        if channels == {0} and len(midi_file.tracks) >= 4:
            candidates.append((2, -note_count, track_index, track_name))

    if not candidates:
        return None, None

    _, _, track_index, track_name = min(candidates)
    return int(track_index), str(track_name)


def prepare_source_bundle(
    selected_input_path: Path,
    workspace_dir: Path,
    warnings: list[str],
    *,
    orchestra_input_path: Path | None = None,
    require_orchestra: bool = False,
) -> PreparedSourceBundle:
    source_dir = workspace_dir / "source"
    orchestra_dir = workspace_dir / "orchestra"
    source_dir.mkdir(parents=True, exist_ok=True)

    if orchestra_input_path is not None:
        resolved_orchestra_input = orchestra_input_path.expanduser().resolve()
        if resolved_orchestra_input == selected_input_path.resolve():
            raise ValueError("Piano MIDI and orchestra MIDI must be different files.")

        imported_midi_path = source_dir / f"source{selected_input_path.suffix.lower()}"
        imported_json_path = imported_midi_path.with_suffix(".json")
        shutil.copy2(selected_input_path, imported_midi_path)

        orchestra_dir.mkdir(parents=True, exist_ok=True)
        imported_orchestra_path = orchestra_dir / f"orchestra{resolved_orchestra_input.suffix.lower()}"
        shutil.copy2(resolved_orchestra_input, imported_orchestra_path)
        warnings.append("Attached the selected orchestra MIDI explicitly.")

        return PreparedSourceBundle(
            selected_input=selected_input_path.resolve(),
            tracking_source_origin=selected_input_path.resolve(),
            imported_midi_path=imported_midi_path,
            imported_json_path=imported_json_path,
            orchestra_midi_path=imported_orchestra_path,
            orchestra_origin=resolved_orchestra_input,
            orchestra_source_kind="explicit_pair",
        )

    linked_pair = detect_linked_solo_orchestra_pair(selected_input_path)
    if linked_pair is not None:
        solo_origin, orchestra_origin = linked_pair
        imported_midi_path = source_dir / f"source{solo_origin.suffix.lower()}"
        imported_json_path = imported_midi_path.with_suffix(".json")
        shutil.copy2(solo_origin, imported_midi_path)

        orchestra_dir.mkdir(parents=True, exist_ok=True)
        imported_orchestra_path = orchestra_dir / f"orchestra{orchestra_origin.suffix.lower()}"
        shutil.copy2(orchestra_origin, imported_orchestra_path)

        if selected_input_path.resolve() == solo_origin.resolve():
            warnings.append(
                "Detected a linked solo/orchestra MIDI pair and attached the orchestra part automatically."
            )
        else:
            warnings.append(
                "Selected MIDI looks like an orchestra-only file. Using the linked solo sibling as the tracking source."
            )

        return PreparedSourceBundle(
            selected_input=selected_input_path.resolve(),
            tracking_source_origin=solo_origin.resolve(),
            imported_midi_path=imported_midi_path,
            imported_json_path=imported_json_path,
            orchestra_midi_path=imported_orchestra_path,
            orchestra_origin=orchestra_origin.resolve(),
            orchestra_source_kind="linked_pair",
        )

    midi_file = mido.MidiFile(selected_input_path)
    piano_track_index, piano_track_name = detect_piano_track_index(midi_file)
    note_track_count = sum(1 for track in midi_file.tracks if track_note_count(track) > 0)
    should_extract_orchestra = piano_track_index is not None and (
        note_track_count >= 4 or (likely_ensemble_source(selected_input_path) and note_track_count > 1)
    )

    if should_extract_orchestra:
        orchestra_dir.mkdir(parents=True, exist_ok=True)
        imported_midi_path = source_dir / "source.mid"
        imported_json_path = imported_midi_path.with_suffix(".json")
        imported_orchestra_path = orchestra_dir / "orchestra.mid"

        solo_file, orchestra_file = build_split_files(midi_file, int(piano_track_index))
        solo_file.save(imported_midi_path)
        orchestra_file.save(imported_orchestra_path)
        warnings.append(
            f"Detected piano track {piano_track_index} ({piano_track_name}) inside a combined MIDI and extracted orchestra accompaniment automatically."
        )

        return PreparedSourceBundle(
            selected_input=selected_input_path.resolve(),
            tracking_source_origin=selected_input_path.resolve(),
            imported_midi_path=imported_midi_path,
            imported_json_path=imported_json_path,
            orchestra_midi_path=imported_orchestra_path,
            orchestra_origin=selected_input_path.resolve(),
            orchestra_source_kind="track_split",
            piano_track_index=int(piano_track_index),
            piano_track_name=piano_track_name,
        )

    imported_midi_path = source_dir / f"source{selected_input_path.suffix.lower()}"
    imported_json_path = imported_midi_path.with_suffix(".json")
    shutil.copy2(selected_input_path, imported_midi_path)

    if likely_ensemble_source(selected_input_path) and piano_track_index is None:
        warnings.append(
            "This MIDI looks orchestral, but no piano track was detected automatically. Full-score import will follow the selected file directly."
        )

    if require_orchestra:
        raise ValueError(
            "Orchestra import requires a separate orchestra MIDI file or a source that can be split automatically."
        )

    return PreparedSourceBundle(
        selected_input=selected_input_path.resolve(),
        tracking_source_origin=selected_input_path.resolve(),
        imported_midi_path=imported_midi_path,
        imported_json_path=imported_json_path,
        orchestra_midi_path=None,
        orchestra_origin=None,
        orchestra_source_kind=None,
    )


def _pick_midi_file_macos(prompt: str) -> Path:
    try:
        quoted_prompt = json.dumps(str(prompt))
        completed = subprocess.run(
            [
                "osascript",
                "-e",
                (
                    f"POSIX path of (choose file with prompt {quoted_prompt} "
                    'of type {"mid","midi"})'
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "osascript was not found. Pass the MIDI path explicitly."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr_text = (exc.stderr or "").strip().lower()
        if "user canceled" in stderr_text or "cancelled" in stderr_text:
            raise SystemExit("No MIDI file selected.") from exc
        raise SystemExit(
            "Failed to open the macOS file picker. Pass the MIDI path explicitly."
        ) from exc

    selected_path = completed.stdout.strip()
    if not selected_path:
        raise SystemExit("No MIDI file selected.")
    return Path(selected_path).expanduser().resolve()


def pick_midi_file(*, prompt: str = "Select MIDI file to import") -> Path:
    # On recent macOS builds, tkinter file dialogs can crash due to the OS
    # version bridge reporting 16.x to older GUI runtimes. Use the native
    # AppleScript picker instead of importing tkinter there.
    if sys.platform == "darwin":
        return _pick_midi_file_macos(prompt)

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "tkinter is unavailable in this Python build. Pass the MIDI path explicitly."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    selected_path = filedialog.askopenfilename(
        title=str(prompt),
        filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
    )
    root.destroy()
    if not selected_path:
        raise SystemExit("No MIDI file selected.")
    return Path(selected_path).expanduser().resolve()


def resolve_input_midi(
    midi_file: Path | None,
    *,
    prompt: str = "Select MIDI file to import",
) -> Path:
    candidate = midi_file if midi_file is not None else pick_midi_file(prompt=prompt)
    resolved = candidate.expanduser().resolve()
    if resolved.suffix.lower() not in {".mid", ".midi"}:
        raise SystemExit(f"Unsupported MIDI input: {resolved}")
    if not resolved.exists():
        raise SystemExit(f"MIDI file not found: {resolved}")
    return resolved


def allocate_workspace_dir(library_root: Path, slug: str, *, force: bool) -> Path:
    candidate = library_root / slug
    if force or not candidate.exists():
        return candidate

    suffix = 2
    while True:
        variant = library_root / f"{slug}_{suffix}"
        if not variant.exists():
            return variant
        suffix += 1


def load_library_entries(library_root: Path) -> list[dict[str, Any]]:
    manifest_path = library_root / LIBRARY_MANIFEST_NAME
    if manifest_path.exists():
        payload = read_json(manifest_path)
        if isinstance(payload, dict) and isinstance(payload.get("pieces"), list):
            return [entry for entry in payload["pieces"] if isinstance(entry, dict)]

    entries: list[dict[str, Any]] = []
    for workspace_file in sorted(library_root.glob(f"*/{WORKSPACE_FILE_NAME}")):
        payload = read_json(workspace_file)
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def update_library_manifest(library_root: Path, workspace_payload: dict[str, Any]) -> Path:
    entries = load_library_entries(library_root)
    workspace_dir = str(workspace_payload.get("workspace_dir", ""))
    replaced = False
    for index, entry in enumerate(entries):
        if str(entry.get("workspace_dir", "")) == workspace_dir:
            entries[index] = workspace_payload
            replaced = True
            break
    if not replaced:
        entries.append(workspace_payload)

    entries.sort(key=lambda entry: (str(entry.get("title", "")).lower(), str(entry.get("created_at", ""))))
    manifest_path = library_root / LIBRARY_MANIFEST_NAME
    write_json(manifest_path, {"pieces": entries})
    return manifest_path


def print_library_entries(entries: list[dict[str, Any]]) -> None:
    if not entries:
        print("No imported piece workspaces found.")
        return

    for entry in entries:
        title = str(entry.get("title", "Untitled"))
        workspace_dir = str(entry.get("workspace_dir", ""))
        source_midi = str(entry.get("source", {}).get("imported_midi", ""))
        print(f"{title}")
        print(f"  workspace: {workspace_dir}")
        print(f"  source:    {source_midi}")


def build_full_score(
    source_midi_path: Path,
    output_json_path: Path,
    *,
    title: str,
    chord_policy: str,
    chord_epsilon: float,
) -> dict[str, Any]:
    score = convert_to_score(
        source_midi_path,
        chord_policy=chord_policy,
        chord_epsilon=float(chord_epsilon),
        default_duration=0.5,
        min_duration=0.05,
    )
    score["piece_name"] = title
    write_json(output_json_path, score)
    return score


def build_calibration_namespace(level: str) -> SimpleNamespace:
    return SimpleNamespace(
        calibration_level=level,
        calibration_mode=None,
        calibration_passes=None,
        calibration_search_preset=None,
        calibration_offset_states=None,
        calibration_max_starts=None,
        jump_states=None,
        jump_prime_states=None,
        max_jump_scenarios=None,
    )


def run_score_calibration(score_json_path: Path, *, level: str) -> tuple[list[str], Path | None]:
    command = calibrate_command(
        score_json_path,
        build_calibration_namespace(level),
    )
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    profile_path = score_json_path.with_suffix(".hybrid_profile.json")
    return command, profile_path if profile_path.exists() else None


def run_score_calibration_with_progress(
    score_json_path: Path,
    *,
    level: str,
    stage_label: str,
    progress_callback: ProgressCallback | None,
) -> tuple[list[str], Path | None]:
    command = calibrate_command(
        score_json_path,
        build_calibration_namespace(level),
    )
    emit_progress(progress_callback, stage_label, f"Starting calibration for {score_json_path.name}")

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        emit_progress(progress_callback, stage_label, line)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    profile_path = score_json_path.with_suffix(".hybrid_profile.json")
    if profile_path.exists():
        emit_progress(progress_callback, stage_label, f"Wrote {profile_path.name}")
    return command, profile_path if profile_path.exists() else None


def likely_ensemble_source(midi_path: Path) -> bool:
    stem = midi_path.stem.lower()
    return any(token in stem for token in ("concerto", "orchestra", "symph", "ensemble"))


def build_tester_commands(
    source_json_path: Path,
    source_midi_path: Path,
    study_mode_dir: Path | None,
    orchestra_midi_path: Path | None = None,
) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {
        "full_score_json": portable_command([
            sys.executable,
            project_relative_path(PROJECT_ROOT / "interactive_tester.py"),
            project_relative_path(source_json_path),
        ]) or [],
        "full_score_midi": portable_command([
            sys.executable,
            project_relative_path(PROJECT_ROOT / "interactive_tester.py"),
            project_relative_path(source_midi_path),
        ]) or [],
    }

    if orchestra_midi_path is not None:
        orchestra_arg = project_relative_path(orchestra_midi_path)
        commands["full_score_json"].extend(["--orchestra-midi", orchestra_arg])
        commands["full_score_midi"].extend(["--orchestra-midi", orchestra_arg])

    if study_mode_dir is not None:
        commands["practice_left"] = portable_command([
            sys.executable,
            project_relative_path(PROJECT_ROOT / "interactive_tester.py"),
            project_relative_path(study_mode_dir),
            "--practice-hand",
            "left",
        ]) or []
        commands["practice_right"] = portable_command([
            sys.executable,
            project_relative_path(PROJECT_ROOT / "interactive_tester.py"),
            project_relative_path(study_mode_dir),
            "--practice-hand",
            "right",
        ]) or []

    return commands


def import_piece_workspace(args: argparse.Namespace) -> dict[str, Any]:
    return import_piece_workspace_with_progress(args, progress_callback=None)


def import_piece_workspace_with_progress(
    args: argparse.Namespace,
    *,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    selected_input_path = resolve_input_midi(
        args.midi_file,
        prompt="Select piano MIDI file to import",
    )
    orchestra_input_path = (
        resolve_input_midi(
            args.orchestra_midi_file,
            prompt="Select orchestra MIDI file to import",
        )
        if getattr(args, "orchestra_midi_file", None) is not None
        else None
    )
    emit_progress(progress_callback, "Loading MIDI", f"Selected {selected_input_path.name}")
    library_root = args.library_root.expanduser().resolve()
    library_root.mkdir(parents=True, exist_ok=True)

    linked_pair = detect_linked_solo_orchestra_pair(selected_input_path)
    default_title_source = linked_pair[0].stem if linked_pair is not None else selected_input_path.stem
    title = (args.title or default_title_source).strip()
    workspace_dir = allocate_workspace_dir(library_root, slugify(title), force=bool(args.force))
    emit_progress(progress_callback, "Preparing Workspace", f"Workspace folder: {workspace_dir.name}")

    warnings: list[str] = []
    source_bundle = prepare_source_bundle(
        selected_input_path,
        workspace_dir,
        warnings,
        orchestra_input_path=orchestra_input_path,
        require_orchestra=bool(getattr(args, "require_orchestra", False)),
    )
    emit_progress(
        progress_callback,
        "Preparing Source Material",
        f"Tracking source: {source_bundle.tracking_source_origin.name}",
    )
    if source_bundle.orchestra_midi_path is not None:
        emit_progress(
            progress_callback,
            "Preparing Source Material",
            f"Orchestra attached: {source_bundle.orchestra_midi_path.name}",
        )

    imported_midi_path = source_bundle.imported_midi_path
    imported_json_path = source_bundle.imported_json_path
    study_mode_dir = workspace_dir / "study_mode"
    emit_progress(progress_callback, "Building Full Score", f"Converting {imported_midi_path.name} to score JSON")
    full_score = build_full_score(
        imported_midi_path,
        imported_json_path,
        title=title,
        chord_policy=args.full_chord_policy,
        chord_epsilon=float(args.full_chord_epsilon),
    )

    source_profile_path: Path | None = None
    source_calibration_cmd: list[str] | None = None
    source_calibration_future = None
    calibration_executor = ThreadPoolExecutor(max_workers=3) if not args.skip_calibration else None
    if args.skip_calibration:
        warnings.append("Hybrid profile calibration was skipped for all generated score JSON files.")
        emit_progress(progress_callback, "Skipping Calibration", "Hybrid profile calibration disabled by settings")
    else:
        source_calibration_future = calibration_executor.submit(
            run_score_calibration_with_progress,
            imported_json_path,
            level=args.calibration_level,
            stage_label="Calibrating Full Score",
            progress_callback=progress_callback,
        )

    try:
        study_mode_payload: dict[str, Any] | None = None
        if args.skip_study_mode:
            warnings.append("Study mode preprocessing was skipped by request.")
            emit_progress(progress_callback, "Skipping Study Mode", "Study mode preprocessing disabled by settings")
        else:
            study_mode_dir.mkdir(parents=True, exist_ok=True)
            left_midi_path = study_mode_dir / "left_hand.mid"
            right_midi_path = study_mode_dir / "right_hand.mid"
            left_json_path = study_mode_dir / "left_hand.json"
            right_json_path = study_mode_dir / "right_hand.json"
            prep_report_path = study_mode_dir / "prep_report.json"
            emit_progress(progress_callback, "Splitting Hands", "Running smart hand splitter")
            split_result = split_midi_file(
                imported_midi_path,
                left_out=left_midi_path,
                right_out=right_midi_path,
            )

            emit_progress(
                progress_callback,
                "Building Study Mode",
                f"Choosing chord grouping for {left_midi_path.name}",
            )
            chord_epsilon, epsilon_candidates = choose_chord_epsilon(
                left_midi_path,
                study_mode_dir,
                list(args.study_chord_epsilons),
            )
            emit_progress(
                progress_callback,
                "Building Study Mode",
                f"Writing left/right score JSON (epsilon={chord_epsilon:.3f})",
            )
            left_score = write_score_json(left_midi_path, left_json_path, chord_epsilon)
            right_score = write_score_json(right_midi_path, right_json_path, chord_epsilon)

            left_calibration_cmd: list[str] | None = None
            right_calibration_cmd: list[str] | None = None
            left_profile_path: Path | None = None
            right_profile_path: Path | None = None
            if not args.skip_calibration:
                left_calibration_future = calibration_executor.submit(
                    run_score_calibration_with_progress,
                    left_json_path,
                    level=args.calibration_level,
                    stage_label="Calibrating Left Hand",
                    progress_callback=progress_callback,
                )
                right_calibration_future = calibration_executor.submit(
                    run_score_calibration_with_progress,
                    right_json_path,
                    level=args.calibration_level,
                    stage_label="Calibrating Right Hand",
                    progress_callback=progress_callback,
                )
                left_calibration_cmd, left_profile_path = left_calibration_future.result()
                right_calibration_cmd, right_profile_path = right_calibration_future.result()

            study_mode_payload = {
                "directory": project_relative_path(study_mode_dir),
                "left_midi": project_relative_path(left_midi_path),
                "right_midi": project_relative_path(right_midi_path),
                "left_json": project_relative_path(left_json_path),
                "right_json": project_relative_path(right_json_path),
                "split_strategy": "smart_hand_splitter",
                "chord_epsilon": float(chord_epsilon),
                "left_notes": int(split_result.left_notes),
                "right_notes": int(split_result.right_notes),
                "left_score_states": len(left_score.get("notes", [])),
                "right_score_states": len(right_score.get("notes", [])),
                "left_summary": split_result.left_summary,
                "right_summary": split_result.right_summary,
                "left_profile_path": (
                    project_relative_path(left_profile_path) if left_profile_path is not None else None
                ),
                "right_profile_path": (
                    project_relative_path(right_profile_path) if right_profile_path is not None else None
                ),
                "left_calibration_command": portable_command(left_calibration_cmd),
                "right_calibration_command": portable_command(right_calibration_cmd),
                "prep_report": project_relative_path(prep_report_path),
            }
            write_json(
                prep_report_path,
                {
                    "study_mode": study_mode_payload,
                    "epsilon_candidates": [asdict(candidate) for candidate in epsilon_candidates],
                },
            )

        if source_calibration_future is not None:
            source_calibration_cmd, source_profile_path = source_calibration_future.result()
    finally:
        if calibration_executor is not None:
            calibration_executor.shutdown(wait=True)

    emit_progress(progress_callback, "Finalizing Workspace", "Saving workspace metadata")
    commands = build_tester_commands(
        imported_json_path,
        imported_midi_path,
        study_mode_dir if study_mode_payload is not None else None,
        source_bundle.orchestra_midi_path,
    )
    orchestra_payload: dict[str, Any] | None = None
    if source_bundle.orchestra_midi_path is not None:
        orchestra_payload = {
            "imported_midi": project_relative_path(source_bundle.orchestra_midi_path),
            "original_input": (
                project_relative_path(source_bundle.orchestra_origin)
                if source_bundle.orchestra_origin is not None
                else None
            ),
            "source_kind": source_bundle.orchestra_source_kind,
            "piano_track_index": source_bundle.piano_track_index,
            "piano_track_name": source_bundle.piano_track_name,
        }

    workspace_payload = {
        "title": title,
        "slug": workspace_dir.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "workspace_dir": project_relative_path(workspace_dir),
        "source": {
            "original_input": project_relative_path(selected_input_path),
            "tracking_source_origin": project_relative_path(source_bundle.tracking_source_origin),
            "imported_midi": project_relative_path(imported_midi_path),
            "imported_score_json": project_relative_path(imported_json_path),
            "full_score_states": len(full_score.get("notes", [])),
            "full_chord_policy": args.full_chord_policy,
            "full_chord_epsilon": float(args.full_chord_epsilon),
            "profile_path": (
                project_relative_path(source_profile_path) if source_profile_path is not None else None
            ),
            "calibration_command": portable_command(source_calibration_cmd),
        },
        "orchestra": orchestra_payload,
        "study_mode": study_mode_payload,
        "commands": commands,
        "warnings": warnings,
    }
    write_json(workspace_dir / WORKSPACE_FILE_NAME, workspace_payload)
    manifest_path = update_library_manifest(library_root, workspace_payload)
    workspace_payload["library_manifest"] = project_relative_path(manifest_path)
    write_json(workspace_dir / WORKSPACE_FILE_NAME, workspace_payload)
    emit_progress(progress_callback, "Workspace Ready", f"Prepared {workspace_payload['title']}")
    return workspace_payload


def main() -> int:
    args = build_parser().parse_args()
    if args.list:
        entries = load_library_entries(args.library_root.expanduser().resolve())
        print_library_entries(entries)
        return 0

    workspace = import_piece_workspace(args)
    print(f"Imported: {workspace['title']}")
    print(f"Workspace: {workspace['workspace_dir']}")
    print(f"Source MIDI: {workspace['source']['imported_midi']}")
    print(f"Source JSON: {workspace['source']['imported_score_json']}")
    if workspace.get("orchestra"):
        orchestra = workspace["orchestra"]
        print(f"Orchestra MIDI: {orchestra['imported_midi']}")
        if orchestra.get("source_kind"):
            print(f"Orchestra source: {orchestra['source_kind']}")
    if workspace.get("study_mode"):
        study_mode = workspace["study_mode"]
        print(f"Study mode: {study_mode['directory']}")
        print(
            f"Split: {study_mode['split_strategy']}, "
            f"epsilon={study_mode['chord_epsilon']:.3f}"
        )
    if workspace.get("warnings"):
        print("\nWarnings:")
        for warning in workspace["warnings"]:
            print(f"  - {warning}")

    print("\nReady commands:")
    for label, command in workspace["commands"].items():
        print(f"  {label}:")
        print(f"    {' '.join(command)}")

    print(f"\nManifest: {workspace['library_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
