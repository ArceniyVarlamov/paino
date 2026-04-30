from __future__ import annotations

from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent


def _project_relative_tail(path: Path) -> Path | None:
    for index in range(len(path.parts)):
        tail_parts = path.parts[index:]
        if not tail_parts:
            continue
        candidate = PROJECT_ROOT.joinpath(*tail_parts)
        if candidate.exists():
            return Path(*tail_parts)

    project_name = PROJECT_ROOT.name
    matching_indexes = [index for index, part in enumerate(path.parts) if part == project_name]
    if matching_indexes:
        index = matching_indexes[-1]
        tail_parts = path.parts[index + 1 :]
        return Path(*tail_parts) if tail_parts else Path()

    return None


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return (PROJECT_ROOT / candidate).resolve()

    if candidate.exists():
        return candidate.resolve()

    relative_tail = _project_relative_tail(candidate)
    if relative_tail is not None:
        return (PROJECT_ROOT / relative_tail).resolve()

    return candidate


def project_relative_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def portable_command(command: Iterable[str | Path] | None, *, executable: str = "python3") -> list[str] | None:
    if command is None:
        return None

    portable: list[str] = []
    for index, part in enumerate(command):
        text = str(part)
        if index == 0:
            portable.append(executable)
            continue
        if text.startswith("--"):
            portable.append(text)
            continue

        candidate = Path(text)
        if candidate.is_absolute() or "/" in text or "\\" in text or candidate.suffix:
            portable.append(project_relative_path(candidate))
            continue

        portable.append(text)
    return portable
