#!/usr/bin/env python3

import os


def main() -> int:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    try:
        import pygame.midi
    except ModuleNotFoundError as exc:
        if exc.name == "pygame":
            print("pygame is not installed. Install it with: python3 -m pip install pygame")
        else:
            print(f"pygame.midi is unavailable in this build (missing module: {exc.name}).")
        return 1

    pygame.midi.init()
    try:
        found = False
        count = pygame.midi.get_count()

        print("Available MIDI output devices:")
        for device_id in range(count):
            interface, name, is_input, is_output, _opened = pygame.midi.get_device_info(
                device_id
            )
            if not is_output:
                continue

            found = True
            device_name = name.decode("utf-8", errors="replace")
            interface_name = interface.decode("utf-8", errors="replace")
            print(f"ID {device_id}: {device_name} ({interface_name})")

        if not found:
            print("No MIDI output devices found.")
    finally:
        pygame.midi.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
