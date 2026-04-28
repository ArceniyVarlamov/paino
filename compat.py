from __future__ import annotations

from itertools import zip_longest

_ZIP_SENTINEL = object()


def compat_zip(*iterables, strict: bool = False):
    if not strict:
        return zip(*iterables)
    return _compat_zip_strict(*iterables)


def _compat_zip_strict(*iterables):
    for values in zip_longest(*iterables, fillvalue=_ZIP_SENTINEL):
        if any(value is _ZIP_SENTINEL for value in values):
            raise ValueError("zip() arguments have different lengths")
        yield values
