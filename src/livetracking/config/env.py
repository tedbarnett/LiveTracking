"""Strict-but-friendly env-var parsing for daemon config.

Parses environment variables into typed values with a "warn and default"
policy: garbage input (e.g. ``LIVETRACKING_PARALLAX_K=1200x``) does NOT
crash the perception daemon at startup — instead the default is kept and
a warning is logged to stderr. Critical for unattended boot via Task
Scheduler, where a crash means a silently-dead service.

All helpers accept an optional ``logger`` callable so tests can capture
the warning text without poking sys.stderr. Default logger writes to
``sys.stderr`` with a ``[env]`` prefix so it stands out in the daemon log.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional


def _default_logger(msg: str) -> None:
    print(f"[env] {msg}", file=sys.stderr, flush=True)


# Values that mean "false" for a boolean env var. Anything else (when the
# var is set) means "true". Case-insensitive.
_FALSE_TOKENS = {"0", "false", "no", "off", "n", "f", ""}


def parse_bool(name: str, default: bool,
               logger: Callable[[str], None] = _default_logger) -> bool:
    """Parse ``$NAME`` as a bool.

    Truthy: anything not in _FALSE_TOKENS. Empty string is treated as
    "unset" — i.e. default applies. Unset env var also returns default.
    Boolean parsing is permissive (no warn-on-garbage path); ``foo=garbage``
    is read as True deliberately, since users sometimes type ``yes``,
    ``on``, ``Y``, etc.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in _FALSE_TOKENS


def parse_float(name: str, default: float,
                min_value: Optional[float] = None,
                max_value: Optional[float] = None,
                logger: Callable[[str], None] = _default_logger) -> float:
    """Parse ``$NAME`` as a float with optional [min, max] clamping.

    On unparseable input (``1200x``, ``abc``), logs a warning and returns
    the default — the daemon keeps booting. On out-of-range input, logs a
    warning and clamps to the bound.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        v = float(raw)
    except ValueError:
        logger(f"{name}={raw!r} is not a number; using default {default}")
        return default
    if min_value is not None and v < min_value:
        logger(f"{name}={v} below min {min_value}; clamping")
        return min_value
    if max_value is not None and v > max_value:
        logger(f"{name}={v} above max {max_value}; clamping")
        return max_value
    return v


def parse_int(name: str, default: int,
              min_value: Optional[int] = None,
              max_value: Optional[int] = None,
              logger: Callable[[str], None] = _default_logger) -> int:
    """Parse ``$NAME`` as an int. Same warn-and-default policy as parse_float.

    Accepts decimal floats by truncating (``1200.7`` -> ``1200``) so a user
    typing ``LIVETRACKING_FOO=1e3`` doesn't get a confusing ValueError.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        # Maybe it's a float-looking literal we should truncate.
        try:
            v = int(float(raw))
        except ValueError:
            logger(f"{name}={raw!r} is not an integer; using default {default}")
            return default
    if min_value is not None and v < min_value:
        logger(f"{name}={v} below min {min_value}; clamping")
        return min_value
    if max_value is not None and v > max_value:
        logger(f"{name}={v} above max {max_value}; clamping")
        return max_value
    return v


def parse_str(name: str, default: str,
              choices: Optional[list[str]] = None,
              logger: Callable[[str], None] = _default_logger) -> str:
    """Parse ``$NAME`` as a string, optionally restricted to ``choices``.

    On unset/empty, returns default. On invalid choice, logs and returns
    default (NOT a coerced choice — wrong is wrong).
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    raw = raw.strip()
    if choices is not None and raw not in choices:
        logger(f"{name}={raw!r} not in {choices}; using default {default!r}")
        return default
    return raw
