#!/usr/bin/env python3
"""Patch a real white background into draw.io-exported SVGs.

The drawio-export container writes ``style="background: #ffffff; ..."`` on
the ``<svg>`` root, but GitHub's HTML sanitizer strips inline ``style``
attributes from embedded SVGs, so the rendered diagram ends up with a
transparent background on GitHub's dark theme. An ordinary ``<rect>`` element
survives sanitization, so this script inserts

    <rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>

as the first child right after the opening ``<defs/><g>``, where ``W``/``H``
are read from the SVG's own ``viewBox``.

Usage:
    python scripts/patch_svg_background.py docs/aws-architecture.svg \
        docs/recommendation-architecture.svg
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

VIEWBOX_RE = re.compile(r'viewBox="0 0 (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)"')
ANCHOR_RE = re.compile(r"(<defs\s*/>\s*<g[^>]*>)")
RECT_RE = re.compile(r'<rect x="0" y="0" width="[\d.]+" height="[\d.]+" fill="#ffffff"\s*/>')


def patch(path: Path) -> bool:
    """Insert (or refresh) the white background rect. Returns True if changed."""
    content = path.read_text()

    viewbox = VIEWBOX_RE.search(content)
    if not viewbox:
        raise ValueError(f"{path}: no viewBox=\"0 0 W H\" found")
    width, height = viewbox.group(1), viewbox.group(2)
    rect = f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'

    if RECT_RE.search(content):
        patched = RECT_RE.sub(rect, content, count=1)
        changed = patched != content
        content = patched
    else:
        anchor = ANCHOR_RE.search(content)
        if not anchor:
            raise ValueError(f"{path}: no <defs/><g> anchor found")
        content = content[: anchor.end()] + rect + content[anchor.end() :]
        changed = True

    if changed:
        path.write_text(content)
    return changed


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    for raw_path in argv:
        path = Path(raw_path)
        changed = patch(path)
        print(f"{'patched' if changed else 'already patched'}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
