#!/usr/bin/env python3
"""Reject mod.rs in crate sources and #[path] anywhere in crate code.

Library and binary sources use `foo.rs` + `foo/`. Test, bench and example
fixtures keep their own layout (`tests/common/mod.rs` stops Cargo from
treating the helper as a test target), so only `src/` trees are checked
for mod.rs. `#[path]` is rejected everywhere.
"""

import os
from pathlib import Path
import re
import sys

ROOTS = ("crates", "src-tauri", "mods")
SKIP = {"target", "node_modules", ".git"}
PATH_ATTR = re.compile(r"^\s*#!?\[path\s*=", re.MULTILINE)


def walk(root):
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = sorted(d for d in subdirectories if d not in SKIP)
        for name in sorted(files):
            yield Path(directory) / name


def main() -> int:
    files = [path for root in ROOTS if Path(root).is_dir() for path in walk(root)]
    crates = sorted({path.parent for path in files if path.name == "Cargo.toml"})
    sources = [crate / "src" for crate in crates if (crate / "src").is_dir()]
    if not sources:
        print("FAIL: run from the repository root (no crate sources found)", file=sys.stderr)
        return 1

    failures = []
    for path in files:
        if path.suffix != ".rs":
            continue
        if path.name == "mod.rs" and any(source in path.parents for source in sources):
            failures.append(f"use foo.rs + foo/ instead of {path}")
        if PATH_ATTR.search(path.read_text(encoding="utf-8", errors="replace")):
            failures.append(f"#[path] is not allowed: {path}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"module layout OK ({len(sources)} crate sources)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
