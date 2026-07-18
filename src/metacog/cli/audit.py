"""Fail on private paths, secrets, stale markers, and damaged artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Sequence

from ..io import sha256_file

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PATTERNS = {
    "absolute home path": re.compile(r"/(?:home|Users|lustre)/"),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "SSH endpoint": re.compile(r"\bssh\s+(?:[\w.-]+@)?[\w.-]+", re.I),
    "scheduler job identifier": re.compile(r"\bjob[._-]?\d{5,}\b", re.I),
    "possible secret": re.compile(r"(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][^'\"]+", re.I),
    "CJK source text": re.compile(r"[\u3400-\u9fff]"),
    "stale zero-training term": re.compile(r"zero[- ]training", re.I),
}


def _scan_paths(root: Path) -> list[Path]:
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return [root / value.decode("utf-8") for value in result.stdout.split(b"\0") if value]
    return [path for path in root.rglob("*") if path.is_file()]


def scan(root: Path) -> list[str]:
    findings = []
    for path in sorted(_scan_paths(root)):
        if ".git" in path.parts or path.suffix not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                if label == "CJK source text" and path.suffix in {".csv", ".jsonl"}:
                    continue
                if pattern.search(line):
                    findings.append(f"{path.relative_to(root)}:{line_number}: {label}")
    return findings


def artifact_findings(root: Path) -> list[str]:
    manifest_path = root / "artifacts" / "publication" / "manifest.json"
    if not manifest_path.exists():
        return ["artifacts/publication/manifest.json: publication manifest is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    findings = []
    for entry in manifest.get("files", []):
        path = manifest_path.parent / "tables" / entry["release_file"]
        if not path.exists():
            findings.append(f"{path.relative_to(root)}: listed file is missing")
            continue
        if path.stat().st_size != int(entry["bytes"]):
            findings.append(f"{path.relative_to(root)}: byte size does not match manifest")
        if sha256_file(path) != entry["sha256"]:
            findings.append(f"{path.relative_to(root)}: checksum does not match manifest")
    return findings


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    findings = scan(root) + artifact_findings(root)
    if findings:
        raise SystemExit("Release-source audit failed:\n" + "\n".join(findings))
    print("Release-source audit passed")


if __name__ == "__main__":
    main()
