"""Build a versioned, source-only package for a fresh Windows installation."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = ("app.py", "requirements.txt", "README.md", "app.yaml", "start_windows.bat")


def package_files(root: Path) -> list[Path]:
    files = [root / name for name in TOP_LEVEL]
    theme = root / ".streamlit" / "config.toml"
    if theme.is_file():
        files.append(theme)
    for directory in ("src", "tests", "scripts"):
        files.extend((root / directory).rglob("*.py"))
    files.extend((root / "sample_data").rglob("*.csv"))
    files.extend((root / "sample_data").rglob("*.tsv"))
    # PDF generation must not depend on fonts installed on the Windows host.
    assets = root / "src" / "experiment_platform" / "assets"
    if assets.exists():
        files.extend(path for path in assets.rglob("*") if path.is_file() and path.suffix in {".ttf", ".txt", ".md"})
    guides = root / "docs" / "customer-v2"
    if guides.exists():
        files.extend(path for path in guides.rglob("*") if path.is_file() and path.suffix in {".md", ".png"})
    missing = [str(path.relative_to(root)) for path in files if not path.is_file()]
    if missing:
        raise ValueError("Required package files are missing: " + ", ".join(missing))
    return sorted(set(files))


def build_package(root: Path = ROOT, output: Path | None = None) -> Path:
    output = output or root / "dist"
    output.mkdir(parents=True, exist_ok=True)
    files = package_files(root)
    # A stale Unix-only directive must never enter a Windows package again.
    source = (root / "app.py").read_text(encoding="utf-8")
    if "%-d" in source or "%#d" in source:
        raise ValueError("app.py contains a platform-specific date format. Fix it before packaging.")
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    build_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    now = datetime.now(timezone.utc)
    manifest = {"build_id": build_id, "built_at_utc": now.isoformat(), "base_commit": commit,
                "note": "File hashes identify this exact package, including uncommitted source changes.",
                "files": hashes}
    versioned = output / f"ab_testing_app_windows_{now:%Y%m%d}_{build_id}.zip"
    temporary = versioned.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "ab_testing_app/" + path.relative_to(root).as_posix())
        archive.writestr("ab_testing_app/BUILD_INFO.json", json.dumps(manifest, indent=2) + "\n")
    temporary.replace(versioned)
    # Replace the old, easy-to-copy filename too; it must not retain old code.
    latest = output / "ab_testing_app_portable.zip"
    latest_temporary = latest.with_suffix(".zip.tmp")
    latest_temporary.write_bytes(versioned.read_bytes())
    latest_temporary.replace(latest)
    return versioned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build_package(output=args.output))
