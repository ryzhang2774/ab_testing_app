"""Report exact installed versions and reproduce Home -> Time Series locally."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]


def check_installation(smoke_test: bool = False) -> bool:
    print("Project:", ROOT)
    print("Python:", sys.version.split()[0], "Executable:", sys.executable)
    print("System:", platform.platform())
    valid = sys.version_info[:2] == (3, 11)
    if not valid:
        print("ERROR: use Python 3.11 for this pinned dependency set.")
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" not in line:
            continue
        name, expected = line.strip().split("==", 1)
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "NOT INSTALLED"
        print(f"{name}: {actual} (required: {expected})")
        valid = valid and actual == expected
    manifest_path = ROOT / "BUILD_INFO.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print("Package build:", manifest["build_id"], "Built:", manifest["built_at_utc"])
        for relative, expected in manifest["files"].items():
            path = ROOT / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                print("ERROR: packaged file is missing or changed:", relative)
                valid = False
    else:
        print("Package build: source checkout (no BUILD_INFO.json)")
    if not valid:
        print("ERROR: installation differs from the package. Install its requirements before retrying.")
        return False
    if smoke_test:
        # Streamlit's app resolves imports relative to this project folder.
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        if app.exception:
            for error in app.exception:
                print("HOME ERROR:", error.message)
            return False
        app.button(key="home_choose_timeseries").click().run()
        if app.exception:
            for error in app.exception:
                print("TIME SERIES ERROR:", error.message)
            return False
        print("PASS: Home -> Time Series landing rendered without Python exceptions.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if check_installation(args.smoke_test) else 1)
