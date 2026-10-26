import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_portable.py"
spec = importlib.util.spec_from_file_location("build_portable", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_portable_package_contains_current_sources_and_replaces_latest(tmp_path):
    latest = tmp_path / "ab_testing_app_portable.zip"
    latest.write_bytes(b"old stale package")
    result = builder.build_package(output=tmp_path)
    assert latest.read_bytes() == result.read_bytes()
    with ZipFile(result) as archive:
        names = archive.namelist()
        assert "ab_testing_app/start_windows.bat" in names
        assert "ab_testing_app/scripts/check_installation.py" in names
        assert "ab_testing_app/src/experiment_platform/customer_ui.py" in names
        assert "ab_testing_app/tests/test_app_portability.py" in names
        assert not any(part in name for name in names for part in ["/.venv/", "/.git/", "/.experiment_portfolio.json", "/.env", "/__pycache__/"])
        assert archive.read("ab_testing_app/app.py") == (builder.ROOT / "app.py").read_bytes()
        manifest = json.loads(archive.read("ab_testing_app/BUILD_INFO.json"))
        assert len(manifest["build_id"]) == 12
        for relative, digest in manifest["files"].items():
            import hashlib
            assert hashlib.sha256(archive.read("ab_testing_app/" + relative)).hexdigest() == digest


def test_package_blocks_old_windows_date_format(tmp_path):
    for name in builder.TOP_LEVEL:
        (tmp_path / name).write_text("placeholder")
    (tmp_path / "app.py").write_text('date.strftime("%b %-d, %Y")')
    with pytest.raises(ValueError, match="platform-specific"):
        builder.build_package(root=tmp_path, output=tmp_path / "dist")
