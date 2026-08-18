from pathlib import Path

from tools import release_audit


def test_release_audit_detects_duplicate_defs(tmp_path, monkeypatch):
    package = tmp_path / "src" / "tensoratlas"
    package.mkdir(parents=True)
    bad = package / "bad.py"
    bad.write_text("def repeated():\n    pass\n\ndef repeated():\n    pass\n")
    monkeypatch.setattr(release_audit, "ROOT", tmp_path)
    errors: list[str] = []
    release_audit.check_duplicate_defs(errors)
    assert any("duplicate top-level definition" in item for item in errors)


def test_release_audit_detects_generated_artifacts(tmp_path, monkeypatch):
    cache_dir = tmp_path / "src" / "tensoratlas" / "__pycache__"
    cache_dir.mkdir(parents=True)
    (cache_dir / "stale.pyc").write_bytes(b"x")
    monkeypatch.setattr(release_audit, "ROOT", tmp_path)
    errors: list[str] = []
    release_audit.check_generated_artifacts(errors)
    assert any("generated artifact" in item for item in errors)


def test_release_audit_detects_forbidden_text(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "bad.md").write_text("forbidden proprietary reference: " + "Transformed" + "Field" + "\n")
    monkeypatch.setattr(release_audit, "ROOT", tmp_path)
    monkeypatch.setattr(release_audit, "PUBLIC_DIRS", [docs])
    monkeypatch.setattr(release_audit, "ROOT_TEXT_FILES", [])
    monkeypatch.setattr(release_audit, "ALLOW_TEXT", set())
    errors: list[str] = []
    release_audit.check_text(errors)
    assert any("release text artifact" in item for item in errors)
