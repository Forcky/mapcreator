"""The build version: the same for the same code whatever the line endings, different when the code changes."""

import shutil

from mapcreator import buildinfo


def test_the_version_ignores_line_endings_and_follows_the_code(tmp_path, monkeypatch):
    src = buildinfo.Path(buildinfo.__file__).parent
    copy = tmp_path / "mapcreator"
    for d in buildinfo.SOURCES:
        shutil.copytree(src / d, copy / d, ignore=shutil.ignore_patterns("__pycache__"))
    for m in buildinfo.MODULES:
        shutil.copy(src / m, copy / m)
    (copy / "buildinfo.py").write_text("", encoding="utf-8")

    def version_of(root):
        monkeypatch.setattr(buildinfo, "__file__", str(root / "buildinfo.py"))
        buildinfo.version.cache_clear()
        return buildinfo.version()

    base = version_of(copy)
    target = copy / "core" / "water.py"
    text = target.read_bytes().replace(b"\r\n", b"\n")
    target.write_bytes(text.replace(b"\n", b"\r\n"))  # a Windows checkout
    assert version_of(copy) == base
    target.write_bytes(text + b"\n# a change\n")
    assert version_of(copy) != base
    buildinfo.version.cache_clear()
    assert len(buildinfo.version()) == 12
