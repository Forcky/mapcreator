"""The track stage's shipped text files come out byte for byte the same on Windows and Linux."""

import inspect
import re

from mapcreator.pipeline import track_stage


def test_crlf_text_is_written_as_given(tmp_path):
    path = tmp_path / "surfaces.ini"
    track_stage._write_text(path, "[SURFACE_0]\r\nKEY=ROAD\r\n")
    # Without newline="", Windows writes each CRLF as CR CR LF (found in M3, in every INI and ATTRIBUTION.txt).
    assert path.read_bytes() == b"[SURFACE_0]\r\nKEY=ROAD\r\n"


def test_every_text_file_goes_through_the_helper():
    # CI runs on Linux, where a plain write_text can't show the bug, so guard the source instead.
    source = inspect.getsource(track_stage)
    helper = inspect.getsource(track_stage._write_text)
    calls = re.findall(r"\.write_text\(", source.replace(helper, ""))
    assert calls == []
