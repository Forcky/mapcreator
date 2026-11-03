"""The parts of the in-game harness that can be checked without the game."""

import configparser
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import ac_harness  # noqa: E402

MESSY = """[RACE]
TRACK=monza
MODEL=ks_ferrari
CARS=12
[REMOTE]
ACTIVE=1
SERVER_IP=203.0.113.4
[REPLAY]
ACTIVE=1
[CAR_0]
MODEL=-
SKIN=red
SETUP=old
_EXT_SETUP_FILENAME=x.ini
[CAR_1]
MODEL=ks_bmw
[SESSION_0]
NAME=Practice
[SESSION_1]
NAME=Race
[CONDITION_0]
X=1
[EVENT]
NAME=weekend
[WEATHER]
NAME=3_clear
"""


def test_race_ini_becomes_a_solo_practice_but_keeps_the_players_settings(tmp_path, monkeypatch):
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "race.ini").write_text(MESSY, encoding="utf-8")
    monkeypatch.setattr(ac_harness, "ac_docs", lambda: tmp_path)
    monkeypatch.setattr(ac_harness, "_first_skin", lambda car: "yellow")
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read_string(ac_harness._patched_race_ini("grasstree_hill", "forward", "ks_mazda_mx5_cup"))
    assert not {"CAR_1", "SESSION_1", "CONDITION_0", "EVENT"} & set(cp.sections())
    for mode in ("REMOTE", "REPLAY", "BENCHMARK", "RESTART"):
        assert cp[mode]["ACTIVE"] == "0"  # not an online join, a replay, a benchmark or a restart
    assert cp["CAR_0"]["SKIN"] == "yellow" and cp["CAR_0"]["SETUP"] == "" and "_EXT_SETUP_FILENAME" not in cp["CAR_0"]
    assert cp["RACE"]["TRACK"] == "grasstree_hill" and cp["RACE"]["CARS"] == "1"
    assert cp["WEATHER"]["NAME"] == "3_clear"


def test_no_race_ini_is_a_clear_error_not_an_empty_session(tmp_path, monkeypatch):
    monkeypatch.setattr(ac_harness, "ac_docs", lambda: tmp_path)
    with pytest.raises(SystemExit, match="start any session"):
        ac_harness._patched_race_ini("t", "", "car")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows known folders")
def test_documents_comes_from_windows_not_a_guess():
    docs = ac_harness._documents()
    assert docs.is_dir() and docs.name.lower().endswith("documents")


# Trimmed from the M3 logs (CSP 0.2.11): no "LOADING MODEL" line, and CSP marks failures with 💥.
AC_LOG = """[SESSION_0] TYPE: 1
Creating physics track: grasstree_hill [forward]
DEBUG TIMER: TrackAvatar::init3D():285.438400
STARTING PHYSICS ENGINE
ERROR: NO STAT FOR grasstree_hill
"""
CSP_LOG = """   ef9     M ✅ Config for grasstree_hill found: E:/AC/content/tracks/grasstree_hill/extension/ext_config.ini
  2557     M 💥 Failed to load trees: Tree manifest is missing: E:/AC/content/tracks/grasstree_hill/extension/trees/x
   205     M 💬 Setting chunks for: grasstree_hill, nodes: 312
"""


def test_the_log_check_reads_a_csp_load_and_catches_a_tree_failure(tmp_path, monkeypatch, capsys):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "log.txt").write_text(AC_LOG, encoding="utf-8")
    (tmp_path / "logs" / "custom_shaders_patch.log").write_text(CSP_LOG, encoding="utf-8")
    monkeypatch.setattr(ac_harness, "ac_docs", lambda: tmp_path)
    assert ac_harness.check_log("grasstree_hill") == 1  # the trees didn't load
    out = capsys.readouterr().out
    assert "track model loaded: True" in out and "physics started: True" in out
    assert "errors naming the track: 0" in out  # NO STAT is AC's lap-record cache, not a track error
    assert "CSP log errors naming the track or its config: 1" in out and "Tree manifest is missing" in out
    assert "CSP model nodes: grasstree_hill, nodes: 312" in out
    (tmp_path / "logs" / "custom_shaders_patch.log").write_text(
        "\n".join(ln for ln in CSP_LOG.splitlines() if "Failed" not in ln), encoding="utf-8")
    assert ac_harness.check_log("grasstree_hill") == 0
    capsys.readouterr()
    ac_harness.check_log("grasstree")  # an id that's only a prefix of the loaded one
    assert "track model loaded: False" in capsys.readouterr().out


def test_probe_points_keep_a_car_length_off_both_ends(tmp_path, monkeypatch):
    import json

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mapcreator.core import ai_line

    n = 301  # a straight 300 m line, a point every metre
    pos = np.column_stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)])
    extra = np.zeros((n, 18))
    extra[:, 13] = 1.0  # forward along +x
    (tmp_path / "track" / "ai").mkdir(parents=True)
    ai_line.write(tmp_path / "track" / "ai" / "fast_lane.ai",
                  ai_line.AiLine(pos=pos, length=np.arange(n, dtype=float), extra=extra))
    monkeypatch.setattr(ac_harness, "ac_docs", lambda: tmp_path / "docs")
    ac_harness.probe_points(tmp_path / "track", "", 100.0, False, 0)
    pts = json.loads((tmp_path / "docs" / "mapcreator_probe" / "points.json").read_text())["points"]
    s = [p["s_m"] for p in pts]
    # A car centred at s = 0 has its rear wheels off the end of the road (Esperance coast road, M3).
    assert s[0] >= ac_harness.PROBE_END_MARGIN_M and s[-1] <= 300 - ac_harness.PROBE_END_MARGIN_M
    assert s == [5.0, 105.0, 205.0]
