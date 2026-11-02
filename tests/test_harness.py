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
