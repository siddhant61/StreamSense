"""Headless tests for the E4 offline importer (synthetic E4 Connect sessions)."""

import zipfile

import pytest

from importer.e4_import import (
    E4Signal, load_session, parse_signal_rows, parse_ibi_rows, parse_tags_rows,
)

START = 1551434400.0


# ----- row parsers ---------------------------------------------------------- #
def test_parse_single_channel_signal():
    rows = [["1551434400.000000"], ["64.000000"], ["-0.1"], ["0.2"], ["0.3"]]
    sig = parse_signal_rows("BVP", rows)
    assert sig.channels == 1
    assert sig.start_time == START
    assert sig.sample_rate == 64.0
    assert sig.samples == [-0.1, 0.2, 0.3]
    assert sig.n == 3


def test_parse_three_channel_acc():
    rows = [
        ["1551434400.0", "1551434400.0", "1551434400.0"],
        ["32.0", "32.0", "32.0"],
        ["-31", "-40", "49"], ["1", "2", "3"],
    ]
    sig = parse_signal_rows("ACC", rows)
    assert sig.channels == 3
    assert sig.sample_rate == 32.0
    assert sig.samples == [[-31.0, -40.0, 49.0], [1.0, 2.0, 3.0]]


def test_parse_signal_skips_blank_rows():
    rows = [["100.0"], ["4.0"], ["1.0"], [""], ["2.0"]]
    assert parse_signal_rows("EDA", rows).samples == [1.0, 2.0]


def test_parse_signal_too_short_raises():
    with pytest.raises(ValueError):
        parse_signal_rows("HR", [["100.0"]])


def test_parse_acc_ragged_row_raises():
    # A multi-channel row with too few columns must fail fast, not produce a short sample.
    rows = [["100", "100", "100"], ["32", "32", "32"], ["1", "2", "3"], ["4", "5"]]
    with pytest.raises(ValueError):
        parse_signal_rows("ACC", rows)


def test_parse_ibi_rows_to_absolute_times():
    rows = [["1551434400.0", "IBI"], ["1.5", "0.82"], ["3.0", "0.79"]]
    ibi = parse_ibi_rows(rows)
    assert ibi == [(START + 1.5, 0.82), (START + 3.0, 0.79)]


def test_parse_tags_rows():
    assert parse_tags_rows([["1551434412.0"], [""], ["1551434430.0"]]) == [
        1551434412.0, 1551434430.0]


# ----- E4Signal helpers ----------------------------------------------------- #
def test_signal_timestamps_and_duration():
    sig = E4Signal(name="EDA", start_time=START, sample_rate=4.0,
                   samples=[0.0, 1.0, 2.0, 3.0])
    assert sig.duration == 1.0
    assert sig.end_time == START + 1.0
    assert sig.timestamps() == [START, START + 0.25, START + 0.5, START + 0.75]


# ----- loaders -------------------------------------------------------------- #
def _write_session(folder):
    (folder / "BVP.csv").write_text("1551434400.0\n64.0\n-0.1\n0.2\n")
    (folder / "ACC.csv").write_text(
        "1551434400.0,1551434400.0,1551434400.0\n32.0,32.0,32.0\n-31,-40,49\n")
    (folder / "EDA.csv").write_text("1551434400.0\n4.0\n0.01\n0.02\n0.03\n")
    (folder / "IBI.csv").write_text("1551434400.0, IBI\n1.5, 0.82\n")
    (folder / "tags.csv").write_text("1551434412.0\n")


def test_load_session_from_directory(tmp_path):
    _write_session(tmp_path)
    s = load_session(str(tmp_path))
    assert set(s.signals) == {"BVP", "ACC", "EDA"}
    assert s.signals["ACC"].channels == 3
    assert s.signals["BVP"].sample_rate == 64.0
    assert s.ibi == [(START + 1.5, 0.82)]
    assert s.tags == [1551434412.0]
    assert s.start_time() == START


def test_load_session_from_zip(tmp_path):
    _write_session(tmp_path)
    zip_path = tmp_path / "session.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in ["BVP.csv", "ACC.csv", "EDA.csv", "IBI.csv", "tags.csv"]:
            zf.write(tmp_path / name, arcname=name)
    s = load_session(str(zip_path))
    assert set(s.signals) == {"BVP", "ACC", "EDA"}
    assert s.tags == [1551434412.0]


def test_load_session_summary_shape(tmp_path):
    _write_session(tmp_path)
    summary = load_session(str(tmp_path)).summary()
    assert summary["start_time"] == START
    assert summary["signals"]["EDA"]["sample_rate"] == 4.0
    assert summary["ibi_count"] == 1
    assert summary["tags"] == [1551434412.0]


def test_load_session_bad_path_raises():
    with pytest.raises(FileNotFoundError):
        load_session("/nonexistent/path/to/session")
