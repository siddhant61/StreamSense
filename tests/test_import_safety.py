"""Regression tests for P0 stabilization fixes.

These lock in two defects found in the 2026-05 audit:

* P0-2: ``data_processor`` executed work at *import* time against a hardcoded
  ``D:/Study Data/...`` path, crashing every importer (including pytest collection).
* P0-1: ``ui/streamsense_controller`` constructed ``StreamE4`` with keyword
  arguments (``device_id=``, ``output_path=``) that do not exist in
  ``StreamE4.__init__``, raising ``TypeError`` on every E4 connect from the UI.
"""

import importlib
import inspect

import pytest


def test_importing_data_processor_has_no_side_effects():
    """Importing data_processor must not run a pipeline or touch the filesystem."""
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")

    module = importlib.import_module("data_processor")
    # The class must be importable and the module-level run must be gone.
    assert hasattr(module, "DataProcessor")
    assert hasattr(module, "main"), "offline run should live behind a main() entry point"


def test_data_processor_constructs_without_listing_files():
    """The constructor must not require an existing folder (no eager os.listdir)."""
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")
    from data_processor import DataProcessor

    # Should not raise even for a path that does not exist.
    DataProcessor(folder_path="/nonexistent/path/for/test")


def test_stream_e4_constructor_signature_contract():
    """Lock the StreamE4 public signature the UI controller depends on.

    The controller calls StreamE4(e4=..., root_output_folder=..., synchronized_start_time=...).
    If these parameter names drift, the UI E4-connect path breaks again.
    """
    pytest.importorskip("pylsl")
    from streamer.stream_e4 import StreamE4

    params = list(inspect.signature(StreamE4.__init__).parameters)[1:]  # drop self
    assert params == ["e4", "root_output_folder", "synchronized_start_time"]
