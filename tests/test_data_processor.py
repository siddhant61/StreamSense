import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from data_processor import DataProcessor


def test_detect_gaps_marks_time_gap_and_mutates_dataframe():
    timestamps = pd.to_datetime([0, 1, 4, 5], unit='s')
    df = pd.DataFrame({'signal': [1.0, 1.1, 1.2, 1.3]}, index=timestamps)
    processor = DataProcessor(folder_path="/tmp")

    gaps = processor.detect_gaps(df, gap_threshold=2, nan_sequence_threshold=2)

    assert gaps.tolist() == [False, False, True, False]
    assert df.iloc[2].isna().all()


def test_detect_gaps_flags_long_nan_sequences():
    timestamps = pd.RangeIndex(start=0, stop=6, step=1)
    df = pd.DataFrame({'signal': [1.0, np.nan, np.nan, np.nan, 2.0, 3.0]}, index=timestamps)
    processor = DataProcessor(folder_path="/tmp")

    gaps = processor.detect_gaps(df, gap_threshold=2, nan_sequence_threshold=2)

    expected = [False, True, True, True, False, False]
    assert gaps.tolist() == expected
    assert df['signal'].isna().tolist() == [False, True, True, True, False, False]
