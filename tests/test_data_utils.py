import pandas as pd
from src.data_utils import clean_dataset, detect_target_column

def test_clean_dataset_removes_duplicates():
    df = pd.DataFrame({
        'age': [50, 50, 60],
        'sex': [1, 1, 0],
        'target': [1, 1, 0]
    })
    stats = clean_dataset(df)
    assert stats["original_count"] == 3
    assert stats["unique_count"] == 2
    assert stats["duplicate_count"] == 1
    assert len(stats["df"]) == 2

def test_detect_target_column():
    df = pd.DataFrame({'age': [50], 'output': [1]})
    assert detect_target_column(df) == 'output'
