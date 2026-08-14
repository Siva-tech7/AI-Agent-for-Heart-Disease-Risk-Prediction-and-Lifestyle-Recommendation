import pandas as pd
import numpy as np
from src.preprocessing import build_preprocessor

def test_build_preprocessor_fit_transform():
    df = pd.DataFrame({
        'age': [50, 60],
        'sex': [1, 0],
        'cp': [0, 1]
    })
    preprocessor = build_preprocessor(df)
    preprocessor.fit(df)
    transformed = preprocessor.transform(df)
    # The output should have scaled age, imputed sex (binary pipeline), and one-hot encoded cp.
    assert transformed is not None
    assert transformed.shape[0] == 2
