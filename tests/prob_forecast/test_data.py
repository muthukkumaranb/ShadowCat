import unittest
import pandas as pd
import numpy as np
from ml1.prob_forecast.data import load_data, process_fold, get_valid_windows

class TestData(unittest.TestCase):
    def test_leakage(self):
        # tests/prob_forecast/test_data.py checks that no statistic is computed with test indices.
        df = pd.DataFrame(np.random.rand(100, 35), columns=[f'f{i}' for i in range(35)])
        df['episode_id'] = np.random.randint(0, 5, 100)
        df['window_start_utc'] = pd.date_range('2023-01-01', periods=100, freq='min')
        features = [f'f{i}' for i in range(35)]
        
        train_indices = list(range(50))
        test_indices = list(range(50, 100))
        
        train_out, test_out, scaler, pca, valid_features = process_fold(df, features, train_indices, test_indices)
        
        # Test refit on train only
        df_train_only = df.iloc[train_indices].copy()
        train_out2, test_out2, scaler2, pca2, valid_features2 = process_fold(df_train_only, features, train_indices, [])
        
        # Assert scaler fits are equal
        np.testing.assert_array_almost_equal(scaler.mean_, scaler2.mean_)
        np.testing.assert_array_almost_equal(scaler.scale_, scaler2.scale_)
        
        # Assert PCA fits are equal (will be up to sign for eigenvectors but components should be the same space)
        # Using the explained variance as a proxy
        np.testing.assert_array_almost_equal(pca.explained_variance_, pca2.explained_variance_)

if __name__ == '__main__':
    unittest.main()
