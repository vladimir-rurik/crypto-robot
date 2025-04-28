import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
# Must enable experimental before import
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

from .feature_engineering import FeatureEngineer

class DataPreprocessor:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        print("[DEBUG]     clean_data() -> handling missing values...")
        data = self._handle_missing_values(data)
        print("[DEBUG]     clean_data() -> removing outliers...")
        data = self._remove_outliers(data)
        return data
    
    ## faster version of the function
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        # Just do forward/backward fill
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        return data

    ## takes a lot of time to run. 
    ## TODO use  RAPIDS cuML library for faster computation on GPU
    # def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
    #     data = data.copy()
    #     missing_pct = data.isnull().sum() / len(data)
        
    #     # Print some stats
    #     print(f"[DEBUG]       -> {missing_pct[missing_pct > 0].shape[0]} columns have missing data.")

    #     fully_missing = missing_pct[missing_pct == 1.0].index
    #     for col in fully_missing:
    #         print(f"[DEBUG]       -> Column '{col}' is fully missing; filling with synthetic data.")
    #         data[col] = np.random.normal(0, 1, size=len(data))

    #     high_missing = missing_pct[(missing_pct > 0.3) & (missing_pct < 1.0)].index
    #     low_missing  = missing_pct[(missing_pct > 0) & (missing_pct <= 0.3)].index

    #     if len(high_missing) > 0:
    #         print(f"[DEBUG]       -> Using IterativeImputer for high-missing columns: {list(high_missing)}")
    #         imp_iter = IterativeImputer(random_state=42)
    #         data[high_missing] = imp_iter.fit_transform(data[high_missing])

    #     if len(low_missing) > 0:
    #         print(f"[DEBUG]       -> Using KNNImputer for low-missing columns: {list(low_missing)}")
    #         imp_knn = KNNImputer(n_neighbors=5)
    #         data[low_missing] = imp_knn.fit_transform(data[low_missing])

    #     data.ffill(inplace=True)
    #     data.bfill(inplace=True)
    #     return data

    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        scaler = RobustScaler()
        cols_to_scale = ["Open", "High", "Low", "Close", "Volume"]

        print("[DEBUG]       -> Removing outliers with RobustScaler threshold=3")
        for col in cols_to_scale:
            if col not in data.columns:
                continue
            data[col] = pd.to_numeric(data[col], errors="coerce")
            scaled = scaler.fit_transform(data[col].values.reshape(-1, 1))
            mask_outliers = (abs(scaled) > 3)
            outlier_count = mask_outliers.sum()
            if outlier_count > 0:
                print(f"[DEBUG]       -> Found {outlier_count} outliers in column '{col}'. Setting to NaN.")
            data.loc[mask_outliers.ravel(), col] = np.nan

        print("[DEBUG]       -> Re-imputing after outlier removal")
        data = self._handle_missing_values(data)
        return data

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        print("[DEBUG]     engineer_features()")
        return self.feature_engineer.process_features(data)
