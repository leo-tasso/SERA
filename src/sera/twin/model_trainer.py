"""Train sklearn models for each indicator based on annual parameters and lagged indicators."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IndicatorModel:
    """Sklearn-based model for a single indicator."""

    def __init__(
        self,
        indicator_name: str,
        model_type: str = "ridge",
        random_state: int = 42,
    ):
        """Initialize indicator model.

        Args:
            indicator_name: Name of the indicator to predict
            model_type: 'ridge' or 'random_forest'
            random_state: Random seed for reproducibility
        """
        self.indicator_name = indicator_name
        self.model_type = model_type
        self.random_state = random_state

        # Initialize model
        if model_type == "ridge":
            self.model = Ridge(alpha=1.0, random_state=random_state)
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2,
    ) -> Dict[str, float]:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Names of features
            test_size: Fraction for test split

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = feature_names

        # Split data
        X_trn, X_tst, y_trn, y_tst = train_test_split(
            X_train, y_train, test_size=test_size, random_state=self.random_state
        )

        # Scale features
        X_trn_scaled = self.scaler.fit_transform(X_trn)
        X_tst_scaled = self.scaler.transform(X_tst)

        # Train model
        self.model.fit(X_trn_scaled, y_trn)
        self.is_trained = True

        # Evaluate
        y_pred_trn = self.model.predict(X_trn_scaled)
        y_pred_tst = self.model.predict(X_tst_scaled)

        metrics = {
            "r2_train": r2_score(y_trn, y_pred_trn),
            "r2_test": r2_score(y_tst, y_pred_tst),
            "mae_train": mean_absolute_error(y_trn, y_pred_trn),
            "mae_test": mean_absolute_error(y_tst, y_pred_tst),
            "train_samples": len(X_trn),
            "test_samples": len(X_tst),
        }

        logger.info(
            f"{self.indicator_name}: R2_test={metrics['r2_test']:.3f}, "
            f"MAE_test={metrics['mae_test']:.3f}"
        )

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise RuntimeError(f"Model for {self.indicator_name} not trained yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            return {}

        if hasattr(self.model, "coef_"):
            # Ridge: use absolute coefficients
            importance = np.abs(self.model.coef_)
        elif hasattr(self.model, "feature_importances_"):
            # Random Forest
            importance = self.model.feature_importances_
        else:
            return {}

        return dict(zip(self.feature_names, importance))


class ModelTrainer:
    """Train and manage models for all indicators."""

    def __init__(
        self,
        model_type: str = "ridge",
        random_state: int = 42,
    ):
        """Initialize trainer.

        Args:
            model_type: 'ridge' or 'random_forest'
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.models: Dict[str, IndicatorModel] = {}

    def save(self, file_path: str | Path) -> None:
        """Save the trained trainer and all indicator models to disk."""
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str | Path) -> "ModelTrainer":
        """Load a previously saved trainer from disk."""
        loaded = joblib.load(file_path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Expected {cls.__name__} in {file_path}")
        return loaded

    def prepare_feature_matrix(
        self,
        indicator_name: str,
        indicators_df: pd.DataFrame,
        parameters_df: pd.DataFrame,
        lag_years: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and targets for a single indicator.

        Features include:
        - Lagged values of the indicator
        - Lagged values of dependent indicators
        - Current year values of all parameters

        Args:
            indicator_name: Name of indicator to predict
            indicators_df: DataFrame with all indicators (area_code, year, indicator_cols)
            parameters_df: DataFrame with all parameters (area_code, year, parameter_cols)
            lag_years: Number of years to lag

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Start with parameters only
        X_data = parameters_df.copy()

        # Add lagged indicator values
        for lag in range(1, lag_years + 1):
            lag_indicators = indicators_df[["area_code", "year", indicator_name]].copy()
            lag_indicators["year"] = lag_indicators["year"] + lag
            lag_indicators = lag_indicators.rename(
                columns={indicator_name: f"{indicator_name}_lag{lag}"}
            )
            X_data = X_data.merge(lag_indicators, on=["area_code", "year"], how="inner")

        # Get target (current year's indicator value)
        y_data = indicators_df[["area_code", "year", indicator_name]].copy()

        # Merge
        data = X_data.merge(y_data, on=["area_code", "year"], how="inner")

        # Get feature columns
        feature_cols = [
            col for col in data.columns if col not in ["area_code", "year", indicator_name]
        ]

        # Remove features that are completely missing for this indicator window.
        # This is common when an indicator has a shorter historical span than some parameters.
        feature_cols = [col for col in feature_cols if data[col].notna().any()]
        if not feature_cols:
            logger.warning(
                f"No usable feature columns for {indicator_name} after filtering all-NaN features"
            )
            return np.array([]), np.array([]), []

        # Drop rows where target is NaN OR where ALL features are NaN
        # (Allow sparse parameters - just require at least some data per row)
        data = data[data[indicator_name].notna()]  # Target must not be NaN

        # For each row, check if there's at least one non-NaN feature
        non_nan_feature_count = data[feature_cols].notna().sum(axis=1)
        data = data[non_nan_feature_count > 0]  # At least one feature must be non-NaN

        # Some features can become entirely NaN after the target/time filtering above.
        # Remove them before imputation so they don't force all rows to be dropped.
        feature_cols = [col for col in feature_cols if data[col].notna().any()]
        if not feature_cols:
            logger.warning(f"No usable feature columns for {indicator_name} after row filtering")
            return np.array([]), np.array([]), []

        if len(data) < 10:
            logger.warning(f"Not enough data for {indicator_name}: only {len(data)} samples")
            return np.array([]), np.array([]), []

        # Impute NaN features with column mean (simple imputation)
        for col in feature_cols:
            if data[col].isna().any():
                mean_val = data[col].mean()
                if not np.isnan(mean_val):
                    data[col] = data[col].fillna(mean_val)

        # Drop any rows that still have NaN in the remaining features.
        data = data.dropna(subset=feature_cols + [indicator_name])

        if len(data) < 10:
            logger.warning(
                f"Not enough data for {indicator_name} after imputation: only {len(data)} samples"
            )
            return np.array([]), np.array([]), []

        # Extract features and target
        X = data[feature_cols].values
        y = data[indicator_name].values

        return X, y, feature_cols

    def train_indicator(
        self,
        indicator_name: str,
        indicators_df: pd.DataFrame,
        parameters_df: pd.DataFrame,
        test_size: float = 0.2,
        lag_years: int = 1,
    ) -> Dict[str, float]:
        """Train a model for a single indicator.

        Args:
            indicator_name: Name of indicator
            indicators_df: DataFrame with indicators
            parameters_df: DataFrame with parameters
            test_size: Test set fraction
            lag_years: Number of years to lag

        Returns:
            Training metrics
        """
        X, y, feature_names = self.prepare_feature_matrix(
            indicator_name, indicators_df, parameters_df, lag_years
        )

        if len(X) == 0:
            return {}

        model = IndicatorModel(indicator_name, self.model_type, self.random_state)
        metrics = model.train(X, y, feature_names, test_size)
        self.models[indicator_name] = model

        return metrics

    def train_all_indicators(
        self,
        indicators_df: pd.DataFrame,
        parameters_df: pd.DataFrame,
        test_size: float = 0.2,
        lag_years: int = 1,
    ) -> Dict[str, Dict[str, float]]:
        """Train models for all indicators.

        Args:
            indicators_df: DataFrame with all indicators
            parameters_df: DataFrame with all parameters
            test_size: Test set fraction
            lag_years: Number of years to lag

        Returns:
            Dictionary mapping indicator names to training metrics
        """
        indicator_cols = [col for col in indicators_df.columns if col not in ["area_code", "year"]]

        all_metrics = {}
        for indicator_name in indicator_cols:
            logger.info(f"Training model for {indicator_name}...")
            metrics = self.train_indicator(
                indicator_name,
                indicators_df,
                parameters_df,
                test_size,
                lag_years,
            )
            all_metrics[indicator_name] = metrics

        logger.info(f"Trained {len(self.models)} models")
        return all_metrics

    def get_model(self, indicator_name: str) -> Optional[IndicatorModel]:
        """Get a trained model.

        Args:
            indicator_name: Name of indicator

        Returns:
            Trained model or None if not found
        """
        return self.models.get(indicator_name)
