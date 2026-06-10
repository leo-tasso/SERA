"""Main simulation engine for the digital twin."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from sera.twin.causal_graph import (
    PARAMETER_TO_INDICATORS,
    INDICATOR_TO_INDICATORS,
    INDICATOR_BOUNDS,
    PARAMETER_EFFECT_DIRECTION,
    INDICATOR_EFFECT_DIRECTION,
    get_parameter_signal,
    get_parameter_reference,
    get_indicator_bounds,
)
from sera.twin.model_trainer import ModelTrainer, IndicatorModel

logger = logging.getLogger(__name__)

# Realism speed limit: the largest plausible year-over-year change for any
# indicator. Without it, the multiplicative causal rules and inter-indicator
# propagation compound across a long horizon and let an optimiser drive GDP to
# absurd values. At baseline policy levers, annual changes are far below this,
# so it only constrains aggressively-pushed scenarios.
DEFAULT_MAX_ANNUAL_GROWTH = 0.06

# The trained indicator models are anchored to the previous-year value and used
# only for the *relative* effect of policy vs. baseline policy (their absolute
# level is not trusted), capped to this band. Empirically the models' relative
# response saturates the cap for almost any lever deviation, so this cap is the
# de-facto size of the ML layer's contribution and must stay small: at large
# values (e.g. 0.5) every intervention — regardless of which levers move — hits
# the realism growth limit, erasing all trade-offs between policies.
POLICY_SIGNAL_CAP = 0.03

# Strength of the hand-written causal rules (max ±2% per lever per year, before
# dampening). NOTE: policy levers deliberately act on indicators through TWO
# layers that compound — the trained models' policy signal above AND these
# explicit causal rules. The models capture the (often weak) statistical
# sensitivity present in the historical data; the rules add a guaranteed
# domain-knowledge elasticity so every documented lever→indicator link responds
# even where the data is too noisy for the models to learn it. Together with
# POLICY_SIGNAL_CAP this is calibrated so that no single lever family can push
# a province to the growth cap on its own — policies must make real choices.
CAUSAL_RULE_STRENGTH = 0.02


class DigitalTwinSimulator:
    """Simulate provincial socioeconomic indicators one year forward."""

    def __init__(
        self,
        trainer: ModelTrainer,
        indicators: List[str],
        parameters: List[str],
        causal_rule_strength: float = CAUSAL_RULE_STRENGTH,
        policy_signal_cap: float = POLICY_SIGNAL_CAP,
    ):
        """Initialize simulator.

        Args:
            trainer: Trained ModelTrainer instance
            indicators: List of indicator names in the system
            parameters: List of parameter names in the system
            causal_rule_strength: Strength of the hand-written causal rules.
                Exposed so sensitivity analysis can re-run a scenario under
                weaker/stronger assumptions and report how much the results
                depend on this hand-tuned constant.
            policy_signal_cap: Cap on the trained models' relative policy
                response per year (prediction under chosen levers vs. baseline
                levers). Exposed for the same calibration/sensitivity reasons.
        """
        self.trainer = trainer
        self.indicators = indicators
        self.parameters = parameters
        self.causal_rule_strength = float(causal_rule_strength)
        self.policy_signal_cap = float(policy_signal_cap)
        self.simulation_history: List[Dict] = []

    def simulate_year(
        self,
        current_state: pd.DataFrame,  # area_code, year, and all indicator columns
        parameters_year: pd.DataFrame,  # area_code, year, and parameter columns
        apply_rules: bool = True,
        apply_bounds: bool = True,
        max_annual_growth: Optional[float] = DEFAULT_MAX_ANNUAL_GROWTH,
    ) -> pd.DataFrame:
        """Simulate one year forward.
        
        Args:
            current_state: DataFrame with current indicator values for each province
                          (area_code, year, indicator_1, indicator_2, ...)
            parameters_year: DataFrame with annual parameters for this year
                            (area_code, year, param_1, param_2, ...)
            apply_rules: Whether to apply causal rules
            apply_bounds: Whether to apply logical bounds to indicators
            
        Returns:
            DataFrame with next-year indicator values
        """
        result = current_state[["area_code"]].copy()
        result["year"] = current_state["year"].max() + 1
        
        # Get lagged indicator values
        current_year = current_state["year"].iloc[0]
        lagged_indicators = current_state[[col for col in current_state.columns 
                                          if col not in ["area_code", "year"]]].copy()
        lagged_indicators = lagged_indicators.add_suffix("_lag1")
        
        parameter_cols = [
            col for col in parameters_year.columns if col not in ["area_code", "year"]
        ]
        # Align parameter rows to the state's provinces explicitly: the frames
        # are combined positionally below, so any difference in row order or
        # province coverage would silently assign levers to the wrong province.
        aligned_parameters = current_state[["area_code"]].merge(
            parameters_year.drop(columns=["year"], errors="ignore"),
            on="area_code",
            how="left",
        )
        for col in parameter_cols:
            if aligned_parameters[col].isna().any():
                aligned_parameters[col] = aligned_parameters[col].fillna(
                    get_parameter_reference(col)[0]
                )
        parameters_only = aligned_parameters[parameter_cols].reset_index(drop=True)

        # Combine lagged indicators with parameters
        features = pd.concat([
            current_state[["area_code"]].reset_index(drop=True),
            lagged_indicators.reset_index(drop=True),
            parameters_only,
        ], axis=1)

        # A twin feature frame with every lever held at its historical baseline.
        # Comparing predictions against this isolates the *policy* effect and
        # cancels the model's absolute-level mis-calibration.
        baseline_parameters = parameters_only.copy()
        for col in parameter_cols:
            baseline_parameters[col] = get_parameter_reference(col)[0]
        features_baseline = pd.concat([
            current_state[["area_code"]].reset_index(drop=True),
            lagged_indicators.reset_index(drop=True),
            baseline_parameters,
        ], axis=1)

        # Predict each indicator
        predictions = {}
        for indicator in self.indicators:
            lagged_col = f"{indicator}_lag1"
            lag_values = (
                lagged_indicators[lagged_col].values
                if lagged_col in lagged_indicators.columns
                else None
            )

            model = self.trainer.get_model(indicator)
            if model is None:
                # No model trained for this indicator: hold at last value.
                logger.warning(f"No trained model for {indicator}, using lagged value")
                predictions[indicator] = (
                    lag_values if lag_values is not None else np.zeros(len(features))
                )
                continue

            # Prepare features for this model
            model_features = [f for f in model.feature_names if f in features.columns]
            if len(model_features) != len(model.feature_names):
                logger.warning(
                    f"Feature mismatch for {indicator}: "
                    f"have {len(model_features)}, expected {len(model.feature_names)}"
                )
                if lag_values is not None:
                    predictions[indicator] = lag_values
                else:
                    logger.warning(f"No lagged value available for {indicator}, using zeros")
                    predictions[indicator] = np.zeros(len(features))
                continue

            pred = model.predict(features[model_features].values)

            if lag_values is None:
                # No persistence anchor available: fall back to the raw prediction.
                predictions[indicator] = pred
                continue

            # Anchor to last year's value and apply only the model's relative
            # response to policy (prediction under chosen levers vs. baseline levers).
            pred_baseline = model.predict(features_baseline[model_features].values)
            denom = np.abs(pred_baseline) + 1e-6
            policy_signal = np.clip(
                (pred - pred_baseline) / denom,
                -self.policy_signal_cap,
                self.policy_signal_cap,
            )
            predictions[indicator] = lag_values * (1.0 + policy_signal)

        # Create results dataframe
        for indicator, pred in predictions.items():
            result[indicator] = pred
        
        # Apply causal rules (on the aligned parameters: same row order as result)
        if apply_rules:
            result = self._apply_causal_rules(
                result, aligned_parameters, lagged_indicators
            )
        
        # Apply bounds
        if apply_bounds:
            result = self._apply_bounds(result)
        
        # Propagate inter-indicator effects
        result = self._propagate_indicator_effects(result, lagged_indicators)

        if apply_bounds:
            result = self._apply_bounds(result)

        # Cap implausible year-over-year swings so policy effects can't compound
        # into runaway growth over a long horizon.
        if max_annual_growth is not None and max_annual_growth > 0:
            result = self._apply_growth_limits(result, lagged_indicators, max_annual_growth)
            if apply_bounds:
                result = self._apply_bounds(result)

        return result

    def simulate_scenario(
        self,
        initial_state: pd.DataFrame,  # year, area_code, all indicators
        parameters_path: List[pd.DataFrame],  # sequence of year parameter dfs
        apply_rules: bool = True,
    ) -> pd.DataFrame:
        """Simulate a multi-year scenario.
        
        Args:
            initial_state: Starting indicator values (year, area_code, indicators)
            parameters_path: List of parameter DataFrames for each year
            apply_rules: Whether to apply causal rules
            
        Returns:
            DataFrame with all years of simulation (concatenated)
        """
        current_state = initial_state.copy()
        all_results = [current_state]
        
        for year_idx, params_df in enumerate(parameters_path):
            logger.info(f"Simulating year {year_idx + 1}...")
            next_state = self.simulate_year(current_state, params_df, apply_rules)
            all_results.append(next_state)
            current_state = next_state
        
        result = pd.concat(all_results, ignore_index=True)
        return result.sort_values(["area_code", "year"])

    def _get_nonlinear_dampening(
        self,
        current_value: np.ndarray,
        indicator: str,
        base_multiplier: float = 0.04,
    ) -> np.ndarray:
        """Compute non-linear dampening multiplier that decreases near bounds.
        
        As indicators approach their realistic maximum, policy effects diminish.
        This creates diminishing returns: easier to improve from 70→75 life exp
        than from 80→85.
        
        Args:
            current_value: Current indicator values
            indicator: Indicator name (to get bounds)
            base_multiplier: Base effect strength (0.04)
            
        Returns:
            Dampened multiplier array where effects weaken near bounds
        """
        lower_bound, upper_bound = get_indicator_bounds(indicator)
        
        # Handle infinite bounds (no dampening)
        if np.isinf(upper_bound):
            return np.full_like(current_value, base_multiplier, dtype=float)
        
        # Normalize current value to [0, 1] within realistic range
        # Lower bound = 0 (strong effect), Upper bound = 1 (weak effect)
        normalized = (current_value - lower_bound) / (upper_bound - lower_bound)
        normalized = np.clip(normalized, 0, 1)
        
        # Quadratic dampening: effect decreases as value rises
        # At 0%: multiplier = 1.0x base
        # At 50%: multiplier = 0.75x base
        # At 100%: multiplier ~0x base (nearly impossible to improve)
        dampening_factor = 1 - (normalized ** 1.5)  # Slightly softer than ^2
        
        return base_multiplier * dampening_factor

    def _apply_causal_rules(
        self,
        predictions: pd.DataFrame,
        parameters: pd.DataFrame,
        lagged_indicators: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply causal rules to adjust predictions based on parameters.

        This is the second, intentional layer of policy response on top of the
        models' policy signal (see ``CAUSAL_RULE_STRENGTH``): it encodes the
        causal-graph links directly so levers always have a plausible effect.

        Args:
            predictions: DataFrame with model predictions
            parameters: DataFrame with annual parameters
            lagged_indicators: DataFrame with lagged indicator values

        Returns:
            DataFrame with rule-adjusted predictions
        """
        result = predictions.copy()
        
        # For each parameter, adjust affected indicators
        for parameter in self.parameters:
            if parameter not in parameters.columns:
                continue
            
            affected_indicators = PARAMETER_TO_INDICATORS.get(parameter, [])
            param_direction = PARAMETER_EFFECT_DIRECTION.get(parameter, 0)
            
            if not affected_indicators or param_direction == 0:
                continue
            
            param_values = parameters[parameter].values
            param_signal = np.array(
                [get_parameter_signal(parameter, value) for value in param_values]
            )
            
            for indicator in affected_indicators:
                if indicator not in result.columns:
                    continue
                
                indicator_direction = INDICATOR_EFFECT_DIRECTION.get(indicator, 0)
                if indicator_direction == 0:
                    continue
                
                # Effect direction: multiply param and indicator directions
                # If same sign: higher parameter helps the indicator
                # If opposite sign: higher parameter hurts the indicator
                effect_sign = param_direction * indicator_direction
                
                # Get non-linear dampening based on current value
                # As indicator approaches upper bound, effect weakens
                current_values = result[indicator].values
                base_adjustment = effect_sign * param_signal * self.causal_rule_strength
                
                # Apply non-linear dampening
                dampening = self._get_nonlinear_dampening(
                    current_values, indicator, base_multiplier=1.0
                )
                adjustment_factor = base_adjustment * dampening
                
                result[indicator] = result[indicator] * (1 + adjustment_factor)
        
        return result

    def _apply_bounds(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply logical bounds to indicator values.
        
        Args:
            predictions: DataFrame with predictions
            
        Returns:
            DataFrame with bounded predictions
        """
        result = predictions.copy()
        
        for indicator in self.indicators:
            if indicator not in result.columns:
                continue
            
            bounds = INDICATOR_BOUNDS.get(indicator)
            if bounds is None:
                continue
            
            lower, upper = bounds
            result[indicator] = np.clip(result[indicator], lower, upper)

        return result

    def _apply_growth_limits(
        self,
        predictions: pd.DataFrame,
        lagged_indicators: pd.DataFrame,
        max_annual_growth: float,
    ) -> pd.DataFrame:
        """Smoothly limit each indicator's year-over-year change to ±max_annual_growth.

        A realism speed limit: no socioeconomic indicator can plausibly jump by a
        large fraction in a single year. This prevents the multiplicative causal
        rules and propagation from compounding into runaway values (e.g. GDP
        exploding when an optimiser pushes the levers to their extremes).

        The change is squashed through ``tanh`` rather than hard-clipped: it
        asymptotes to ±max_annual_growth but always keeps a non-zero slope. A
        hard clip would flatten the objective landscape (every aggressive policy
        lands on the exact cap), leaving a policy optimiser with no gradient to
        follow; the smooth version preserves that gradient.
        """
        result = predictions.copy()

        for indicator in self.indicators:
            if indicator not in result.columns:
                continue

            lagged_col = f"{indicator}_lag1"
            if lagged_col not in lagged_indicators.columns:
                continue

            lag = lagged_indicators[lagged_col].values.astype(float)
            current = result[indicator].values.astype(float)

            # Only constrain where there is a usable previous value.
            mask = np.isfinite(lag) & (np.abs(lag) > 1e-9) & np.isfinite(current)

            relative_change = np.zeros_like(current)
            relative_change[mask] = (current[mask] - lag[mask]) / np.abs(lag[mask])
            squashed = max_annual_growth * np.tanh(relative_change / max_annual_growth)
            limited = lag * (1.0 + squashed)

            result[indicator] = np.where(mask, limited, current)

        return result

    def _propagate_indicator_effects(
        self,
        predictions: pd.DataFrame,
        lagged_indicators: pd.DataFrame,
    ) -> pd.DataFrame:
        """Propagate inter-indicator effects through the system.
        
        Args:
            predictions: DataFrame with predictions
            lagged_indicators: DataFrame with lagged values
            
        Returns:
            DataFrame with propagated effects
        """
        result = predictions.copy()
        
        # For each indicator, check if it has changed significantly
        for source_indicator in self.indicators:
            if source_indicator not in result.columns:
                continue
            
            lagged_col = f"{source_indicator}_lag1"
            if lagged_col not in lagged_indicators.columns:
                continue
            
            lagged_values = lagged_indicators[lagged_col].values
            predicted_values = result[source_indicator].values
            
            # Calculate relative change
            with np.errstate(divide="ignore", invalid="ignore"):
                relative_change = (predicted_values - lagged_values) / (
                    np.abs(lagged_values) + 1e-6
                )

            relative_change = np.tanh(relative_change)
            
            # Get dependent indicators
            dependent_indicators = INDICATOR_TO_INDICATORS.get(source_indicator, [])
            
            for target_indicator in dependent_indicators:
                if target_indicator not in result.columns:
                    continue
                
                # Propagate effect: multiply dependent indicator change by source change
                # Dampened to avoid explosive behavior
                propagation_factor = 0.12
                result[target_indicator] = (
                    result[target_indicator]
                    * (1 + relative_change * propagation_factor)
                )
        
        return result

    def get_provincial_rankings(
        self, results: pd.DataFrame, year: int, indicator: str
    ) -> pd.DataFrame:
        """Get provincial rankings for an indicator in a specific year.
        
        Args:
            results: Simulation results
            year: Year to rank
            indicator: Indicator to rank
            
        Returns:
            DataFrame with rankings (area_code, value, rank)
        """
        year_data = results[results["year"] == year][["area_code", indicator]].copy()
        year_data.columns = ["area_code", "value"]
        year_data["rank"] = year_data["value"].rank(ascending=False)
        return year_data.sort_values("rank")

    def get_convergence_divergence(
        self,
        results: pd.DataFrame,
        indicator: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Dict[str, float]:
        """Analyze convergence/divergence of an indicator across provinces.
        
        Args:
            results: Simulation results
            indicator: Indicator to analyze
            start_year: Starting year (if None, uses first year)
            end_year: Ending year (if None, uses last year)
            
        Returns:
            Dictionary with convergence metrics
        """
        if start_year is None:
            start_year = results["year"].min()
        if end_year is None:
            end_year = results["year"].max()
        
        start_data = results[results["year"] == start_year][indicator]
        end_data = results[results["year"] == end_year][indicator]
        
        if start_data.empty or end_data.empty:
            return {}
        
        metrics = {
            "start_year": start_year,
            "end_year": end_year,
            "start_std": float(start_data.std()),
            "end_std": float(end_data.std()),
            "start_mean": float(start_data.mean()),
            "end_mean": float(end_data.mean()),
            "start_cv": float(start_data.std() / (start_data.mean() + 1e-6)),
            "end_cv": float(end_data.std() / (end_data.mean() + 1e-6)),
            "convergence": start_data.std() > end_data.std(),
        }
        
        return metrics
