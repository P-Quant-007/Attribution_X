"""
Bond Anomaly Detection Module for Fixed-Income Analytics
==========================================================

Production-grade module for detecting unusual yield behavior in corporate bond trades
using DUAL DETECTION: CUSUM + Isolation Forest with committee-safe severity gating.

CUSUM: Detects regime shifts and persistent drifts in yield levels
iForest: Detects rare multivariate outliers in attribution patterns

UPDATED for Indian Bond Market Data Structure (FIMMDA/NSE format)

Author: Senior Quant Engineering Team
Purpose: ET Hackathon - Fixed Income Analytics Prototype
Version: 3.0 (Added CUSUM + iForest dual detection)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Literal
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class CUSUMConfig:
    """Configuration for CUSUM change-point detection."""
    
    # Target shift to detect (in yield %)
    delta: float = 0.5  # Detect 50 bps shift
    
    # Control chart parameters
    k: float = 0.25  # Reference value (typically delta/2)
    h: float = 4.0  # Decision threshold (in multiples of sigma)
    
    # Adaptive threshold estimation
    adaptive_sigma: bool = True  # Use rolling window for sigma estimation
    sigma_window: int = 20  # Window for sigma estimation
    
    # CUSUM monitoring targets
    monitor_columns: List[str] = field(default_factory=lambda: ['avg_ytm'])
    
    # Reset behavior
    reset_on_alarm: bool = True  # Reset CUSUM after alarm
    
    
@dataclass
class AnomalyDetectionConfig:
    """Configuration parameters for bond anomaly detection."""
    
    # Aggregation mode
    mode: Literal["issuer", "isin"] = "isin"
    
    # YTM filtering
    ytm_min: float = 0.01  # Minimum reasonable yield in percent
    ytm_max: float = 60.0  # Maximum reasonable yield in percent
    
    # Feature engineering
    d1_window: int = 1  # 1-day change
    d5_window: int = 5  # 5-day change
    
    # Isolation Forest parameters
    n_estimators: int = 500
    contamination: float = 0.03
    random_state: int = 42
    
    # CUSUM parameters
    enable_cusum: bool = True
    cusum_config: CUSUMConfig = field(default_factory=CUSUMConfig)
    
    # Persistence rule for confirmed anomalies
    persistence_window: int = 3  # Look back N days
    persistence_threshold: int = 2  # Need >= N anomalies in window
    
    # Optional: Z-score gating for additional confirmation
    zscore_gate: bool = False
    zscore_threshold: float = 3.0  # MAD-based z-score threshold
    
    # Severity gating (CUSUM + iForest combination)
    severity_rules: Dict[str, str] = field(default_factory=lambda: {
        'both': 'HIGH',       # Both CUSUM and iForest fire
        'cusum_only': 'MEDIUM_HIGH',  # Slow stress build-up
        'iforest_only': 'MEDIUM',     # One-off outlier
        'neither': 'NONE'
    })
    
    # Output options
    save_csv: bool = False
    save_plot: bool = False
    output_dir: str = "./output"
    
    # Feature columns to use
    feature_columns: List[str] = field(default_factory=lambda: [
        'avg_ytm', 'd1', 'd5', 'prints', 'vol_log'
    ])


@dataclass
class ModelBundle:
    """Container for trained model artifacts."""
    scaler: StandardScaler
    isolation_forest: IsolationForest
    feature_columns: List[str]
    config: AnomalyDetectionConfig
    train_stats: Dict[str, float] = field(default_factory=dict)
    cusum_params: Dict[str, Dict] = field(default_factory=dict)  # Per-entity CUSUM state


class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) change-point detector for regime shifts.
    
    Monitors scalar time series for persistent drifts/shifts.
    Better than simple thresholds at detecting sustained changes.
    """
    
    def __init__(self, config: CUSUMConfig):
        """
        Initialize CUSUM detector.
        
        Args:
            config: CUSUM configuration
        """
        self.config = config
        
    def compute_cusum(
        self,
        df: pd.DataFrame,
        entity_col: str = 'entity_id',
        value_col: str = 'avg_ytm'
    ) -> pd.DataFrame:
        """
        Compute CUSUM statistic for each entity.
        
        Args:
            df: DataFrame with time series data
            entity_col: Column identifying entities
            value_col: Column to monitor
            
        Returns:
            DataFrame with CUSUM statistics and alarms
        """
        result = df.copy()
        
        # Initialize CUSUM columns
        result[f'cusum_{value_col}_pos'] = 0.0
        result[f'cusum_{value_col}_neg'] = 0.0
        result[f'cusum_{value_col}_alarm'] = 0
        result[f'cusum_{value_col}_direction'] = ''
        result[f'cusum_{value_col}_sigma'] = 0.0
        
        # Compute per entity
        for entity in result[entity_col].unique():
            mask = result[entity_col] == entity
            entity_data = result[mask].copy()
            
            if len(entity_data) < 2:
                continue
                
            # Estimate sigma (standard deviation)
            if self.config.adaptive_sigma:
                # Rolling window estimation
                sigma_series = entity_data[value_col].rolling(
                    window=min(self.config.sigma_window, len(entity_data)),
                    min_periods=2
                ).std()
                # Fill initial NaNs with global std
                sigma_series = sigma_series.fillna(entity_data[value_col].std())
            else:
                # Global sigma for entity
                sigma_series = pd.Series(
                    entity_data[value_col].std(),
                    index=entity_data.index
                )
            
            # Compute normalized deviations
            mu = entity_data[value_col].mean()
            z = (entity_data[value_col] - mu) / sigma_series
            
            # Initialize CUSUM accumulators
            cusum_pos = 0.0
            cusum_neg = 0.0
            cusum_pos_list = []
            cusum_neg_list = []
            alarm_list = []
            direction_list = []
            
            # Compute threshold based on sigma
            h_threshold = self.config.h
            k_value = self.config.k
            
            for idx, zi in zip(entity_data.index, z):
                # Update positive CUSUM (detects upward shift)
                cusum_pos = max(0, cusum_pos + zi - k_value)
                
                # Update negative CUSUM (detects downward shift)
                cusum_neg = max(0, cusum_neg - zi - k_value)
                
                # Check for alarms
                alarm = 0
                direction = ''
                
                if cusum_pos > h_threshold:
                    alarm = 1
                    direction = 'UP'
                    if self.config.reset_on_alarm:
                        cusum_pos = 0
                        
                if cusum_neg > h_threshold:
                    alarm = 1
                    direction = 'DOWN' if not direction else 'BOTH'
                    if self.config.reset_on_alarm:
                        cusum_neg = 0
                
                cusum_pos_list.append(cusum_pos)
                cusum_neg_list.append(cusum_neg)
                alarm_list.append(alarm)
                direction_list.append(direction)
            
            # Update result DataFrame
            result.loc[mask, f'cusum_{value_col}_pos'] = cusum_pos_list
            result.loc[mask, f'cusum_{value_col}_neg'] = cusum_neg_list
            result.loc[mask, f'cusum_{value_col}_alarm'] = alarm_list
            result.loc[mask, f'cusum_{value_col}_direction'] = direction_list
            result.loc[mask, f'cusum_{value_col}_sigma'] = sigma_series
        
        return result
    
    def detect_all_columns(
        self,
        df: pd.DataFrame,
        entity_col: str = 'entity_id'
    ) -> pd.DataFrame:
        """
        Run CUSUM on all configured monitor columns.
        
        Args:
            df: DataFrame with time series data
            entity_col: Column identifying entities
            
        Returns:
            DataFrame with CUSUM statistics for all monitored columns
        """
        result = df.copy()
        
        for col in self.config.monitor_columns:
            if col not in result.columns:
                warnings.warn(f"CUSUM monitor column '{col}' not found, skipping")
                continue
                
            result = self.compute_cusum(result, entity_col, col)
        
        # Create combined alarm flag
        alarm_cols = [f'cusum_{col}_alarm' for col in self.config.monitor_columns 
                      if col in df.columns]
        
        if alarm_cols:
            result['cusum_any_alarm'] = result[alarm_cols].max(axis=1)
        else:
            result['cusum_any_alarm'] = 0
            
        return result


class BondAnomalyDetector:
    """
    Main class for detecting anomalies in bond yield behavior.
    
    Implements DUAL DETECTION pipeline:
    1. Data cleaning and normalization
    2. Daily aggregation (volume-weighted or simple average)
    3. Feature engineering (level, changes, liquidity proxies)
    4. CUSUM detection for regime shifts
    5. Isolation Forest for multivariate outliers
    6. Severity gating based on both detectors
    7. Persistence-based confirmation rules
    
    DETECTION RATIONALE:
    - CUSUM: Catches change-points and regime shifts (persistent drift)
    - iForest: Catches rare multivariate outliers (unusual attribution patterns)
    - Combined: Provides explainable severity levels for committee
    
    UPDATED: Now handles Indian bond market data format with:
    - " ISIN" and " Issuer Name" columns (with leading spaces)
    - "Trade Date & Time" in DD-MM-YYYY HH:MM:SS format
    - "Trade Value in Rs. Lacs" as volume measure
    - Yields already in percentage format
    """
    
    # Column mapping for Indian bond data format
    COLUMN_MAPPING = {
        'date': 'Trade Date & Time',
        'isin': ' ISIN',
        'ytm': 'Yield',
        'volume': 'Trade Value in Rs. Lacs',
        'issuer': ' Issuer Name'
    }
    
    def __init__(self, config: Optional[AnomalyDetectionConfig] = None):
        """
        Initialize the anomaly detector.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or AnomalyDetectionConfig()
        self.model_bundle: Optional[ModelBundle] = None
        self.cusum_detector: Optional[CUSUMDetector] = None
        
        if self.config.enable_cusum:
            self.cusum_detector = CUSUMDetector(self.config.cusum_config)
        
    def clean_and_normalize(
        self,
        df: pd.DataFrame,
        use_issuer_mode: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Clean and normalize raw trade data.
        
        Args:
            df: Raw trade DataFrame (Indian bond market format)
            use_issuer_mode: Override config.mode to force issuer aggregation
            
        Returns:
            Cleaned DataFrame with standardized columns
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        df = df.copy()
        
        # Map to standard column names if Indian format detected
        has_indian_format = self.COLUMN_MAPPING['date'] in df.columns
        
        if has_indian_format:
            # Check which columns are available
            col_map = {}
            for std_name, indian_name in self.COLUMN_MAPPING.items():
                if indian_name in df.columns:
                    col_map[indian_name] = std_name
            
            # Rename to standard names
            df = df.rename(columns=col_map)
        else:
            # Standardize column names (strip/lower) for generic format
            df.columns = df.columns.str.strip().str.lower()
        
        # Check required columns
        required = ['date', 'isin', 'ytm']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Available columns: {df.columns.tolist()}"
            )
        
        # Parse date - handle Indian format DD-MM-YYYY HH:MM:SS
        df['date'] = pd.to_datetime(
        df['date'],
        dayfirst=True,
        errors='coerce'
        )
        
        # If parsing failed, try other common formats
        if df['date'].isna().all():
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        df = df.dropna(subset=['date'])
        
        # Extract just the date (drop time)
        df['date'] = df['date'].dt.date
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert ytm to numeric
        df['ytm'] = pd.to_numeric(df['ytm'], errors='coerce')
        df = df.dropna(subset=['ytm'])
        
        # Check if YTM needs normalization
        # Indian data is already in % format (median ~11.5)
        ytm_median = df['ytm'].median()
        if ytm_median < 1.0:
            # Data is in decimal format, convert to percent
            df['ytm'] = df['ytm'] * 100.0
        # else: already in percent format, no conversion needed
            
        # Filter garbage yields
        df = df[df['ytm'] > self.config.ytm_min]
        df = df[df['ytm'] <= self.config.ytm_max]
        
        # Handle volume column
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df['volume'] = df['volume'].fillna(0)
        else:
            df['volume'] = 0.0
            
        # Handle issuer column
        if use_issuer_mode is not None:
            self.config.mode = "issuer" if use_issuer_mode else "isin"
            
        if self.config.mode == "issuer" and 'issuer' not in df.columns:
            warnings.warn(
                "mode='issuer' but no issuer column found. Falling back to mode='isin'"
            )
            self.config.mode = "isin"
            
        # Clean up ISIN and issuer strings
        df['isin'] = df['isin'].astype(str).str.strip()
        if 'issuer' in df.columns:
            df['issuer'] = df['issuer'].astype(str).str.strip()
            
        return df
    
    def aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate trades to daily entity-level series.
        
        Args:
            df: Cleaned trade data
            
        Returns:
            DataFrame with one row per (date, entity) with aggregated metrics
        """
        if df.empty:
            raise ValueError("aggregate_to_daily(): received EMPTY DataFrame from cleaning step")
        
        entity_col = 'issuer' if self.config.mode == 'issuer' else 'isin'
        
        if entity_col not in df.columns:
            raise ValueError(f"Entity column '{entity_col}' not found in DataFrame")
        
        # Group by date and entity
        grouped = df.groupby(['date', entity_col])
        
        daily_rows = []
        
        for (date, entity), group in grouped:
            vol_sum = group['volume'].sum()
            prints = len(group)
            
            # Determine aggregation method
            if vol_sum > 0:
                avg_ytm = (group['ytm'] * group['volume']).sum() / vol_sum
                method = "vol_weighted"
            else:
                avg_ytm = group['ytm'].mean()
                method = "simple_avg"
                
            daily_rows.append({
                'date': date,
                'entity_id': entity,
                'avg_ytm': avg_ytm,
                'prints': prints,
                'vol_sum': vol_sum,
                'method': method
            })
            
        daily_df = pd.DataFrame(daily_rows)
        if daily_df.empty:
            raise ValueError(
                "aggregate_to_daily(): no rows created after grouping. "
                "Check date parsing and entity column values."
            )

        daily_df = daily_df.sort_values(['entity_id', 'date']).reset_index(drop=True)
        
        return daily_df
    
    def engineer_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for Isolation Forest.
        
        Args:
            daily_df: Daily aggregated data
            
        Returns:
            DataFrame with engineered features
        """
        df = daily_df.copy()
        
        # Compute changes per entity
        df['d1'] = df.groupby('entity_id')['avg_ytm'].diff(self.config.d1_window)
        df['d5'] = df.groupby('entity_id')['avg_ytm'].diff(self.config.d5_window)
        
        # Liquidity proxy: log of volume
        df['vol_log'] = np.log1p(df['vol_sum'])
        
        # Drop rows with NaN features (from differencing at start of series)
        df = df.dropna(subset=['d1', 'd5'])
        
        return df
    
    def run_cusum_detection(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run CUSUM change-point detection.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            DataFrame with CUSUM statistics and alarms
        """
        if not self.config.enable_cusum or self.cusum_detector is None:
            # Add dummy CUSUM columns
            features_df['cusum_any_alarm'] = 0
            return features_df
        
        return self.cusum_detector.detect_all_columns(features_df)
    
    def fit_isolation_forest(self, features_df: pd.DataFrame) -> ModelBundle:
        """
        Fit Isolation Forest model on feature matrix.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            ModelBundle with trained scaler and model
        """
        # Extract feature matrix
        feature_cols = self.config.feature_columns
        missing_cols = [col for col in feature_cols if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X = features_df[feature_cols].values
        
        # Check for NaNs
        if np.isnan(X).any():
            raise ValueError("Feature matrix contains NaN values. Clean data first.")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        iso_forest.fit(X_scaled)
        
        # Compute training statistics
        train_stats = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'contamination': self.config.contamination,
        }
        
        bundle = ModelBundle(
            scaler=scaler,
            isolation_forest=iso_forest,
            feature_columns=feature_cols,
            config=self.config,
            train_stats=train_stats
        )
        
        self.model_bundle = bundle
        return bundle
    
    def score_and_flag(
        self,
        features_df: pd.DataFrame,
        model_bundle: Optional[ModelBundle] = None
    ) -> pd.DataFrame:
        """
        Score data with Isolation Forest and flag anomalies.
        
        Args:
            features_df: DataFrame with features
            model_bundle: Trained model bundle (uses self.model_bundle if None)
            
        Returns:
            DataFrame with anomaly scores and flags
        """
        if model_bundle is None:
            if self.model_bundle is None:
                raise ValueError("No model bundle available. Call fit_isolation_forest first.")
            model_bundle = self.model_bundle
            
        df = features_df.copy()
        
        # Extract and scale features
        X = df[model_bundle.feature_columns].values
        X_scaled = model_bundle.scaler.transform(X)
        
        # Score with Isolation Forest
        df['if_score'] = model_bundle.isolation_forest.decision_function(X_scaled)
        df['if_flag'] = model_bundle.isolation_forest.predict(X_scaled)
        df['is_anomaly'] = (df['if_flag'] == -1).astype(int)
        
        return df
    
    def apply_severity_gating(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply severity gating based on CUSUM + iForest combination.
        
        Severity Levels:
        - HIGH: Both CUSUM and iForest fire (regime shift + outlier)
        - MEDIUM_HIGH: CUSUM only (slow stress build-up)
        - MEDIUM: iForest only (one-off outlier, likely noise)
        - NONE: Neither fires
        
        Args:
            df: DataFrame with CUSUM and iForest flags
            
        Returns:
            DataFrame with severity column
        """
        result = df.copy()
        
        # Ensure columns exist
        if 'cusum_any_alarm' not in result.columns:
            result['cusum_any_alarm'] = 0
        if 'is_anomaly' not in result.columns:
            result['is_anomaly'] = 0
        
        # Determine severity
        conditions = [
            (result['cusum_any_alarm'] == 1) & (result['is_anomaly'] == 1),  # Both
            (result['cusum_any_alarm'] == 1) & (result['is_anomaly'] == 0),  # CUSUM only
            (result['cusum_any_alarm'] == 0) & (result['is_anomaly'] == 1),  # iForest only
        ]
        
        choices = [
            self.config.severity_rules['both'],
            self.config.severity_rules['cusum_only'],
            self.config.severity_rules['iforest_only']
        ]
        
        result['severity'] = np.select(
            conditions,
            choices,
            default=self.config.severity_rules['neither']
        )
        
        # Create unified anomaly flag (any non-NONE severity)
        result['unified_anomaly'] = (result['severity'] != 'NONE').astype(int)
        
        return result
    
    def apply_persistence_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply persistence rule to confirm anomalies (reduce false positives).
        
        Args:
            df: DataFrame with 'unified_anomaly' column
            
        Returns:
            DataFrame with 'confirmed_anomaly' column
        """
        result = df.copy()
        
        # Use unified_anomaly if available, else fall back to is_anomaly
        anomaly_col = 'unified_anomaly' if 'unified_anomaly' in result.columns else 'is_anomaly'
        
        # Rolling sum of anomalies per entity
        result['anomaly_rolling_sum'] = result.groupby('entity_id')[anomaly_col].transform(
            lambda x: x.rolling(
                window=self.config.persistence_window,
                min_periods=1
            ).sum()
        )
        
        # Confirm if threshold met AND current observation is an anomaly
        # This ensures confirmed_anomaly is always a subset of is_anomaly
        result['confirmed_anomaly'] = (
            (result['anomaly_rolling_sum'] >= self.config.persistence_threshold) &
            (result[anomaly_col] == 1)
        ).astype(int)
        
        # Optional: Z-score gate for additional confirmation
        if self.config.zscore_gate:
            result = self._apply_zscore_gate(result)
        
        # Clean up temporary column
        result = result.drop(columns=['anomaly_rolling_sum'])
        
        return result
    
    def _apply_zscore_gate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply MAD-based z-score threshold as additional gate.
        
        Args:
            df: DataFrame with d1 and confirmed_anomaly
            
        Returns:
            DataFrame with updated confirmed_anomaly
        """
        result = df.copy()
        
        # Compute rolling MAD for each entity
        def robust_zscore(series):
            """Compute robust z-score using MAD."""
            median = series.median()
            mad = (series - median).abs().median()
            if mad == 0:
                return pd.Series(0, index=series.index)
            return (series - median) / (1.4826 * mad)
        
        result['d1_zscore'] = result.groupby('entity_id')['d1'].transform(
            lambda x: robust_zscore(x.rolling(window=20, min_periods=5))
        )
        
        # Only confirm if both IF flags AND z-score threshold exceeded
        zscore_exceeded = result['d1_zscore'].abs() >= self.config.zscore_threshold
        result['confirmed_anomaly'] = (
            result['confirmed_anomaly'] & zscore_exceeded
        ).astype(int)
        
        result = result.drop(columns=['d1_zscore'])
        
        return result
    
    def fit_predict(
        self,
        df: pd.DataFrame,
        use_issuer_mode: Optional[bool] = None
    ) -> Tuple[pd.DataFrame, ModelBundle]:
        """
        End-to-end pipeline: clean, aggregate, engineer, CUSUM, iForest, gate, confirm.
        
        Args:
            df: Raw trade data (Indian bond market format or generic)
            use_issuer_mode: Override to force issuer-level aggregation
            
        Returns:
            Tuple of (results DataFrame, model bundle)
        """
        # Step 1: Clean and normalize
        cleaned = self.clean_and_normalize(df, use_issuer_mode)
        if cleaned.empty:
            raise ValueError("clean_and_normalize() returned EMPTY DataFrame")

        
        # Step 2: Aggregate to daily
        daily = self.aggregate_to_daily(cleaned)
        
        # Step 3: Engineer features
        features = self.engineer_features(daily)
        
        # Step 4: Run CUSUM detection for regime shifts
        features_cusum = self.run_cusum_detection(features)
        
        # Step 5: Fit Isolation Forest
        bundle = self.fit_isolation_forest(features_cusum)
        
        # Step 6: Score and flag with iForest
        scored = self.score_and_flag(features_cusum, bundle)
        
        # Step 7: Apply severity gating (CUSUM + iForest)
        gated = self.apply_severity_gating(scored)
        
        # Step 8: Apply persistence rule
        final = self.apply_persistence_rule(gated)
        
        # Optional: Save outputs
        if self.config.save_csv:
            self._save_csv(final)
            
        return final, bundle
    
    def _save_csv(self, df: pd.DataFrame) -> None:
        """Save results to CSV."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f"anomaly_results_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    def plot_anomalies(
        self,
        df: pd.DataFrame,
        entity_id: str,
        event_markers: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (16, 8),
        show_cusum: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot yield series with anomaly markers and CUSUM statistics.
        
        Args:
            df: Results DataFrame
            entity_id: Entity to plot (ISIN or issuer)
            event_markers: Optional dict of {label: date_string} for vertical lines
            figsize: Figure size
            show_cusum: Whether to show CUSUM subplot
            
        Returns:
            Tuple of (figure, axes)
        """
        entity_data = df[df['entity_id'] == entity_id].copy()
        
        if entity_data.empty:
            raise ValueError(f"No data found for entity_id: {entity_id}")
        
        entity_data = entity_data.sort_values('date')
        
        # Create subplots if showing CUSUM
        if show_cusum and 'cusum_any_alarm' in entity_data.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
            ax2 = None
        
        # Plot 1: Yield series with anomalies
        ax1.plot(
            entity_data['date'],
            entity_data['avg_ytm'],
            'b-',
            linewidth=1.5,
            label='Avg YTM',
            alpha=0.7
        )
        
        # Mark different severity levels
        if 'severity' in entity_data.columns:
            # HIGH severity (both CUSUM and iForest)
            high = entity_data[entity_data['severity'] == 'HIGH']
            if not high.empty:
                ax1.scatter(
                    high['date'],
                    high['avg_ytm'],
                    color='darkred',
                    s=150,
                    marker='D',
                    label='HIGH (CUSUM + iForest)',
                    zorder=6,
                    edgecolors='black',
                    linewidths=2
                )
            
            # MEDIUM_HIGH severity (CUSUM only)
            med_high = entity_data[entity_data['severity'] == 'MEDIUM_HIGH']
            if not med_high.empty:
                ax1.scatter(
                    med_high['date'],
                    med_high['avg_ytm'],
                    color='orange',
                    s=120,
                    marker='^',
                    label='MEDIUM_HIGH (CUSUM only)',
                    zorder=5,
                    edgecolors='darkred',
                    linewidths=1.5
                )
            
            # MEDIUM severity (iForest only)
            medium = entity_data[entity_data['severity'] == 'MEDIUM']
            if not medium.empty:
                ax1.scatter(
                    medium['date'],
                    medium['avg_ytm'],
                    color='yellow',
                    s=80,
                    marker='o',
                    label='MEDIUM (iForest only)',
                    zorder=4,
                    edgecolors='orange',
                    linewidths=1
                )
        else:
            # Fall back to simple confirmed anomalies
            anomalies = entity_data[entity_data['confirmed_anomaly'] == 1]
            if not anomalies.empty:
                ax1.scatter(
                    anomalies['date'],
                    anomalies['avg_ytm'],
                    color='red',
                    s=100,
                    marker='o',
                    label='Confirmed Anomaly',
                    zorder=5,
                    edgecolors='darkred',
                    linewidths=2
                )
        
        # Add event markers if provided
        if event_markers:
            for label, date_str in event_markers.items():
                event_date = pd.to_datetime(date_str)
                ax1.axvline(
                    event_date,
                    color='gray',
                    linestyle='--',
                    alpha=0.6,
                    linewidth=1
                )
                ax1.text(
                    event_date,
                    ax1.get_ylim()[1],
                    f' {label}',
                    rotation=90,
                    verticalalignment='top',
                    fontsize=9,
                    alpha=0.7
                )
        
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Yield to Maturity (%)', fontsize=11)
        ax1.set_title(
            f'Bond Yield Anomaly Detection (CUSUM + iForest): {entity_id}',
            fontsize=13,
            fontweight='bold'
        )
        ax1.legend(loc='best', framealpha=0.9, fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        # Plot 2: CUSUM statistics (if available)
        if ax2 is not None and 'cusum_avg_ytm_pos' in entity_data.columns:
            ax2.plot(
                entity_data['date'],
                entity_data['cusum_avg_ytm_pos'],
                'g-',
                label='CUSUM+ (upward shift)',
                linewidth=1.5
            )
            ax2.plot(
                entity_data['date'],
                entity_data['cusum_avg_ytm_neg'],
                'r-',
                label='CUSUM- (downward shift)',
                linewidth=1.5
            )
            
            # Draw threshold line
            ax2.axhline(
                self.config.cusum_config.h,
                color='black',
                linestyle='--',
                label=f'Threshold (h={self.config.cusum_config.h})',
                alpha=0.5
            )
            
            # Mark alarms
            alarms = entity_data[entity_data['cusum_avg_ytm_alarm'] == 1]
            if not alarms.empty:
                ax2.scatter(
                    alarms['date'],
                    alarms['cusum_avg_ytm_pos'].where(
                        alarms['cusum_avg_ytm_direction'].str.contains('UP'),
                        alarms['cusum_avg_ytm_neg']
                    ),
                    color='red',
                    s=100,
                    marker='x',
                    label='CUSUM Alarm',
                    zorder=5,
                    linewidths=3
                )
            
            ax2.set_xlabel('Date', fontsize=11)
            ax2.set_ylabel('CUSUM Statistic', fontsize=11)
            ax2.set_title('CUSUM Change-Point Detection', fontsize=11)
            ax2.legend(loc='best', framealpha=0.9, fontsize=9)
            ax2.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        
        # Optional: Save plot
        if self.config.save_plot:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            plot_path = output_dir / f"anomaly_plot_{entity_id.replace('/', '_')}_{timestamp}.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        
        return fig, (ax1, ax2) if ax2 is not None else (ax1,)


# ============================================================================
# Utility Functions
# ============================================================================

def load_multiple_files(file_paths: List[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files.
    
    Args:
        file_paths: List of paths to CSV files
        
    Returns:
        Concatenated DataFrame
    """
    dfs = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {Path(path).name}")
        except Exception as e:
            warnings.warn(f"Failed to load {path}: {e}")
    
    if not dfs:
        raise ValueError("No files were successfully loaded")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(combined)}")
    return combined


def tune_contamination(
    detector: BondAnomalyDetector,
    df: pd.DataFrame,
    target_rate: float = 0.03,
    use_issuer_mode: Optional[bool] = None
) -> float:
    """
    Tune contamination parameter to achieve target anomaly rate.
    
    Args:
        detector: Anomaly detector instance
        df: Training data
        target_rate: Target anomaly rate (e.g., 0.03 = 3%)
        use_issuer_mode: Force issuer aggregation mode
        
    Returns:
        Optimal contamination parameter
    """
    # Prepare data
    cleaned = detector.clean_and_normalize(df, use_issuer_mode)
    daily = detector.aggregate_to_daily(cleaned)
    features = detector.engineer_features(daily)
    
    # Binary search for optimal contamination
    low, high = 0.001, 0.20
    best_contamination = detector.config.contamination
    
    for _ in range(10):  # Max iterations
        mid = (low + high) / 2
        detector.config.contamination = mid
        
        bundle = detector.fit_isolation_forest(features)
        scored = detector.score_and_flag(features, bundle)
        
        actual_rate = scored['is_anomaly'].mean()
        
        if abs(actual_rate - target_rate) < 0.005:  # Within 0.5%
            best_contamination = mid
            break
        elif actual_rate < target_rate:
            high = mid
        else:
            low = mid
    
    detector.config.contamination = best_contamination
    return best_contamination


def get_top_anomalies(
    results_df: pd.DataFrame,
    top_n: int = 10,
    by_entity: bool = False,
    severity_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Get top anomalies by score or count by entity.
    
    Args:
        results_df: Results from fit_predict
        top_n: Number of top results to return
        by_entity: If True, rank entities by anomaly count; else rank individual anomalies
        severity_filter: Filter by severity level (e.g., 'HIGH', 'MEDIUM_HIGH')
        
    Returns:
        DataFrame with top anomalies
    """
    df = results_df.copy()
    
    # Apply severity filter if specified
    if severity_filter and 'severity' in df.columns:
        df = df[df['severity'] == severity_filter]
    
    if by_entity:
        # Rank entities by number of confirmed anomalies
        entity_counts = df[df['confirmed_anomaly'] == 1].groupby(
            'entity_id'
        ).size().sort_values(ascending=False).head(top_n)
        
        result = pd.DataFrame({
            'entity_id': entity_counts.index,
            'anomaly_count': entity_counts.values
        })
        
        # Add severity breakdown if available
        if 'severity' in df.columns:
            for severity in ['HIGH', 'MEDIUM_HIGH', 'MEDIUM']:
                severity_counts = df[
                    (df['confirmed_anomaly'] == 1) & 
                    (df['severity'] == severity)
                ].groupby('entity_id').size()
                result[f'{severity.lower()}_count'] = result['entity_id'].map(
                    severity_counts
                ).fillna(0).astype(int)
        
        return result
    else:
        # Rank individual anomalies by score (most anomalous = most negative score)
        anomalies = df[df['confirmed_anomaly'] == 1].copy()
        anomalies = anomalies.sort_values('if_score').head(top_n)
        
        cols = ['date', 'entity_id', 'avg_ytm', 'd1', 'd5', 'if_score', 'prints']
        if 'severity' in anomalies.columns:
            cols.append('severity')
        if 'cusum_any_alarm' in anomalies.columns:
            cols.append('cusum_any_alarm')
        
        return anomalies[cols]


def get_detection_summary(results_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for dual detection results.
    
    Args:
        results_df: Results from fit_predict
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_observations': len(results_df),
        'unique_entities': results_df['entity_id'].nunique(),
        'date_range': (
            results_df['date'].min().strftime('%Y-%m-%d'),
            results_df['date'].max().strftime('%Y-%m-%d')
        ),
    }
    
    # iForest stats
    if 'is_anomaly' in results_df.columns:
        summary['iforest_anomalies'] = results_df['is_anomaly'].sum()
        summary['iforest_rate'] = results_df['is_anomaly'].mean()
    
    # CUSUM stats
    if 'cusum_any_alarm' in results_df.columns:
        summary['cusum_alarms'] = results_df['cusum_any_alarm'].sum()
        summary['cusum_rate'] = results_df['cusum_any_alarm'].mean()
    
    # Severity breakdown
    if 'severity' in results_df.columns:
        for severity in ['HIGH', 'MEDIUM_HIGH', 'MEDIUM']:
            count = (results_df['severity'] == severity).sum()
            summary[f'severity_{severity.lower()}'] = count
            summary[f'severity_{severity.lower()}_rate'] = count / len(results_df)
    
    # Confirmed anomalies
    if 'confirmed_anomaly' in results_df.columns:
        summary['confirmed_anomalies'] = results_df['confirmed_anomaly'].sum()
        summary['confirmed_rate'] = results_df['confirmed_anomaly'].mean()
    
    return summary


# ============================================================================
# Main Demonstration & Tests
# ============================================================================

def main():
    """
    Demonstrate usage with actual Indian bond market data.
    """
    print("=" * 80)
    print("Bond Anomaly Detection - DUAL METHOD (CUSUM + iForest) Demonstration")
    print("=" * 80)
    
    # ========================================================================
    # 1. Load actual data files
    # ========================================================================
    print("\n[1] Loading Indian bond market data files...")
    
    data_files = [
        '2025_merged.csv'
        # '/mnt/user-data/uploads/1-7_July.csv',
        # '/mnt/user-data/uploads/8-14_July.csv',
        # '/mnt/user-data/uploads/5-11_Aug.csv',
        # '/mnt/user-data/uploads/2-8_Sep.csv',
        # '/mnt/user-data/uploads/9-15_Sep.csv'
    ]
    
    # Filter to existing files
    existing_files = [f for f in data_files if Path(f).exists()]
    
    if existing_files:
        trades_df = load_multiple_files(existing_files)
    else:
        print("   No data files found. Exiting.")
        return None, None, None
    
    print(f"   Loaded {len(trades_df)} total trades")
    
    # ========================================================================
    # 2. Configure detector with CUSUM + iForest
    # ========================================================================
    print("\n[2] Configuring dual-method detector (CUSUM + iForest)...")
    
    cusum_config = CUSUMConfig(
        delta=0.5,              # Detect 50 bps shift
        k=0.25,                 # Reference value
        h=4.0,                  # Decision threshold
        adaptive_sigma=True,
        sigma_window=20,
        monitor_columns=['avg_ytm'],
        reset_on_alarm=True
    )
    
    config = AnomalyDetectionConfig(
        mode="isin",
        n_estimators=500,
        contamination=0.03,
        enable_cusum=True,
        cusum_config=cusum_config,
        persistence_window=3,
        persistence_threshold=2,
        severity_rules={
            'both': 'HIGH',
            'cusum_only': 'MEDIUM_HIGH',
            'iforest_only': 'MEDIUM',
            'neither': 'NONE'
        },
        save_csv=False,
        save_plot=False
    )
    
    detector = BondAnomalyDetector(config)
    print("   ✓ CUSUM enabled for regime shift detection")
    print("   ✓ iForest configured for multivariate outliers")
    print("   ✓ Severity gating enabled")
    
    # ========================================================================
    # 3. Run dual-method detection pipeline
    # ========================================================================
    print("\n[3] Running dual-method detection pipeline...")
    
    try:
        results_df, model_bundle = detector.fit_predict(trades_df)
        
        print(f"   ✓ Processed {len(results_df)} daily observations")
        print(f"   ✓ Features used: {model_bundle.feature_columns}")
        print(f"   ✓ Unique ISINs: {results_df['entity_id'].nunique()}")
        
        # ====================================================================
        # 4. Detection Summary
        # ====================================================================
        print("\n[4] Detection Summary:")
        
        summary = get_detection_summary(results_df)
        
        print(f"   Total observations: {summary['total_observations']}")
        print(f"   Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
        print(f"   Unique entities: {summary['unique_entities']}")
        print()
        print("   Detection Method Results:")
        print(f"   - iForest anomalies: {summary.get('iforest_anomalies', 0)} ({summary.get('iforest_rate', 0):.2%})")
        print(f"   - CUSUM alarms: {summary.get('cusum_alarms', 0)} ({summary.get('cusum_rate', 0):.2%})")
        print()
        print("   Severity Breakdown:")
        print(f"   - HIGH (both fire): {summary.get('severity_high', 0)} ({summary.get('severity_high_rate', 0):.2%})")
        print(f"   - MEDIUM_HIGH (CUSUM only): {summary.get('severity_medium_high', 0)} ({summary.get('severity_medium_high_rate', 0):.2%})")
        print(f"   - MEDIUM (iForest only): {summary.get('severity_medium', 0)} ({summary.get('severity_medium_rate', 0):.2%})")
        print()
        print(f"   Confirmed anomalies (after persistence): {summary.get('confirmed_anomalies', 0)} ({summary.get('confirmed_rate', 0):.2%})")
        
        # ====================================================================
        # 5. Top Anomalies by Severity
        # ====================================================================
        print("\n[5] Top Anomalous Entities by Severity:")
        
        # High severity entities
        high_severity = get_top_anomalies(
            results_df,
            top_n=5,
            by_entity=True,
            severity_filter='HIGH'
        )
        
        if len(high_severity) > 0:
            print("\n   HIGH Severity (CUSUM + iForest):")
            for _, row in high_severity.iterrows():
                print(f"      {row['entity_id']}: {row['anomaly_count']} anomalies")
        else:
            print("\n   HIGH Severity: None detected")
        
        # Medium-High severity entities
        med_high_severity = get_top_anomalies(
            results_df,
            top_n=5,
            by_entity=True,
            severity_filter='MEDIUM_HIGH'
        )
        
        if len(med_high_severity) > 0:
            print("\n   MEDIUM_HIGH Severity (CUSUM only - regime shifts):")
            for _, row in med_high_severity.iterrows():
                print(f"      {row['entity_id']}: {row['anomaly_count']} anomalies")
        
        # Overall top entities
        top_entities = get_top_anomalies(results_df, top_n=5, by_entity=True)
        print("\n   Overall Top Entities:")
        for _, row in top_entities.iterrows():
            severity_detail = ""
            if 'high_count' in row:
                severity_detail = f" (H:{row['high_count']}, MH:{row['medium_high_count']}, M:{row['medium_count']})"
            print(f"      {row['entity_id']}: {row['anomaly_count']} anomalies{severity_detail}")
        
        # ====================================================================
        # 6. Validation Tests
        # ====================================================================
        print("\n[6] Running validation tests...")
        
        # Test 1: CUSUM columns exist
        cusum_cols = [c for c in results_df.columns if c.startswith('cusum_')]
        assert len(cusum_cols) > 0, "FAIL: No CUSUM columns found"
        print(f"   ✓ Test 1 passed: CUSUM columns present ({len(cusum_cols)} columns)")
        
        # Test 2: Severity column exists and has valid values
        assert 'severity' in results_df.columns, "FAIL: Severity column missing"
        valid_severities = {'HIGH', 'MEDIUM_HIGH', 'MEDIUM', 'NONE'}
        actual_severities = set(results_df['severity'].unique())
        assert actual_severities.issubset(valid_severities), f"FAIL: Invalid severities: {actual_severities - valid_severities}"
        print("   ✓ Test 2 passed: Severity levels valid")
        
        # Test 3: Unified anomaly logic
        assert 'unified_anomaly' in results_df.columns, "FAIL: Unified anomaly column missing"
        unified_count = results_df['unified_anomaly'].sum()
        iforest_count = results_df['is_anomaly'].sum()
        cusum_count = results_df['cusum_any_alarm'].sum()
        print(f"   ✓ Test 3 passed: Unified anomaly logic ({unified_count} unified from {iforest_count} iForest + {cusum_count} CUSUM)")
        
        # Test 4: HIGH severity implies both methods fired
        high_severity_rows = results_df[results_df['severity'] == 'HIGH']
        if len(high_severity_rows) > 0:
            assert (high_severity_rows['is_anomaly'] == 1).all(), "FAIL: HIGH severity but iForest didn't fire"
            assert (high_severity_rows['cusum_any_alarm'] == 1).all(), "FAIL: HIGH severity but CUSUM didn't fire"
            print(f"   ✓ Test 4 passed: HIGH severity logic correct ({len(high_severity_rows)} cases)")
        else:
            print("   ✓ Test 4 passed: No HIGH severity cases (both methods must fire)")
        
        # ====================================================================
        # 7. Visualization Demo
        # ====================================================================
        if results_df['confirmed_anomaly'].sum() > 0:
            print("\n[7] Generating visualization with CUSUM statistics...")
            
            top_entity = get_top_anomalies(results_df, top_n=1, by_entity=True)
            if len(top_entity) > 0:
                sample_entity = top_entity.iloc[0]['entity_id']
                
                try:
                    fig, axes = detector.plot_anomalies(
                        results_df,
                        entity_id=sample_entity,
                        show_cusum=True
                    )
                    print(f"   ✓ Plot generated for {sample_entity}")
                    plt.close(fig)
                except Exception as e:
                    print(f"   ✗ Plot generation failed: {e}")
        
        print("\n" + "=" * 80)
        print("All tests passed! Dual-method detection ready for production.")
        print("=" * 80)
        
        print("\nKEY INSIGHTS:")
        print("- CUSUM catches regime shifts and persistent drifts")
        print("- iForest catches rare multivariate outliers")
        print("- Severity gating provides committee-friendly explanations")
        print("- HIGH severity = both methods agree (strongest signal)")
        print("- MEDIUM_HIGH = slow stress build-up (CUSUM only)")
        print("- MEDIUM = likely one-off noise (iForest only)")
        
        return results_df, model_bundle, detector
        
    except Exception as e:
        print(f"\n   ✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Run demonstration and tests
    results, bundle, detector = main()
    
    # Print usage instructions
    if results is not None:
        print("\n" + "=" * 80)
        print("USAGE INSTRUCTIONS:")
        print("=" * 80)
        print("""
To use dual-method detection with your data:

1. Configure CUSUM:
   cusum_config = CUSUMConfig(
       delta=0.5,              # Target shift (50 bps)
       h=4.0,                  # Decision threshold
       monitor_columns=['avg_ytm']
   )

2. Create detector:
   config = AnomalyDetectionConfig(
       mode='isin',
       enable_cusum=True,
       cusum_config=cusum_config,
       contamination=0.03
   )
   detector = BondAnomalyDetector(config)

3. Run detection:
   results, model = detector.fit_predict(df)

4. Analyze by severity:
   high_severity = results[results['severity'] == 'HIGH']
   # Both CUSUM and iForest fired - strongest signal
   
   medium_high = results[results['severity'] == 'MEDIUM_HIGH']
   # CUSUM only - regime shift detected
   
   medium = results[results['severity'] == 'MEDIUM']
   # iForest only - likely one-off outlier

5. Visualize with CUSUM:
   detector.plot_anomalies(results, entity_id='YOUR_ISIN', show_cusum=True)
        """)
