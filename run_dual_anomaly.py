import pandas as pd
from bond_anomaly_detector_v3_cusum import (
    BondAnomalyDetector,
    AnomalyDetectionConfig,
    CUSUMConfig
)

# 1. Load raw trade-level data (NSE/FIMMDA format)
df = pd.read_csv("2025_merged.csv", low_memory=False)

# 2. Configure CUSUM (optional but recommended)
cusum_config = CUSUMConfig(
    delta=0.5,                # 50 bps structural shift
    h=4.0,                    # Decision threshold
    monitor_columns=['avg_ytm']
)

# 3. Create anomaly detector
config = AnomalyDetectionConfig(
    mode='isin',
    enable_cusum=True,
    cusum_config=cusum_config,
    contamination=0.03,
    save_plot=True,
    output_dir="./output"
)


detector = BondAnomalyDetector(config)

# 4. Run end-to-end detection
results, model = detector.fit_predict(df)

# 5. Analyze anomalies by severity
high_severity = results[results['severity'] == 'HIGH']
medium_high  = results[results['severity'] == 'MEDIUM_HIGH']
medium       = results[results['severity'] == 'MEDIUM']

# 6. Visualize one ISIN
high = results[results['severity'] == 'HIGH']

if high.empty:
    print("No HIGH severity anomalies found.")
else:
    sample_isin = high['entity_id'].iloc[0]
    print("Plotting HIGH severity ISIN:", sample_isin)

    detector.plot_anomalies(
        results,
        entity_id=sample_isin,
        show_cusum=True
    )

import matplotlib.pyplot as plt
plt.show()
