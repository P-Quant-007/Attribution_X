import pandas as pd

df = pd.read_csv('data/processed_trades.csv')
df['date'] = pd.to_datetime(df['date'])

# DHFL YTM trajectory
dhfl = df[df['stress_tag'] == 'DHFL'].copy()
dhfl_monthly = dhfl.groupby(dhfl['date'].dt.to_period('M'))['ytm'].mean().round(2)
print('=== DHFL Monthly Avg YTM ===')
print(dhfl_monthly[dhfl_monthly.index >= '2017-01'].to_string())

# Data coverage
print()
daily_counts = df.groupby('date').size()
print(f'Trading days in dataset: {len(daily_counts)}')
print(f'Avg trades per day:      {daily_counts.mean():.0f}')
print(f'Min trades in a day:     {daily_counts.min()} on {daily_counts.idxmin()}')
print(f'Max trades in a day:     {daily_counts.max()} on {daily_counts.idxmax()}')