import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set visualization style
sns.set(style="whitegrid")

# Directory containing CSV files
data_dir = '../data'

# List all CSV files
csv_files = [f for f in os.listdir(data_dir) if f.endswith('_minutely_data.csv')]

# Function to detect outliers using IQR
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

# Function to perform EDA on a single file
def perform_eda(file_path, coin_name):
    # Load only necessary columns to optimize memory usage
    use_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv(file_path, usecols=use_cols, parse_dates=['timestamp'])

    # Filter data from 2020-10-16 onwards
    df = df[df['timestamp'] >= '2020-10-16']

    print(f"\n--- EDA for {coin_name} ---")
    print(f"Data Shape: {df.shape}")
    print("\nData Info:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values)

    # Remove rows with zero volume
    zero_volume_count = (df['volume'] == 0).sum()
    df = df[df['volume'] != 0]
    print(f"\nRemoved {zero_volume_count} rows with zero volume.")

    # Detect outliers in volume
    outliers = detect_outliers_iqr(df['volume'])
    outlier_count = outliers.sum()
    print(f"Identified {outlier_count} volume outliers.")
    df = df[~outliers]  # Remove outliers

    # Check for missing minutely timestamps
    expected_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='min')
    actual_timestamps = pd.Series(df['timestamp'].values)
    missing_timestamps = expected_range.difference(actual_timestamps)
    print(f"\nMissing Timestamps: {len(missing_timestamps)}")
    if len(missing_timestamps) > 0:
        print(missing_timestamps[:10])  # Display only the first 10 missing timestamps

    # Detect sudden price jumps (anomalies) with a 5% threshold
    df['price_change'] = df['close'].pct_change().abs()
    anomalies = df[df['price_change'] > 0.05]  # Adjusted threshold to 5%
    print(f"\nDetected {len(anomalies)} potential price anomalies (sudden jumps).")

    # Downsample data for faster plotting
    downsample_rate = 100  # Plot every 100th point
    df_downsampled = df.iloc[::downsample_rate, :]

    # Plot Closing Price Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(df_downsampled['timestamp'], df_downsampled['close'], label='Close Price')
    plt.scatter(anomalies['timestamp'], anomalies['close'], color='red', label='Anomalies', s=10)
    plt.title(f'{coin_name} Closing Price Over Time (Downsampled)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.show()

    # Rolling average for closing price (60-minute window)
    df['rolling_mean'] = df['close'].rolling(window=60).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='Close Price', alpha=0.5)
    plt.plot(df['timestamp'], df['rolling_mean'], label='60-Min Rolling Avg', color='orange')
    plt.title(f'{coin_name} Closing Price with Rolling Average')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Volume Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(df_downsampled['timestamp'], df_downsampled['volume'], color='orange', label='Volume')
    plt.title(f'{coin_name} Volume Over Time (Downsampled)')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid()
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df[['open', 'high', 'low', 'close', 'volume']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'{coin_name} Feature Correlation (From 2020-10-16)')
    plt.show()

    # Additional verbose insights
    print("\nAdditional Insights:")
    print(f"Maximum Closing Price: {df['close'].max()} on {df[df['close'] == df['close'].max()]['timestamp'].values[0]}")
    print(f"Minimum Closing Price: {df['close'].min()} on {df[df['close'] == df['close'].min()]['timestamp'].values[0]}")
    print(f"Average Volume (post-cleaning): {df['volume'].mean()}")

# Perform EDA on all CSV files
if __name__ == "__main__":
    for csv_file in csv_files:
        coin_name = csv_file.split('_')[0]
        file_path = os.path.join(data_dir, csv_file)
        perform_eda(file_path, coin_name)
