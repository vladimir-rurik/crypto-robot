import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
sns.set(style="whitegrid")

# Directory containing CSV files
data_dir = '../data'

# List all CSV files
csv_files = [f for f in os.listdir(data_dir) if f.endswith('_hourly_data.csv')]

# Function to perform EDA on a single file
def perform_eda(file_path, coin_name):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])

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

    # Plot Closing Price Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['close'], label='Close Price')
    plt.title(f'{coin_name} Closing Price Over Time (From 2020-10-16)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Volume Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['volume'], color='orange', label='Volume')
    plt.title(f'{coin_name} Volume Over Time (From 2020-10-16)')
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
    print(f"Average Volume: {df['volume'].mean()}")

# Perform EDA on all CSV files
if __name__ == "__main__":
    for csv_file in csv_files:
        coin_name = csv_file.split('_')[0]
        file_path = os.path.join(data_dir, csv_file)
        perform_eda(file_path, coin_name)
