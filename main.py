# Stock Market Analysis Project - Enhanced
# Tools: Python, Pandas, Matplotlib, Seaborn
# Author: [Your Name]
# Date: [Current Date]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ======================
# 1. Configuration
# ======================
# Create output directories
os.makedirs('images', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Set global styles
sns.set(style="whitegrid")  # Apply seaborn style
sns.set_palette('colorblind')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 7)})

# ======================
# 2. Data Loading & Preparation
# ======================
try:
    # Load dataset (use relative path for GitHub)
    # df = pd.read_csv("data/stocks.csv")  
    df = pd.read_csv("C:/Users/KHUSHBOO/OneDrive/Desktop/stock-analysis/data/stocks.csv")

    # Convert date with format inference
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

    # Sort by ticker and date
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Handle missing values with groupby + transform (avoids apply deprecation)
    df = df.groupby('Ticker').transform(lambda group: group.ffill().bfill())
    # The above returns only transformed columns; merge back non-numeric columns separately:
    # So better to do like this instead:

except FileNotFoundError:
    print("Error: Data file not found. Please check the path.")
    exit()

# Reload the data to fix groupby.apply issue properly:
# Because transform returns only numeric columns and drops non-numeric columns like 'Date' and 'Ticker'

# Correct missing value filling without dropping columns:
df = pd.read_csv("C:/Users/KHUSHBOO/OneDrive/Desktop/stock-analysis/data/stocks.csv")
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Fill missing values per ticker without using apply (avoiding deprecation warning):
def fill_na(group):
    return group.ffill().bfill()

df.update(df.groupby('Ticker').transform(fill_na))

# Add new features
df['Day'] = df['Date'].dt.day_name()
df['Month'] = df['Date'].dt.month_name()
df['Year'] = df['Date'].dt.year

# Calculate Daily_Return here so it is available for EDA and later steps
df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change() * 100

print("\n=== Missing Values After Handling ===")
print(df.isnull().sum())

print("\n=== Data Overview ===")
print(f"Time Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Tickers: {', '.join(df['Ticker'].unique())}")
print(f"Total Records: {len(df):,}")

# ======================
# 3. Visualization Utilities
# ======================
def save_fig(plt, name):
    """Save plot to file with standardized formatting"""
    plt.tight_layout()
    plt.savefig(f'docs/images/{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ======================
# 4. Exploratory Data Analysis
# ======================
def perform_eda(df):
    """Perform exploratory data analysis and generate visualizations"""
    print("\n=== Performing EDA ===")

    # 1. Basic statistics
    stats = df.groupby('Ticker')['Close'].describe()
    print("\n=== Statistics ===")
    print(stats)

    # Save statistics to CSV
    stats.to_csv('data/stock_statistics.csv')

    # 2. Closing price distribution
    plt.figure()
    sns.histplot(data=df, x='Close', hue='Ticker', element='step', bins=30, kde=True)
    plt.title('Closing Price Distribution by Ticker')
    plt.xlabel('Closing Price ($)')
    save_fig(plt, 'price_distribution')

    # 3. Price trends over time
    plt.figure()
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker]
        plt.plot(ticker_data['Date'], ticker_data['Close'], label=ticker, linewidth=2)

    plt.title('Stock Price Trends')
    plt.xlabel('Date')
    plt.ylabel('Closing Price ($)')
    plt.legend()
    save_fig(plt, 'price_trends')

    # 4. Daily returns distribution
    plt.figure()
    sns.boxplot(data=df, x='Ticker', y='Daily_Return')
    plt.title('Daily Returns Distribution')
    plt.ylabel('Daily Return (%)')
    save_fig(plt, 'returns_distribution')

    # 5. Volume analysis
    plt.figure()
    sns.barplot(data=df, x='Ticker', y='Volume', estimator=np.mean)
    plt.title('Average Trading Volume by Stock')
    plt.ylabel('Average Volume')
    save_fig(plt, 'average_volume')

    # 6. Weekly patterns
    if 'Day' in df.columns:
        plt.figure()
        sns.lineplot(data=df, x='Day', y='Close', hue='Ticker',
                     estimator='mean', errorbar=None, sort=False)
        plt.title('Average Closing Price by Day of Week')
        plt.ylabel('Average Price ($)')
        save_fig(plt, 'weekly_patterns')

# ======================
# 5. Technical Analysis
# ======================


def technical_analysis(df):
    """Perform technical analysis and generate indicators"""
    print("\n=== Performing Technical Analysis ===")

    # Moving averages
    windows = [7, 20, 50]
    for window in windows:
        df[f'MA{window}'] = df.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(window).mean()
        )

    # Bollinger Bands (reusing MA20 from above)
    df['Std20'] = df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(20).std()
    )
    df['Upper_Band'] = df['MA20'] + (2 * df['Std20'])
    df['Lower_Band'] = df['MA20'] - (2 * df['Std20'])

    # RSI calculation remains same...
        # RSI calculation
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = df.groupby('Ticker')['Close'].transform(calculate_rsi)


    # Visualization with 'MA20' and 'MA50'
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker].set_index('Date')

        plt.figure(figsize=(14, 8))
        plt.plot(ticker_data['Close'], label='Close Price', color='black', alpha=0.7)
        plt.plot(ticker_data['MA20'], label='20-day MA', color='blue')
        plt.plot(ticker_data['MA50'], label='50-day MA', color='red')
        # Bollinger bands plotting...

        # Bollinger Bands
        plt.fill_between(ticker_data.index,
                         ticker_data['Upper_Band'],
                         ticker_data['Lower_Band'],
                         color='gray', alpha=0.2, label='Bollinger Bands')

        plt.title(f'{ticker} - Technical Analysis')
        plt.legend()
        save_fig(plt, f'{ticker}_technical_analysis')

        # RSI plot
        plt.figure()
        plt.plot(ticker_data['RSI'], label='RSI', color='purple')
        plt.axhline(70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(30, color='green', linestyle='--', alpha=0.5)
        plt.title(f'{ticker} - Relative Strength Index (RSI)')
        plt.ylim(0, 100)
        plt.legend()
        save_fig(plt, f'{ticker}_rsi')

    # Calculate and print volatility
    volatility = df.groupby('Ticker')['Daily_Return'].std() * np.sqrt(252)  # Annualized
    print("\n=== Volatility (Annualized) ===")
    print(volatility)

    return df

# ======================
# 6. Correlation Analysis
# ======================
def correlation_analysis(df):
    """Perform correlation analysis between stocks"""
    print("\n=== Performing Correlation Analysis ===")

    # Pivot to compare stocks
    pivot_df = df.pivot(index='Date', columns='Ticker', values='Close')

    # Correlation matrix
    corr_matrix = pivot_df.corr()
    plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Stock Price Correlations')
    save_fig(plt, 'correlation_matrix')

    # Correlation over time (rolling)
    plt.figure()
    rolling_corr = pivot_df[pivot_df.columns[:2]].rolling(window=30).corr().iloc[0::2, -1]
    rolling_corr.plot(title='30-Day Rolling Correlation')
    plt.ylabel('Correlation Coefficient')
    save_fig(plt, 'rolling_correlation')

    return corr_matrix

# ======================
# 7. Comparative Analysis
# ======================
def comparative_analysis(df):
    """Compare performance of different stocks"""
    print("\n=== Performing Comparative Analysis ===")

    # Pivot to compare stocks
    pivot_df = df.pivot(index='Date', columns='Ticker', values='Close')

    # Normalized price comparison
    pivot_df_normalized = pivot_df.apply(lambda x: x / x.iloc[0] * 100)

    plt.figure()
    for column in pivot_df_normalized.columns:
        plt.plot(pivot_df_normalized.index, pivot_df_normalized[column], label=column, linewidth=2)

    plt.title('Normalized Price Comparison (Base=100)')
    plt.ylabel('Normalized Price')
    plt.legend()
    save_fig(plt, 'normalized_comparison')

    # Cumulative returns comparison
    cumulative_returns = pivot_df.pct_change().add(1).cumprod().sub(1) * 100
    plt.figure()
    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column], label=column, linewidth=2)

    plt.title('Cumulative Returns Comparison')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    save_fig(plt, 'cumulative_returns')

    return pivot_df_normalized
    def generate_processed_data(df):
    """Create processed data CSV with key metrics"""
    processed_df = df.copy()
    
    # Calculate additional metrics
    processed_df['Daily_Return'] = processed_df.groupby('Ticker')['Close'].pct_change()
    processed_df['Cumulative_Return'] = processed_df.groupby('Ticker')['Daily_Return'].cumsum()
    
    # Select key columns
    keep_columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 
                   'Daily_Return', 'Cumulative_Return']
    return processed_df[keep_columns]

# Add this before your main execution
processed_data = generate_processed_data(df)
processed_data.to_csv('docs/processed_stock_data.csv', index=False)

# ======================
# 8. Main Execution
# ======================
if __name__ == "__main__":
    perform_eda(df)
    df = technical_analysis(df)
    corr_matrix = correlation_analysis(df)
    normalized_prices = comparative_analysis(df)

    print("\nAnalysis complete! All plots saved in 'images/' directory.")
