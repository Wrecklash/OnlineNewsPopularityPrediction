import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
print("Reading the Online News Popularity dataset...")
df = pd.read_csv('OnlineNewsPopularity.csv')

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Display basic information about the dataset
print("\n" + "="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

print("\n" + "="*50)
print("COLUMN NAMES")
print("="*50)
for i, col in enumerate(df.columns):
    print(f"{i:2d}. {col}")

print("\n" + "="*50)
print("FIRST 5 ROWS")
print("="*50)
print(df.head())

print("\n" + "="*50)
print("DATA TYPES")
print("="*50)
print(df.dtypes)

print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)
print(df.describe())

print("\n" + "="*50)
print("MISSING VALUES")
print("="*50)
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("No missing values found in the dataset!")
else:
    print(missing_values[missing_values > 0])

print("\n" + "="*50)
print("TARGET VARIABLE ANALYSIS (shares)")
print("="*50)
shares_stats = df['shares'].describe()
print(shares_stats)

# Create binary classification based on threshold of 1400 shares
df['is_popular'] = (df['shares'] >= 1400).astype(int)
popular_count = df['is_popular'].sum()
unpopular_count = len(df) - popular_count

print(f"\nBinary Classification (threshold: 1400 shares):")
print(f"Popular articles (>= 1400 shares): {popular_count:,} ({popular_count/len(df)*100:.1f}%)")
print(f"Unpopular articles (< 1400 shares): {unpopular_count:,} ({unpopular_count/len(df)*100:.1f}%)")

print("\n" + "="*50)
print("DATA CHANNEL DISTRIBUTION")
print("="*50)
channel_columns = [col for col in df.columns if col.startswith('data_channel_is_')]
for col in channel_columns:
    channel_name = col.replace('data_channel_is_', '').title()
    count = df[col].sum()
    percentage = count / len(df) * 100
    print(f"{channel_name}: {count:,} articles ({percentage:.1f}%)")

print("\n" + "="*50)
print("WEEKDAY DISTRIBUTION")
print("="*50)
weekday_columns = [col for col in df.columns if col.startswith('weekday_is_') and col != 'weekday_is_weekend']
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for i, col in enumerate(weekday_columns):
    count = df[col].sum()
    percentage = count / len(df) * 100
    print(f"{weekday_names[i]}: {count:,} articles ({percentage:.1f}%)")

print("\n" + "="*50)
print("CORRELATION WITH TARGET VARIABLE")
print("="*50)
# Calculate correlations with shares (excluding non-numeric columns)
numeric_columns = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_columns].corr()['shares'].sort_values(ascending=False)
print("Top 10 features most correlated with shares:")
print(correlations.head(11))  # 11 to include shares itself
print("\nBottom 10 features least correlated with shares:")
print(correlations.tail(10))

# Save the processed dataframe
print("\n" + "="*50)
print("SAVING PROCESSED DATA")
print("="*50)
df.to_csv('processed_news_data.csv', index=False)
print("Processed data saved to 'processed_news_data.csv'")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"• Dataset contains {len(df):,} news articles from Mashable")
print(f"• {len(df.columns)} features including the target variable 'shares'")
print(f"• Articles published between {df['timedelta'].min():.0f} and {df['timedelta'].max():.0f} days before dataset acquisition")
print(f"• Target variable 'shares' ranges from {df['shares'].min():.0f} to {df['shares'].max():.0f}")
print(f"• Using threshold of 1400 shares: {popular_count:,} popular vs {unpopular_count:,} unpopular articles")
print("• No missing values in the dataset")
print("• Ready for machine learning analysis!")
