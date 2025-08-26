import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")

# Read the processed data
print("Loading the processed dataset...")
df = pd.read_csv('processed_news_data.csv')

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Online News Popularity Dataset Analysis', fontsize=16, fontweight='bold')

# 1. Distribution of shares (target variable)
axes[0, 0].hist(df['shares'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Shares')
axes[0, 0].set_xlabel('Number of Shares')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(x=1400, color='red', linestyle='--', label='Popularity Threshold')
axes[0, 0].legend()

# 2. Log distribution of shares (better visualization)
axes[0, 1].hist(np.log1p(df['shares']), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Distribution of Log(Shares + 1)')
axes[0, 1].set_xlabel('Log(Number of Shares + 1)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(x=np.log1p(1400), color='red', linestyle='--', label='Popularity Threshold')
axes[0, 1].legend()

# 3. Data channel distribution
channel_columns = [col for col in df.columns if col.startswith('data_channel_is_')]
channel_counts = [df[col].sum() for col in channel_columns]
channel_names = [col.replace('data_channel_is_', '').title() for col in channel_columns]

axes[0, 2].pie(channel_counts, labels=channel_names, autopct='%1.1f%%', startangle=90)
axes[0, 2].set_title('Distribution by Data Channel')

# 4. Weekday distribution
weekday_columns = [col for col in df.columns if col.startswith('weekday_is_') and col != 'weekday_is_weekend']
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_counts = [df[col].sum() for col in weekday_columns]

axes[1, 0].bar(weekday_names, weekday_counts, color='coral', alpha=0.7)
axes[1, 0].set_title('Articles Published by Day of Week')
axes[1, 0].set_xlabel('Day of Week')
axes[1, 0].set_ylabel('Number of Articles')
axes[1, 0].tick_params(axis='x', rotation=45)

# 5. Popular vs Unpopular distribution
popular_counts = df['is_popular'].value_counts()
axes[1, 1].pie(popular_counts.values, labels=['Unpopular (<1400)', 'Popular (≥1400)'], 
               autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue'])
axes[1, 1].set_title('Popular vs Unpopular Articles')

# 6. Box plot of shares by channel
channel_data = []
channel_labels = []
for col in channel_columns:
    channel_articles = df[df[col] == 1]
    if len(channel_articles) > 0:
        channel_data.append(channel_articles['shares'].values)
        channel_labels.append(col.replace('data_channel_is_', '').title())

axes[1, 2].boxplot(channel_data, labels=channel_labels)
axes[1, 2].set_title('Shares Distribution by Data Channel')
axes[1, 2].set_ylabel('Number of Shares')
axes[1, 2].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()
plt.savefig('news_popularity_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'news_popularity_analysis.png'")

# Create additional detailed analysis
print("\n" + "="*60)
print("DETAILED FEATURE ANALYSIS")
print("="*60)

# Content features analysis
print("\nCONTENT FEATURES:")
content_features = ['n_tokens_title', 'n_tokens_content', 'num_hrefs', 'num_imgs', 'num_videos']
for feature in content_features:
    print(f"{feature}:")
    print(f"  Mean: {df[feature].mean():.2f}")
    print(f"  Median: {df[feature].median():.2f}")
    print(f"  Correlation with shares: {df[feature].corr(df['shares']):.4f}")

# Sentiment analysis
print("\nSENTIMENT FEATURES:")
sentiment_features = ['global_subjectivity', 'global_sentiment_polarity', 
                     'global_rate_positive_words', 'global_rate_negative_words']
for feature in sentiment_features:
    print(f"{feature}:")
    print(f"  Mean: {df[feature].mean():.4f}")
    print(f"  Correlation with shares: {df[feature].corr(df['shares']):.4f}")

# Popularity by channel
print("\nPOPULARITY BY DATA CHANNEL:")
for col in channel_columns:
    channel_name = col.replace('data_channel_is_', '').title()
    channel_articles = df[df[col] == 1]
    avg_shares = channel_articles['shares'].mean()
    popular_pct = (channel_articles['is_popular'].sum() / len(channel_articles)) * 100
    print(f"{channel_name}:")
    print(f"  Average shares: {avg_shares:.0f}")
    print(f"  Popular articles: {popular_pct:.1f}%")

# Popularity by weekday
print("\nPOPULARITY BY WEEKDAY:")
for i, col in enumerate(weekday_columns):
    weekday_articles = df[df[col] == 1]
    avg_shares = weekday_articles['shares'].mean()
    popular_pct = (weekday_articles['is_popular'].sum() / len(weekday_articles)) * 100
    print(f"{weekday_names[i]}:")
    print(f"  Average shares: {avg_shares:.0f}")
    print(f"  Popular articles: {popular_pct:.1f}%")

print(f"\nAnalysis complete!")
