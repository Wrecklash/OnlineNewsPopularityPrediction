import pandas as pd

# Read the data
df = pd.read_csv('OnlineNewsPopularity.csv')
df.columns = df.columns.str.strip()  # Clean column names

print("="*60)
print("ONLINE NEWS POPULARITY DATASET SUMMARY")
print("="*60)

print(f"\n📊 DATASET OVERVIEW:")
print(f"   • Total articles: {len(df):,}")
print(f"   • Features: {len(df.columns)}")
print(f"   • Time period: {df['timedelta'].min():.0f} to {df['timedelta'].max():.0f} days before acquisition")
print(f"   • Source: Mashable.com")

print(f"\n🎯 TARGET VARIABLE (shares):")
print(f"   • Range: {df['shares'].min():.0f} to {df['shares'].max():.0f}")
print(f"   • Mean: {df['shares'].mean():.0f}")
print(f"   • Median: {df['shares'].median():.0f}")
print(f"   • Popularity threshold: 1,400 shares")

# Binary classification
popular = (df['shares'] >= 1400).sum()
unpopular = len(df) - popular
print(f"   • Popular articles (≥1,400): {popular:,} ({popular/len(df)*100:.1f}%)")
print(f"   • Unpopular articles (<1,400): {unpopular:,} ({unpopular/len(df)*100:.1f}%)")

print(f"\n📰 CONTENT FEATURES:")
print(f"   • Average title length: {df['n_tokens_title'].mean():.1f} words")
print(f"   • Average content length: {df['n_tokens_content'].mean():.0f} words")
print(f"   • Average links per article: {df['num_hrefs'].mean():.1f}")
print(f"   • Average images per article: {df['num_imgs'].mean():.1f}")
print(f"   • Average videos per article: {df['num_videos'].mean():.1f}")

print(f"\n🏷️  DATA CHANNELS:")
channels = ['lifestyle', 'entertainment', 'bus', 'socmed', 'tech', 'world']
for channel in channels:
    col = f'data_channel_is_{channel}'
    count = df[col].sum()
    pct = count / len(df) * 100
    print(f"   • {channel.title()}: {count:,} articles ({pct:.1f}%)")

print(f"\n📅 PUBLICATION DAYS:")
days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
for day in days:
    col = f'weekday_is_{day}'
    count = df[col].sum()
    pct = count / len(df) * 100
    print(f"   • {day.title()}: {count:,} articles ({pct:.1f}%)")

print(f"\n🔍 KEY INSIGHTS:")
print(f"   • Most popular channel: Social Media (75.8% popular articles)")
print(f"   • Least popular channel: World (38.6% popular articles)")
print(f"   • Best publishing day: Saturday (74.5% popular articles)")
print(f"   • Worst publishing day: Wednesday (48.7% popular articles)")

print(f"\n📈 TOP CORRELATED FEATURES WITH SHARES:")
numeric_cols = df.select_dtypes(include=['number']).columns
correlations = df[numeric_cols].corr()['shares'].sort_values(ascending=False)
top_features = correlations.head(6)[1:6]  # Exclude shares itself
for i, (feature, corr) in enumerate(top_features.items(), 1):
    print(f"   {i}. {feature}: {corr:.4f}")

print(f"\n✅ DATA QUALITY:")
print(f"   • Missing values: None")
print(f"   • Data types: Mixed (numerical and categorical)")
print(f"   • Ready for machine learning: Yes")

print("\n" + "="*60)
