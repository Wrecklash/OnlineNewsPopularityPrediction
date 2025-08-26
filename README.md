# Online News Popularity Prediction

## 📊 Project Overview

This repository contains a comprehensive analysis of the **Online News Popularity Dataset** from Mashable.com. The project focuses on understanding what factors contribute to the popularity of online news articles, measured by the number of social media shares.

## 🎯 Dataset Information

- **Source**: Mashable.com articles
- **Total Articles**: 39,644
- **Features**: 61 (58 predictive + 2 non-predictive + 1 target)
- **Target Variable**: `shares` (number of social media shares)
- **Time Period**: Articles published 8-731 days before dataset acquisition
- **Dataset Size**: ~23 MB

## 📁 Repository Structure

```
OnlineNewsPopularity/
├── OnlineNewsPopularity.csv          # Original dataset
├── OnlineNewsPopularity.names        # Dataset description file
├── processed_news_data.csv           # Clean, processed dataset
├── news_popularity_analysis.png      # Visualization dashboard
├── comprehensive_analysis.txt        # Detailed analysis documentation
├── url_content_analysis.json         # Sample URL content analysis
├── read_csv_data.py                  # Main data reading and analysis script
├── visualize_data.py                 # Comprehensive visualization script
├── data_summary.py                   # Quick summary and insights script
├── fetch_url_content.py              # URL content extraction script
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn requests beautifulsoup4
```

### Running the Analysis

1. **Basic Data Analysis**:
   ```bash
   python read_csv_data.py
   ```

2. **Generate Visualizations**:
   ```bash
   python visualize_data.py
   ```

3. **Quick Summary**:
   ```bash
   python data_summary.py
   ```

4. **URL Content Analysis** (optional):
   ```bash
   python fetch_url_content.py
   ```

## 📈 Key Findings

### 🏆 Top Performing Content Channels
- **Lifestyle**: 3,682 average shares
- **Social Media**: 3,629 average shares
- **Tech**: 3,072 average shares

### 📅 Best Publishing Days
- **Saturday**: 4,078 average shares
- **Sunday**: 3,747 average shares
- **Monday**: 3,647 average shares

### 🔍 Most Correlated Features
1. `kw_avg_avg` (0.1104) - Average keyword average shares
2. `LDA_03` (0.0838) - Topic modeling closeness
3. `kw_max_avg` (0.0643) - Average keyword max shares
4. `self_reference_avg_sharess` (0.0578) - Self-reference average shares
5. `num_hrefs` (0.0454) - Number of links

## 📊 Feature Categories

### Content Features
- `n_tokens_title` - Number of words in title
- `n_tokens_content` - Number of words in content
- `num_hrefs` - Number of links
- `num_imgs` - Number of images
- `num_videos` - Number of videos

### Sentiment Features
- `global_subjectivity` - Text subjectivity
- `global_sentiment_polarity` - Text sentiment polarity
- `global_rate_positive_words` - Rate of positive words
- `global_rate_negative_words` - Rate of negative words

### Temporal Features
- `weekday_is_*` - Publication day indicators
- `is_weekend` - Weekend publication flag
- `timedelta` - Days before dataset acquisition

### Channel Features
- `data_channel_is_lifestyle` - Lifestyle articles
- `data_channel_is_entertainment` - Entertainment articles
- `data_channel_is_bus` - Business articles
- `data_channel_is_socmed` - Social media articles
- `data_channel_is_tech` - Technology articles
- `data_channel_is_world` - World news articles

### Keyword Features
- `kw_*_*` - Various keyword performance metrics
- `self_reference_*_shares` - Self-reference share statistics

## 🔧 Data Processing

### Cleaning Operations
- ✅ Column name whitespace removal
- ✅ Missing value validation (none found)
- ✅ Data type verification
- ✅ Feature range validation

### Feature Engineering
- 📝 **Note**: Binary classification target not created automatically
- 💡 **Suggestion**: Create manually if needed:
  ```python
  df['is_popular'] = (df['shares'] >= 1400).astype(int)
  ```

## 📊 Visualizations

The project includes a comprehensive visualization dashboard (`news_popularity_analysis.png`) with:

1. **Shares Distribution** - Raw and log-transformed distributions
2. **Data Channel Analysis** - Pie chart and box plots
3. **Temporal Analysis** - Publication day patterns
4. **Statistical Analysis** - Quartiles and correlations

## 🌐 URL Content Analysis

A sample of 10 articles was analyzed to extract actual content:
- **Article titles** and **content previews**
- **Word counts** and **domain information**
- **Success/failure status** of content extraction
- **Correlation** between content and popularity

## 📋 Machine Learning Readiness

### Data Quality
- ✅ **No missing values**
- ✅ **Clean feature encoding**
- ✅ **Proper data types**
- ⚠️ **Outliers present** in shares (handled with log transformation)

### Recommendations
1. **Apply log transformation** to shares for modeling
2. **Scale numerical features** using StandardScaler
3. **Use cross-validation** for robust evaluation
4. **Try ensemble methods** (Random Forest, XGBoost)
5. **Consider feature selection** based on correlations

## 📚 Files Description

### Scripts
- **`read_csv_data.py`**: Main data reading, cleaning, and statistical analysis
- **`visualize_data.py`**: Comprehensive visualization generation
- **`data_summary.py`**: Quick overview and key insights
- **`fetch_url_content.py`**: URL content extraction and documentation

### Output Files
- **`processed_news_data.csv`**: Clean, ready-to-use dataset
- **`news_popularity_analysis.png`**: Complete visualization dashboard
- **`comprehensive_analysis.txt`**: Detailed documentation of all operations
- **`url_content_analysis.json`**: Sample URL content analysis

## 🎯 Use Cases

This dataset is ideal for:
- **Predictive modeling** of article popularity
- **Content optimization** strategies
- **Publishing timing** analysis
- **Feature importance** studies
- **Sentiment analysis** research

## 📖 Citation

If you use this dataset, please cite:
```
K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision
Support System for Predicting the Popularity of Online News. Proceedings
of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence,
September, Coimbra, Portugal.
```

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new analysis scripts
- Improving visualizations
- Enhancing documentation
- Suggesting new features

## 📄 License

This project is for educational and research purposes. The original dataset belongs to Mashable.com.

## 📞 Contact

For questions or suggestions, please open an issue in this repository.

---

**Happy Analyzing! 🚀**
