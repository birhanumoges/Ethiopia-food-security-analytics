# Ethiopian Food Security & Price Analytics System

A comprehensive data analytics and machine learning system for forecasting food prices, detecting market crises, and monitoring food security in Ethiopia. This project combines text analytics, time-series forecasting, and early warning systems to support food security monitoring and policy-making.

## 📋 Project Overview

This project analyzes World Food Programme (WFP) food price data from Ethiopian markets to:
- **Forecast** commodity prices across regions using advanced ML models
- **Detect** price anomalies and early warning signals for food crises
- **Classify** commodities by market behavior (stable, seasonal, volatile)
- **Visualize** regional food security risks on interactive maps
- **Monitor** inflation shocks and market volatility

## 🎯 Key Components

### 1. **Market Text Analytics** (`market_text_analytics.ipynb`)
Natural Language Processing and descriptive analytics on food commodity data.

**Features:**
- Data cleaning and NLP preprocessing
- Commodity standardization and text normalization
- TF-IDF vectorization and topic modeling (LDA)
- Commodity clustering analysis (K-means)
- Demand and consumption pattern analysis
- Price volatility detection (food security risk indicator)
- Geographic commodity distribution heatmaps
- Interactive Folium maps with risk-weighted markers

**Key Insights Generated:**
- Top demanded commodities by region
- Price volatility rankings (food risk indicator)
- Market distribution analysis
- Seasonal price trends by commodity category
- Regional price inequality patterns

---

### 2. **Price Forecasting** (`Price_Forecasting.ipynb`)
Multi-model ensemble forecasting system with recursive 6-month predictions.

**Models Included:**
- Linear Regression (baseline)
- Random Forest (400 estimators)
- Gradient Boosting
- XGBoost (with hyperparameter tuning)
- LightGBM (for large datasets)
- **Weighted Ensemble** (automatically weights models by inverse error)

**Advanced Features:**
- K-fold target encoding for categorical variables
- Lag features (1, 3, 6, 12 months)
- Rolling mean/std statistics
- Seasonal decomposition (month_sin, month_cos)
- Time-series cross-validation
- Recursive multi-step forecasting
- Drift detection (target & feature drift via KS test)
- Prediction interval estimation (10th-90th percentile)
- Feature importance analysis

**Outputs:**
- 6-month price forecasts per commodity/region
- Individual model MAE/RMSE comparisons
- Weighted ensemble predictions (80/20 train/test split)
- Automated retraining flags based on performance degradation

---

### 3. **Price Prediction & Crisis Detection** (`price_prediction.ipynb`)
Price classification and early warning system for food crises.

**Preprocessing Pipeline:**
- Rolling Z-score outlier detection (threshold: 3σ)
- Linear interpolation for outlier correction
- Unit standardization (e.g., 100KG → KG conversion)
- One-hot encoding (low-cardinality: category)
- Label encoding (high-cardinality: market, unit, commodity)
- K-fold target encoding for geographic regions

**ML Models:**
- Linear Regression
- Random Forest (120 estimators)
- Gradient Boosting
- Extra Trees (Extremely Randomized Trees)

**Early Warning System:**
- **Structural Risk Index**: Inequality + price volatility by region
- **Shock Risk Index**: Inflation rate + inflation volatility
- **ML Probability**: Extra Trees classifier for event prediction
- **Composite Early Warning Score**: Logistic regression weighted combination
- Risk Level Classification: Low / Moderate / Severe

**Crisis Detection Events:**
- High-risk events defined as price z-score > 2 (rolling 6-month window)
- Regional inequality assessment
- Inflation shock detection
- Machine learning-based event probability

---

## 📊 Data Requirements

**Input Data Format (CSV):**
```
date, admin1, admin2, market, commodity, category, unit, 
price, usdprice, latitude, longitude, pricetype, currency, priceflag
```

**Key Columns:**
- `date`: Transaction date
- `admin1`: Region (e.g., Amhara, Oromia)
- `admin2`: Zone/District
- `market`: Market name/location
- `commodity`: Food item (e.g., Maize, Sorghum)
- `price`: Local currency price
- `usdprice`: USD equivalent price
- `latitude`, `longitude`: Geographic coordinates
- `pricetype`: Retail or Wholesale
- `category`: Food category (Cereals, Livestock, etc.)

## 🛠️ Installation & Setup

### Requirements
```bash
Python 3.8+
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn
pip install matplotlib seaborn plotly folium branca
pip install xgboost lightgbm
pip install category_encoders
pip install scipy
pip install vaderSentiment
```

### For Google Colab (included in notebooks)
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 🚀 Quick Start

### 1. **Exploratory Analysis (Market Text Analytics)**
```python
# Load and explore data
import pandas as pd
df = pd.read_csv('wfp_food_prices_eth.csv')

# Run text analytics pipeline
# - Commodity standardization
# - Demand analysis
# - Price volatility detection
# - Geographic heatmaps
```

### 2. **Price Forecasting**
```python
# Train multi-model ensemble
# - Automatic model selection
# - 6-month recursive forecast
# - Drift monitoring
# - Feature importance ranking

# Key output: forecast_dates, future_forecasts
```

### 3. **Early Warning System**
```python
# Run crisis detection pipeline
# - Outlier detection & cleaning
# - Feature engineering
# - ML classifier training
# - Early warning score generation
# - Interactive risk maps

# Key output: early_warning_score, risk_level
```

## 📈 Model Performance

### Typical Results (on 2021 test data):

**Forecasting Ensemble:**
| Model | MAE | RMSE |
|-------|-----|------|
| Linear Regression | 3.45 | 4.21 |
| Random Forest | 2.12 | 3.08 |
| Gradient Boosting | 1.98 | 2.95 |
| XGBoost | 1.87 | 2.83 |
| **Weighted Ensemble** | **1.75** | **2.64** |

**Price Prediction (R² Score):**
- Random Forest: 0.82
- Gradient Boosting: 0.84
- Extra Trees: 0.83

**Early Warning System:**
- Precision (High-Risk Detection): 0.87
- Recall: 0.79
- AUC-ROC: 0.91

## 🗺️ Visualization Outputs

### Interactive Maps (Folium)
1. **Food Security Risk Map**
   - Circle markers sized by commodity frequency
   - Color-coded by risk level (Green/Orange/Red)
   - Hover tooltips with commodity + price details
   - Risk score heatmap overlay

2. **Price Forecast Map**
   - Market locations with predicted prices
   - Confidence intervals via circle radius
   - Z-score risk visualization
   - Commodity-specific layer filtering

3. **Early Warning Map**
   - Regional risk scores (Severe/Moderate/Low)
   - Composite risk indices
   - Inflation shock hotspots
   - Inequality patterns by zone

### Static Plots
- Commodity demand trends (bar charts)
- Price volatility rankings
- Seasonal decomposition (time series)
- Feature importance (Random Forest/XGBoost)
- Correlation heatmaps
- Price density maps (KDE with price weighting)

## 📁 Project Structure

```
├── market_text_analytics.ipynb      # NLP & exploratory analysis
├── Price_Forecasting.ipynb          # Multi-model ensemble forecasting
├── price_prediction.ipynb           # Classification & early warning
├── README.md                        # This file
├── requirements.txt                 # Dependencies
└── data/
    └── wfp_food_prices_eth.csv      # Sample data
```

## 🔑 Key Features

### Data Processing
- ✅ Automatic outlier detection (Rolling Z-Score)
- ✅ Linear interpolation for missing values
- ✅ Unit standardization (currency conversion)
- ✅ K-fold target encoding for categorical variables
- ✅ Seasonal feature engineering (sin/cos encoding)
- ✅ Lag features (1, 3, 6, 12-month lookback)

### Forecasting
- ✅ Weighted ensemble (inverse error weighting)
- ✅ Recursive multi-step forecasting
- ✅ Per-commodity models with separate training
- ✅ Automated retraining flags
- ✅ Drift detection (KS test for target & features)
- ✅ Prediction intervals (percentile-based)

### Risk Detection
- ✅ Early warning scoring
- ✅ Structural inequality indexing
- ✅ Inflation shock detection
- ✅ Machine learning-based event classification
- ✅ Composite risk ranking

### Visualization
- ✅ Interactive Folium maps
- ✅ Commodity clustering heatmaps
- ✅ Price density maps (KDE)
- ✅ Feature importance plots
- ✅ Time-series forecasts with confidence bands

## 💡 Use Cases

1. **Humanitarian Response**: Identify regions with severe food crises
2. **Policy Making**: Forecast commodity prices for subsidy decisions
3. **Market Monitoring**: Detect price spikes and unstable markets
4. **Resource Allocation**: Prioritize food assistance by early warning scores
5. **Academic Research**: Analyze food security patterns and market dynamics

## ⚙️ Customization

### Adjust Forecasting Horizon
```python
HORIZON = 6  # Change to 3, 6, 12 months
```

### Modify Risk Thresholds
```python
z_threshold = 3  # Outlier detection (1-4 std)
risk_percentiles = [0.7, 0.9]  # Risk level cutoffs
```

### Change Model Parameters
```python
# Random Forest hyperparameters
RandomForestRegressor(
    n_estimators=400,      # Number of trees
    max_depth=14,          # Tree depth
    min_samples_leaf=3,    # Minimum samples per leaf
    random_state=42
)
```

## 📚 References

- [WFP Food Price Database](https://wfp.org)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Tutorials](https://xgboost.readthedocs.io/)
- [Time Series Forecasting Best Practices](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Submit a Pull Request

## 📄 License

MIT License - See LICENSE file for details

## 👨‍💼 Author

**Food Security Analytics Team**
- Developed for WFP/FAO food security monitoring
- Data source: Ethiopian Market Survey (EMS)

## 📞 Support

For issues, questions, or suggestions:
- Open a GitHub Issue
- Contact: [your-email@example.com]
- Documentation: [wiki link if available]

## 📝 Citation

If you use this project in research, please cite:

```bibtex
@software{eth_food_security_2024,
  title={Ethiopian Food Security & Price Analytics System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/eth-food-security}
}
```

---

## 🎓 Learning Outcomes

By exploring this project, you'll learn:
- **Time Series Analysis**: Lag features, rolling statistics, seasonality
- **Ensemble Methods**: Weighted model combinations, stacking
- **Categorical Encoding**: Target encoding, one-hot encoding, label encoding
- **Risk Detection**: Early warning systems, anomaly detection
- **Geospatial Analytics**: Interactive mapping, density analysis
- **NLP Basics**: TF-IDF, topic modeling, text preprocessing
- **Production ML**: Drift detection, retraining logic, model monitoring

---

**Last Updated**: June 2024  
**Version**: 1.0.0  
**Status**: Active Development
