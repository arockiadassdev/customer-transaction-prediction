# 🏦 Customer Transaction Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PRCP-1003:** A complete machine learning project to predict customer transactions using anonymized banking data. Built with production-ready code, comprehensive EDA, and an interactive web application.

---

## 📊 Project Overview

### Business Problem
A bank wants to proactively identify which customers are likely to make a specific transaction in the future — irrespective of the transaction amount. Being able to predict this in advance enables the bank to:

- **Personalize** outreach, offers, and retention strategies
- **Improve** liquidity planning by forecasting transaction volumes
- **Reduce** customer churn by engaging potential transactors

### Technical Approach
- **Task:** Binary classification (Will transact: Yes/No)
- **Primary Metric:** ROC-AUC (standard for imbalanced classification)
- **Challenge:** Severe class imbalance (~10% positive class) + 200 anonymized features
- **Solution:** Ensemble methods (Gradient Boosting) with consensus feature selection

### Expected Outcomes
- ✅ Trained, production-ready model with documented performance
- ✅ Model comparison across 6+ algorithms
- ✅ Feature importance analysis and business recommendations
- ✅ Interactive Streamlit web application for predictions
- ✅ Complete reproducible pipeline

---

## 📁 Project Structure

```
customer-transaction-prediction/
├── data/
│   ├── raw/                    # Raw dataset files
│   └── processed/              # Processed dataset files
├── notebooks/
│   └── project_analysis.ipynb  # Complete analysis notebook
├── src/
│   ├── utils.py                # Utility functions
│   ├── data_preprocessing.py   # Data loading and cleaning
│   ├── feature_engineering.py  # Feature selection and scaling
│   ├── train_model.py          # Model training functions
│   ├── evaluate_model.py       # Model evaluation and metrics
│   └── predict.py              # Prediction class
├── models/
│   ├── gradient_boosting_tuned.pkl  # Best trained model
│   ├── scaler.pkl                    # Feature scaler
│   └── selected_features.pkl         # Selected features list
├── app/
│   └── app.py                  # Streamlit web application
├── reports/                    # Generated reports and visualizations
├── images/                     # Saved plots and charts
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

---

## 🎯 Dataset Information

| Property | Detail |
|----------|--------|
| **Rows** | 200,000 |
| **Columns** | 202 (`ID_code`, `target`, `var_0`–`var_199`) |
| **Features** | 200 anonymized continuous float features |
| **Target** | Binary: `1` = will transact (~10%), `0` = won't transact (~90%) |
| **Missing Values** | None |
| **Duplicates** | None |
| **Class Imbalance** | ~10:1 (majority:minority) |
| **Domain** | Banking/Finance |

**Key Challenge:** The dataset is fully anonymized with no disclosed feature names, requiring robust ML approaches without domain-driven feature engineering.

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/arockiadassdev/customer-transaction-prediction.git
cd customer-transaction-prediction
```

2. **Create virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
   - Place the dataset in `data/raw/train.csv`
   - The dataset file should be named `train.csv`

---

## 💻 Usage

### 1. Run the Complete Pipeline

```python
from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.train_model import train_all_models, hyperparameter_tuning, train_tuned_model
from src.evaluate_model import full_evaluation_pipeline
from src.utils import save_model, save_features
import joblib

# Step 1: Preprocess data
preprocessing = preprocess_pipeline(filename='train.csv')
X_train, X_test = preprocessing['X_train'], preprocessing['X_test']
y_train, y_test = preprocessing['y_train'], preprocessing['y_test']
X_train_sc, X_test_sc = preprocessing['X_train_scaled'], preprocessing['X_test_scaled']

# Step 2: Feature engineering
feature_eng = feature_engineering_pipeline(
    X_train, X_test, y_train,
    n_top_features=80,
    apply_scaling=True
)
X_train_sel = feature_eng['X_train_selected']
X_test_sel = feature_eng['X_test_selected']
selected_features = feature_eng['selected_features']

# Step 3: Train all models
training_results = train_all_models(
    X_train_sel, X_test_sel,
    y_train, y_test,
    scaled=True
)

# Step 4: Hyperparameter tuning (optional, takes longer)
best_model, best_params = hyperparameter_tuning(
    X_train_sel, y_train,
    model_type='gradient_boosting',
    n_iter=6
)

# Step 5: Train tuned model on full data
tuned_model, tuned_metrics = train_tuned_model(
    X_train_sel, y_train,
    X_test_sel, y_test,
    best_params,
    model_type='gradient_boosting'
)

# Step 6: Save artifacts
joblib.dump(tuned_model, 'models/gradient_boosting_tuned.pkl')
joblib.dump(preprocessing['scaler'], 'models/scaler.pkl')
joblib.dump(selected_features, 'models/selected_features.pkl')

# Step 7: Evaluate
results_df = full_evaluation_pipeline(
    training_results['fitted_models'],
    X_test_sel, y_test,
    selected_features
)
```

### 2. Make Predictions

```python
from src.predict import TransactionPredictor

# Load the predictor
predictor = TransactionPredictor()

# Single prediction
customer_features = {
    'var_0': 1.2345,
    'var_1': -0.5678,
    # ... all required features
}
result = predictor.predict_single(customer_features)
print(f"Will transact: {result['will_transact']}")
print(f"Probability: {result['probability']:.2%}")

# Batch prediction
import pandas as pd
test_df = pd.read_csv('data/raw/test.csv')
results = predictor.batch_predict(test_df, return_proba=True)
print(results)
```

### 3. Run the Streamlit Web App

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501` with three main features:
- **Single Prediction:** Input customer features manually
- **Batch Prediction:** Upload CSV files for bulk predictions
- **Model Info:** View dataset statistics and model performance

---

## 📈 Exploratory Data Analysis (EDA)

The analysis notebook (`notebooks/project_analysis.ipynb`) includes comprehensive EDA:

### Key Findings:
1. **Zero Missing Values** — No imputation needed
2. **No Duplicates** — Dataset is clean
3. **Class Imbalance: 10:1** — Addressed via `class_weight='balanced'` and SMOTE
4. **All Continuous Features** — 200 float features, no categorical encoding needed
5. **Weak Individual Signals** — Feature-target correlations max ~0.06-0.10
6. **Low Multicollinearity** — Features are largely independent
7. **Mild Skewness** — Most features within acceptable range

### Visualizations Included:
- Target variable distribution (class imbalance analysis)
- Histograms for first 16 features
- Feature distributions by target class
- Boxplots for outlier inspection
- Correlation heatmap (first 30 features)
- Top 20 features by correlation with target
- Skewness distribution across all features
- Feature selection comparison (MI, Correlation, RF Importance)

---

## 🔧 Feature Engineering

### Feature Selection Strategy
**Consensus Approach (Top 80 from at least 2/3 methods):**

1. **Mutual Information** — Captures linear and non-linear associations
2. **Pearson Correlation** — Fast linear baseline
3. **Random Forest Importance** — Model-driven, captures interactions

**Result:** ~70-100 features selected from 200 original features

### Preprocessing Steps:
- ✅ Missing value handling (none found)
- ✅ Outlier detection (IQR method — outliers retained as they carry predictive signal)
- ✅ Feature scaling (StandardScaler for linear/distance-based models)
- ✅ Train/test split with stratification (80/20)
- ✅ SMOTE applied to training set only (for KNN and Gaussian NB)

---

## 🤖 Model Building

### Models Trained (6 algorithms):

| Model | Type | Scaling Required | Notes |
|-------|------|-----------------|-------|
| **Logistic Regression** | Linear | Yes | With `class_weight='balanced'` |
| **Decision Tree** | Tree | No | Max depth 8, min samples leaf 50 |
| **Random Forest** | Ensemble | No | 200 trees, balanced weights |
| **Gradient Boosting** | Ensemble | No | HistGradientBoostingClassifier |
| **KNN** | Distance-based | Yes | With SMOTE balancing |
| **Gaussian NB** | Probabilistic | No | With SMOTE balancing |

### Hyperparameter Tuning
- **Method:** RandomizedSearchCV with 3-fold Stratified CV
- **Sample Size:** 60,000 stratified samples (for computational efficiency)
- **Models Tuned:** Random Forest and Gradient Boosting
- **Metric:** ROC-AUC

**Best Parameters Found:**
```python
# Gradient Boosting (Best Model)
HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=6,
    learning_rate=0.08,
    class_weight='balanced',
    random_state=42
)
```

---

## 📊 Model Evaluation

### Evaluation Metrics:
- **Primary:** ROC-AUC (ranking quality across all thresholds)
- **Secondary:** Average Precision (PR-AUC), F1-Score, Recall, Precision
- **Validation:** 5-Fold Stratified Cross-Validation

### Model Comparison Results:

| Model | ROC-AUC | Avg Precision | F1-Score | Recall |
|-------|---------|---------------|----------|--------|
| **Gradient Boosting (Tuned)** | ~0.87-0.90 | ~0.50-0.55 | ~0.45-0.50 | ~0.65-0.70 |
| Random Forest (Tuned) | ~0.85-0.88 | ~0.48-0.52 | ~0.42-0.47 | ~0.60-0.65 |
| Logistic Regression | ~0.82-0.85 | ~0.40-0.45 | ~0.35-0.40 | ~0.55-0.60 |
| ... | ... | ... | ... | ... |

**Note:** Exact values vary with random seed and feature selection. Gradient Boosting consistently outperforms other models.

### Visualizations:
- ROC Curves for all models
- Precision-Recall Curves
- Confusion Matrices
- Feature Importance Plot
- Model Comparison Charts
- Cross-Validation Boxplots

### Why Gradient Boosting Was Selected:
1. **Highest ROC-AUC** — Best ranking capability
2. **Highest Average Precision** — Best performance on imbalanced data
3. **Handles Weak Signals** — Effective at capturing diffuse feature interactions
4. **Robust to Outliers** — Tree-based methods are naturally robust
5. **Industry Standard** — Gradient boosting is state-of-the-art for tabular data

---

## 🎯 Business Insights

1. **Transaction prediction is a weak-signal, multi-feature problem.** No single feature dominates — transaction behavior is the aggregate of many small customer signals.

2. **Class imbalance mirrors real-world rarity.** Only ~10% of customers transact in a given window. Accuracy is misleading (90% by predicting all "No Transaction").

3. **Gradient Boosting family consistently outperforms.** Histogram-based GBMs handle large feature spaces efficiently while capturing complex non-linear interactions.

4. **Probability scores are more valuable than binary predictions.** Deploy as a probability scorer (0 to 1) rather than binary classifier. The bank can set different thresholds for different use cases.

5. **Top features cluster into consistent groups.** Despite anonymization, feature importance is stable across methods — suggesting coherent latent dimensions of customer behavior.

---

## 💡 Recommendations

### For the Bank (Marketing & Operations):
1. **Deploy as a real-time probability scorer** within CRM — score each customer monthly
2. **Set threshold based on campaign economics:** Break-even threshold = Cost/Revenue
3. **Segment by probability:**
   - Top 10%: Premium personalized outreach
   - Bottom 10%: Targeted retention offers
4. **Monitor quarterly** for model drift and retrain with fresh data

### For the Data Science Team:
5. **Pursue feature de-anonymization** (if legally permitted) for domain-driven engineering
6. **Explore stacking/blending** of Gradient Boosting + Logistic Regression
7. **A/B test** model-driven targeting against current approach

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Programming** | Python 3.8+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, Imbalanced-learn |
| **Model Persistence** | Joblib |
| **Web Application** | Streamlit |
| **Development** | Jupyter, Git |

---

## 📝 Model Performance

### Final Model: Gradient Boosting (HistGradientBoostingClassifier)

**Expected Performance Metrics:**
- **ROC-AUC:** 0.87 - 0.90
- **Average Precision:** 0.50 - 0.55
- **F1-Score:** 0.45 - 0.50
- **Recall:** 0.65 - 0.70
- **Accuracy:** ~90% (misleading due to class imbalance)

### Model Saved:
- ✅ `models/gradient_boosting_tuned.pkl` — Best trained model
- ✅ `models/scaler.pkl` — Feature scaler for preprocessing
- ✅ `models/selected_features.pkl` — List of selected features

### Sanity Check:
The reloaded model produces identical predictions to the original, confirming successful serialization.

---

## 🌐 Web Application

### Features:
- **Single Customer Prediction** — Input features manually and get instant predictions
- **Batch Prediction** — Upload CSV files with multiple customers
- **Probability Distribution** — Visualize prediction confidence
- **Model Information** — View dataset statistics and performance metrics
- **Business Recommendations** — Actionable insights for marketing

### Screenshots:
> *Screenshots will be added here after running the app*

```
1. Home page with prediction interface
2. Single prediction result
3. Batch prediction results
4. Model information dashboard
```

### Running the App:
```bash
streamlit run app/app.py
```

---

## 🔄 Future Improvements

1. **Model Deployment**
   - Deploy as REST API using FastAPI
   - Containerize with Docker
   - CI/CD pipeline for automated retraining

2. **Advanced Techniques**
   - Implement stacking/blending ensembles
   - Add XGBoost and LightGBM for comparison
   - Experiment with deep learning models

3. **Feature Engineering**
   - Pursue feature de-anonymization for domain-driven features
   - Create interaction terms for top features
   - Add temporal features if timestamp data becomes available

4. **Monitoring & Maintenance**
   - Implement model drift detection
   - Set up automated retraining pipeline
   - Create monitoring dashboard for production metrics

5. **Explainability**
   - Add SHAP values for model interpretability
   - Create LIME explanations for individual predictions
   - Build fairness metrics for bias detection

6. **Performance Optimization**
   - Optimize model for faster inference
   - Implement model quantization
   - Add caching for frequent predictions

---

## 📚 Documentation

### Notebooks:
- **`notebooks/project_analysis.ipynb`** — Complete analysis with markdown explanations and outputs

### Code Documentation:
- All modules include detailed docstrings
- Type hints for better IDE support
- Logging throughout for debugging

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Arockia Dass**
- GitHub: [@arockiadassdev](https://github.com/arockiadassdev)
- Email: arockiadassdev@gmail.com

---

## 🙏 Acknowledgments

- Dataset provided by Santander Bank (Kaggle Competition)
- Project inspired by PRCP-1003 Data Science Program
- Built as a portfolio-ready GitHub project

---

## 📞 Contact

For questions or feedback:
- **Email:** arockiadassdev@gmail.com
- **GitHub:** [@arockiadassdev](https://github.com/arockiadassdev)

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐ on GitHub!

---

**Built with ❤️ for the Data Science Community**