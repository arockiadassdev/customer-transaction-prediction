"""
Streamlit Web Application for Customer Transaction Prediction.

This app loads the trained model and allows users to make predictions
for customer transactions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import TransactionPredictor
from src.utils import get_project_root
from src.data_preprocessing import load_raw_data, clean_data, get_feature_columns

import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Customer Transaction Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .positive-prediction {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .negative-prediction {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """
    Load model, scaler, and features (cached for performance).
    
    Returns:
    --------
    TransactionPredictor
        Loaded predictor object.
    """
    try:
        predictor = TransactionPredictor()
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_data
def load_data_info():
    """
    Load dataset statistics for display.
    
    Returns:
    --------
    Dict with dataset information
    """
    try:
        root = get_project_root()
        df = load_raw_data('train.csv')
        df_clean = clean_data(df)
        feature_cols = get_feature_columns(df_clean)
        
        info = {
            'total_rows': len(df_clean),
            'total_features': len(feature_cols),
            'target_distribution': df_clean['target'].value_counts().to_dict(),
            'target_percentage': (df_clean['target'].value_counts(normalize=True) * 100).round(2).to_dict(),
            'feature_names': feature_cols
        }
        
        return info
    except Exception as e:
        logger.error(f"Error loading data info: {str(e)}")
        return None


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Customer Transaction Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether a customer will make a transaction based on anonymized features</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.info("""
        **Project:** PRCP-1003
        
        **Problem:** Binary classification to predict if a customer will make a transaction.
        
        **Model:** Gradient Boosting (HistGradientBoostingClassifier)
        
        **Primary Metric:** ROC-AUC
        
        **Class Distribution:**
        - No Transaction (0): ~90%
        - Transaction (1): ~10%
        """)
        
        st.header("⚙️ Settings")
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Adjust the threshold for classifying a customer as likely to transact"
        )
    
    # Load model and data
    predictor = load_model_artifacts()
    data_info = load_data_info()
    
    if predictor is None:
        st.error("Failed to load model. Please ensure the model is trained and saved.")
        st.stop()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📈 Batch Prediction", "📊 Model Info"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Single Customer Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Customer Features")
            
            # Get feature names from model
            if hasattr(predictor, 'selected_features') and predictor.selected_features:
                feature_names = predictor.selected_features
            else:
                feature_names = data_info['feature_names'] if data_info else [f'var_{i}' for i in range(200)]
            
            # Create input fields for features
            input_features = {}
            
            # Show first 20 features in expandable sections
            with st.expander(f"Feature Input ({len(feature_names)} features)", expanded=True):
                # We'll show a sample of features and generate random values
                # In production, these would come from actual customer data
                st.write("**Note:** Enter feature values for the customer. For anonymized features, use values from the dataset.")
                
                # Generate sample input based on dataset statistics
                if data_info:
                    # Use median values from dataset as defaults
                    df = load_raw_data('train.csv')
                    df_clean = clean_data(df)
                    medians = df_clean[feature_names].median()
                else:
                    medians = {f: 0.0 for f in feature_names}
                
                # Create input fields (show first 10, rest in expander)
                cols = st.columns(2)
                for idx, feature in enumerate(feature_names[:10]):
                    with cols[idx % 2]:
                        input_features[feature] = st.number_input(
                            feature,
                            value=float(medians.get(feature, 0.0)),
                            format="%.4f",
                            key=f"single_{feature}"
                        )
                
                if len(feature_names) > 10:
                    st.write(f"*... and {len(feature_names) - 10} more features*")
                    with st.expander("Show remaining features"):
                        for feature in feature_names[10:]:
                            input_features[feature] = st.number_input(
                                feature,
                                value=float(medians.get(feature, 0.0)),
                                format="%.4f",
                                key=f"single_{feature}"
                            )
        
        with col2:
            st.subheader("Prediction")
            
            # Make prediction button
            if st.button("🔮 Predict", type="primary", use_container_width=True):
                try:
                    # Prepare input
                    X = pd.DataFrame([input_features])
                    
                    # Get prediction
                    prediction = predictor.predict(X)[0]
                    probability = predictor.predict_proba(X)[0]
                    confidence = max(probability, 1 - probability)
                    
                    # Display prediction
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-box positive-prediction">
                            <h3>✅ Will Transact</h3>
                            <p><strong>Probability:</strong> {probability:.2%}</p>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box negative-prediction">
                            <h3>❌ Will Not Transact</h3>
                            <p><strong>Probability:</strong> {probability:.2%}</p>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Probability (Transaction)", f"{probability:.2%}")
                    with col_b:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Interpretation
                    st.subheader("Interpretation")
                    if probability >= 0.7:
                        st.success("High probability of transaction. Recommend priority outreach.")
                    elif probability >= threshold:
                        st.info("Moderate probability. Consider including in campaign.")
                    else:
                        st.warning("Low probability. May not be cost-effective to target.")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Prediction")
        
        st.subheader("Upload Customer Data")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with customer features",
            type=['csv'],
            help="Upload a CSV file containing customer features (same format as training data)"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                test_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(test_df)} customers")
                
                # Show preview
                with st.expander("Preview uploaded data"):
                    st.dataframe(test_df.head())
                
                # Make predictions
                if st.button("🔮 Predict for All Customers", type="primary"):
                    with st.spinner("Making predictions..."):
                        # Remove ID column if present
                        if 'ID_code' in test_df.columns:
                            X_test = test_df.drop(columns=['ID_code'])
                        else:
                            X_test = test_df
                        
                        # Get predictions
                        results = predictor.batch_predict(X_test, return_proba=True)
                        
                        # Combine with original data
                        if 'ID_code' in test_df.columns:
                            results.insert(0, 'ID_code', test_df['ID_code'])
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", len(results))
                        with col2:
                            will_transact = results['will_transact'].sum()
                            st.metric("Will Transact", will_transact)
                        with col3:
                            st.metric("Won't Transact", len(results) - will_transact)
                        
                        # Probability distribution
                        st.subheader("Probability Distribution")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.hist(results['probability'], bins=50, color='steelblue', edgecolor='white')
                        ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
                        ax.set_xlabel('Probability of Transaction')
                        ax.set_ylabel('Count')
                        ax.set_title('Distribution of Transaction Probabilities')
                        ax.legend()
                        st.pyplot(fig)
                        
                        # Results table
                        st.subheader("Detailed Results")
                        st.dataframe(results)
                        
                        # Download button
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.info("👆 Upload a CSV file to make batch predictions")
            
            # Show example
            with st.expander("Example CSV format"):
                example = pd.DataFrame({
                    'ID_code': ['example_1'],
                    'var_0': [1.2345],
                    'var_1': [-0.5678],
                    'var_199': [0.1234]
                })
                st.dataframe(example.head())
    
    # Tab 3: Model Info
    with tab3:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            if data_info:
                st.write(f"**Total Samples:** {data_info['total_rows']:,}")
                st.write(f"**Total Features:** {data_info['total_features']}")
                st.write(f"**Target Distribution:**")
                for label, pct in data_info['target_percentage'].items():
                    label_name = "No Transaction" if label == 0 else "Transaction"
                    st.write(f"  - {label_name}: {pct}%")
        
        with col2:
            st.subheader("Model Performance")
            st.write("**Algorithm:** Gradient Boosting (HistGradientBoostingClassifier)")
            st.write("**Primary Metric:** ROC-AUC")
            st.write("**Expected Performance:**")
            st.write("  - ROC-AUC: ~0.87-0.90")
            st.write("  - Average Precision: ~0.50-0.55")
            st.write("  - Accuracy: ~90% (but misleading due to class imbalance)")
            
            st.warning("""
            **Note:** Accuracy is not the best metric for this imbalanced dataset.
            A model predicting all 'No Transaction' would achieve 90% accuracy
            but would miss all actual transactions. Use ROC-AUC or Average
            Precision instead.
            """)
        
        st.subheader("Business Recommendations")
        st.write("""
        1. **Use probability scores** rather than binary predictions for decision making
        2. **Set threshold based on campaign economics:**
           - Lower threshold for broad outreach campaigns
           - Higher threshold for personalized high-cost interventions
        3. **Segment customers** by probability score:
           - Top 10%: Premium personalized outreach
           - Bottom 10%: Targeted retention offers
        4. **Monitor quarterly** for model drift and retrain with fresh data
        """)
        
        st.subheader("Technical Details")
        with st.expander("View Feature Engineering Details"):
            st.write("""
            **Feature Selection Method:** Consensus approach
            1. Mutual Information (top 80 features)
            2. Pearson Correlation (top 80 features)
            3. Random Forest Importance (top 80 features)
            
            **Final Selection:** Features appearing in at least 2 of 3 methods
            
            **Preprocessing:**
            - Missing values: None found
            - Duplicates: None found
            - Scaling: StandardScaler applied
            - Class imbalance: Handled via class_weight='balanced'
            """)
        
        with st.expander("View Model Parameters"):
            st.code("""
HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=6,
    learning_rate=0.08,
    class_weight='balanced',
    random_state=42
)
            """, language='python')


if __name__ == "__main__":
    main()