"""
Feature engineering module for Customer Transaction Prediction.
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

from src.utils import get_project_root

logger = logging.getLogger(__name__)


def select_features_mi(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 80,
    random_state: int = 42
) -> List[str]:
    """
    Select top features using Mutual Information.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_top : int
        Number of top features to select.
    random_state : int
        Random seed.
    
    Returns:
    --------
    List[str]
        List of selected feature names.
    """
    logger.info(f"Computing Mutual Information for {X.shape[1]} features...")
    t0 = time.time()
    
    mi = mutual_info_classif(X, y, random_state=random_state, n_neighbors=3)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    selected = mi_series.head(n_top).index.tolist()
    
    logger.info(f"MI selection completed in {time.time()-t0:.1f}s")
    logger.info(f"Selected top {len(selected)} features by MI")
    
    return selected


def select_features_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 80
) -> List[str]:
    """
    Select top features using Pearson correlation with target.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_top : int
        Number of top features to select.
    
    Returns:
    --------
    List[str]
        List of selected feature names.
    """
    logger.info("Computing Pearson correlation with target...")
    
    corr_series = X.corrwith(y).abs().sort_values(ascending=False)
    selected = corr_series.head(n_top).index.tolist()
    
    logger.info(f"Selected top {len(selected)} features by correlation")
    
    return selected


def select_features_rf_importance(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 80,
    sample_size: int = 50000,
    random_state: int = 42
) -> List[str]:
    """
    Select top features using Random Forest feature importance.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_top : int
        Number of top features to select.
    sample_size : int
        Size of stratified sample for RF training.
    random_state : int
        Random seed.
    
    Returns:
    --------
    List[str]
        List of selected feature names.
    """
    logger.info(f"Computing RF feature importance on {sample_size:,} sample...")
    t0 = time.time()
    
    # Stratified sample for speed
    n_pos = int(sample_size * y.mean())
    n_neg = sample_size - n_pos
    
    idx_sample = pd.concat([
        y[y == 0].sample(n_neg, random_state=random_state),
        y[y == 1].sample(n_pos, random_state=random_state)
    ]).index
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X.loc[idx_sample], y.loc[idx_sample])
    
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected = rf_imp.head(n_top).index.tolist()
    
    logger.info(f"RF importance selection completed in {time.time()-t0:.1f}s")
    logger.info(f"Selected top {len(selected)} features by RF importance")
    
    return selected


def consensus_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 80,
    min_votes: int = 2,
    random_state: int = 42
) -> List[str]:
    """
    Perform consensus feature selection using multiple methods.
    Features in top-n of at least min_votes methods are selected.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_top : int
        Number of top features per method.
    min_votes : int
        Minimum number of methods that must select a feature.
    random_state : int
        Random seed.
    
    Returns:
    --------
    List[str]
        List of selected feature names.
    """
    logger.info("="*60)
    logger.info("FEATURE SELECTION - CONSENSUS APPROACH")
    logger.info("="*60)
    
    # Get selections from each method
    mi_features = set(select_features_mi(X, y, n_top, random_state))
    corr_features = set(select_features_correlation(X, y, n_top))
    rf_features = set(select_features_rf_importance(X, y, n_top, random_state=random_state))
    
    # Count votes
    all_selections = list(mi_features) + list(corr_features) + list(rf_features)
    from collections import Counter
    vote_counts = Counter(all_selections)
    
    # Select features with at least min_votes
    selected = sorted([f for f, votes in vote_counts.items() if votes >= min_votes])
    
    logger.info(f"Consensus selection: {len(selected)} features (top-{n_top} in ≥{min_votes}/3 methods)")
    logger.info(f"MI: {len(mi_features)}, Corr: {len(corr_features)}, RF: {len(rf_features)}")
    
    return selected


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_fraction: float = 1.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling to handle class imbalance.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    sample_fraction : float
        Fraction of training data to use (for memory management).
    random_state : int
        Random seed.
    
    Returns:
    --------
    Tuple of X_resampled, y_resampled
    """
    logger.info("Applying SMOTE oversampling...")
    
    # Sample if needed for memory
    if sample_fraction < 1.0:
        n_samples = int(len(X_train) * sample_fraction)
        n_pos = int(n_samples * y_train.mean())
        n_neg = n_samples - n_pos
        
        idx_sample = pd.concat([
            y_train[y_train == 0].sample(n_neg, random_state=random_state),
            y_train[y_train == 1].sample(n_pos, random_state=random_state)
        ]).index
        
        X_sample = X_train.loc[idx_sample]
        y_sample = y_train.loc[idx_sample]
    else:
        X_sample = X_train
        y_sample = y_train
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_sample, y_sample)
    
    logger.info(f"SMOTE applied. Before: {dict(y_sample.value_counts())}, After: {dict(pd.Series(y_resampled).value_counts())}")
    
    return X_resampled, y_resampled


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale selected features using StandardScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    features : List[str]
        Features to scale.
    
    Returns:
    --------
    Tuple of X_train_scaled, X_test_scaled, scaler
    """
    logger.info(f"Scaling {len(features)} features with StandardScaler...")
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[features]),
        columns=features,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[features]),
        columns=features,
        index=X_test.index
    )
    
    logger.info("Feature scaling completed")
    
    return X_train_scaled, X_test_scaled, scaler


def feature_engineering_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    n_top_features: int = 80,
    apply_scaling: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training labels.
    n_top_features : int
        Number of features to select.
    apply_scaling : bool
        Whether to apply StandardScaler.
    random_state : int
        Random seed.
    
    Returns:
    --------
    Dict containing feature engineering outputs
    """
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*60)
    
    # Feature selection
    selected_features = consensus_feature_selection(
        X_train, y_train,
        n_top=n_top_features,
        random_state=random_state
    )
    
    result = {
        'selected_features': selected_features,
        'n_selected': len(selected_features)
    }
    
    # Scale features if needed
    if apply_scaling:
        X_train_sc, X_test_sc, scaler = scale_features(
            X_train, X_test, selected_features
        )
        result['X_train_selected'] = X_train_sc
        result['X_test_selected'] = X_test_sc
        result['scaler'] = scaler
    else:
        result['X_train_selected'] = X_train[selected_features]
        result['X_test_selected'] = X_test[selected_features]
    
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING COMPLETED")
    logger.info("="*60)
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Feature engineering module loaded successfully")