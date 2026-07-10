"""
Model training module for Customer Transaction Prediction.
"""

import pandas as pd
import numpy as np
import logging
import time
import os
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import joblib

from src.utils import get_project_root, save_model

logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict = None
) -> Tuple[Any, Dict]:
    """
    Train Logistic Regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features (scaled).
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test features (scaled).
    y_test : pd.Series
        Test labels.
    params : Dict
        Model parameters.
    
    Returns:
    --------
    Tuple of (trained model, metrics dict)
    """
    logger.info("Training Logistic Regression...")
    
    default_params = {
        'max_iter': 1000,
        'class_weight': 'balanced',
        'C': 0.1,
        'solver': 'saga',
        'n_jobs': -1,
        'random_state': 42
    }
    if params:
        default_params.update(params)
    
    model = LogisticRegression(**default_params)
    model.fit(X_train, y_train)
    
    from src.evaluate_model import evaluate_model
    metrics = evaluate_model(model, X_test, y_test)
    
    logger.info(f"Logistic Regression - ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    return model, metrics


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict = None
) -> Tuple[Any, Dict]:
    """
    Train Decision Tree model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    params : Dict
        Model parameters.
    
    Returns:
    --------
    Tuple of (trained model, metrics dict)
    """
    logger.info("Training Decision Tree...")
    
    default_params = {
        'max_depth': 8,
        'min_samples_leaf': 50,
        'class_weight': 'balanced',
        'random_state': 42
    }
    if params:
        default_params.update(params)
    
    model = DecisionTreeClassifier(**default_params)
    model.fit(X_train, y_train)
    
    from src.evaluate_model import evaluate_model
    metrics = evaluate_model(model, X_test, y_test)
    
    logger.info(f"Decision Tree - ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    return model, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict = None
) -> Tuple[Any, Dict]:
    """
    Train Random Forest model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    params : Dict
        Model parameters.
    
    Returns:
    --------
    Tuple of (trained model, metrics dict)
    """
    logger.info("Training Random Forest...")
    
    default_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_leaf': 20,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    if params:
        default_params.update(params)
    
    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    
    from src.evaluate_model import evaluate_model
    metrics = evaluate_model(model, X_test, y_test)
    
    logger.info(f"Random Forest - ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    return model, metrics


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict = None
) -> Tuple[Any, Dict]:
    """
    Train Gradient Boosting model (HistGradientBoostingClassifier).
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    params : Dict
        Model parameters.
    
    Returns:
    --------
    Tuple of (trained model, metrics dict)
    """
    logger.info("Training Gradient Boosting...")
    
    default_params = {
        'max_iter': 200,
        'max_depth': 6,
        'learning_rate': 0.08,
        'class_weight': 'balanced',
        'random_state': 42
    }
    if params:
        default_params.update(params)
    
    model = HistGradientBoostingClassifier(**default_params)
    model.fit(X_train, y_train)
    
    from src.evaluate_model import evaluate_model
    metrics = evaluate_model(model, X_test, y_test)
    
    logger.info(f"Gradient Boosting - ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    return model, metrics


def train_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict = None
) -> Tuple[Any, Dict]:
    """
    Train K-Nearest Neighbors model (uses SMOTE for balancing).
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    params : Dict
        Model parameters.
    
    Returns:
    --------
    Tuple of (trained model, metrics dict)
    """
    logger.info("Training KNN (with SMOTE)...")
    
    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    default_params = {
        'n_neighbors': 15,
        'n_jobs': -1
    }
    if params:
        default_params.update(params)
    
    model = KNeighborsClassifier(**default_params)
    model.fit(X_train_res, y_train_res)
    
    from src.evaluate_model import evaluate_model
    metrics = evaluate_model(model, X_test, y_test)
    
    logger.info(f"KNN - ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    return model, metrics


def train_gaussian_nb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: Dict = None
) -> Tuple[Any, Dict]:
    """
    Train Gaussian Naive Bayes model (uses SMOTE for balancing).
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    params : Dict
        Model parameters.
    
    Returns:
    --------
    Tuple of (trained model, metrics dict)
    """
    logger.info("Training Gaussian Naive Bayes (with SMOTE)...")
    
    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model = GaussianNB()
    if params:
        model.set_params(**params)
    model.fit(X_train_res, y_train_res)
    
    from src.evaluate_model import evaluate_model
    metrics = evaluate_model(model, X_test, y_test)
    
    logger.info(f"Gaussian NB - ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    return model, metrics


def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaled: bool = True
) -> Dict[str, Any]:
    """
    Train all models and return results.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.
    scaled : bool
        Whether features are scaled (for model selection).
    
    Returns:
    --------
    Dict containing all models, metrics, and results
    """
    logger.info("="*60)
    logger.info("TRAINING ALL MODELS")
    logger.info("="*60)
    
    results = {}
    fitted_models = {}
    
    # Use scaled data for models that need it
    if scaled:
        X_train_use = X_train
        X_test_use = X_test
    else:
        X_train_use = X_train
        X_test_use = X_test
    
    # 1. Logistic Regression (needs scaled data)
    if scaled:
        model, metrics = train_logistic_regression(X_train_use, y_train, X_test_use, y_test)
        results['Logistic Regression'] = metrics
        fitted_models['Logistic Regression'] = model
    
    # 2. Decision Tree
    model, metrics = train_decision_tree(X_train_use, y_train, X_test_use, y_test)
    results['Decision Tree'] = metrics
    fitted_models['Decision Tree'] = model
    
    # 3. Random Forest
    model, metrics = train_random_forest(X_train_use, y_train, X_test_use, y_test)
    results['Random Forest'] = metrics
    fitted_models['Random Forest'] = model
    
    # 4. Gradient Boosting
    model, metrics = train_gradient_boosting(X_train_use, y_train, X_test_use, y_test)
    results['Gradient Boosting'] = metrics
    fitted_models['Gradient Boosting'] = model
    
    # 5. KNN (needs scaled data)
    if scaled:
        model, metrics = train_knn(X_train_use, y_train, X_test_use, y_test)
        results['KNN'] = metrics
        fitted_models['KNN'] = model
    
    # 6. Gaussian NB (uses raw data with SMOTE)
    model, metrics = train_gaussian_nb(X_train_use, y_train, X_test_use, y_test)
    results['Gaussian NB'] = metrics
    fitted_models['Gaussian NB'] = model
    
    # Create results dataframe
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('ROC-AUC', ascending=False)
    
    logger.info("="*60)
    logger.info("ALL MODELS TRAINED")
    logger.info("="*60)
    
    return {
        'results': results_df,
        'fitted_models': fitted_models,
        'best_model_name': results_df.index[0],
        'best_model': fitted_models[results_df.index[0]]
    }


def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    param_dist: Dict = None,
    n_iter: int = 6,
    cv: int = 3,
    sample_size: int = 60000,
    random_state: int = 42
) -> Tuple[Any, Dict]:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    model_type : str
        Type of model to tune ('random_forest' or 'gradient_boosting').
    param_dist : Dict
        Parameter distribution for search.
    n_iter : int
        Number of parameter settings sampled.
    cv : int
        Number of cross-validation folds.
    sample_size : int
        Size of stratified sample for tuning.
    random_state : int
        Random seed.
    
    Returns:
    --------
    Tuple of (best estimator, best params)
    """
    logger.info(f"Starting hyperparameter tuning for {model_type}...")
    t0 = time.time()
    
    # Stratified sample for speed
    n_pos = int(sample_size * y_train.mean())
    n_neg = sample_size - n_pos
    
    idx_sample = pd.concat([
        y_train[y_train == 0].sample(n_neg, random_state=random_state),
        y_train[y_train == 1].sample(n_pos, random_state=random_state)
    ]).index
    
    X_tune = X_train.loc[idx_sample]
    y_tune = y_train.loc[idx_sample]
    
    # Define model and parameter distributions
    if model_type == 'random_forest':
        if param_dist is None:
            param_dist = {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10, 12],
                'min_samples_leaf': [10, 20, 50],
                'max_features': ['sqrt', 'log2']
            }
        model = RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1)
    
    elif model_type == 'gradient_boosting':
        if param_dist is None:
            param_dist = {
                'max_iter': [150, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.03, 0.05, 0.08, 0.1],
                'l2_regularization': [0, 0.5, 1.0]
            }
        model = HistGradientBoostingClassifier(class_weight='balanced', random_state=random_state)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Perform search
    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        random_state=random_state,
        n_jobs=-1
    )
    
    search.fit(X_tune, y_tune)
    
    logger.info(f"Tuning completed in {time.time()-t0:.1f}s")
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV ROC-AUC: {search.best_score_:.5f}")
    
    return search.best_estimator_, search.best_params_


def train_tuned_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Dict,
    model_type: str = 'random_forest'
) -> Tuple[Any, Dict]:
    """
    Train a model with best hyperparameters on full training data.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    best_params : Dict
        Best hyperparameters from tuning.
    model_type : str
        Type of model.
    
    Returns:
    --------
    Tuple of (trained model, metrics dict)
    """
    logger.info(f"Training tuned {model_type} on full data...")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            **best_params,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = HistGradientBoostingClassifier(
            **best_params,
            class_weight='balanced',
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.fit(X_train, y_train)
    
    from src.evaluate_model import evaluate_model
    metrics = evaluate_model(model, X_test, y_test)
    
    logger.info(f"Tuned {model_type} - ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    return model, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Training module loaded successfully")