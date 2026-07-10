"""
Model evaluation module for Customer Transaction Prediction.
"""

import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

from src.utils import get_project_root

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a trained model and return comprehensive metrics.
    
    Parameters:
    -----------
    model : Any
        Trained model object.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    threshold : float
        Decision threshold for binary classification.
    
    Returns:
    --------
    Dict containing all evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Avg Prec': average_precision_score(y_test, y_proba)
    }
    
    logger.info(f"Model evaluation completed. ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    return metrics


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'roc_auc',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform stratified k-fold cross-validation.
    
    Parameters:
    -----------
    model : Any
        Model object to evaluate.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    cv : int
        Number of folds.
    scoring : str
        Scoring metric.
    random_state : int
        Random seed.
    
    Returns:
    --------
    Dict containing CV scores
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    
    logger.info(f"Performing {cv}-fold stratified cross-validation...")
    t0 = time.time()
    
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
    
    cv_results = {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
        'cv': cv
    }
    
    logger.info(f"CV completed in {time.time()-t0:.1f}s")
    logger.info(f"Mean {scoring}: {scores.mean():.5f} (±{scores.std():.5f})")
    
    return cv_results


def plot_roc_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_modes: Dict[str, str] = None,
    figsize: Tuple[int, int] = (10, 7)
) -> None:
    """
    Plot ROC curves for multiple models.
    
    Parameters:
    -----------
    models : Dict[str, Any]
        Dictionary of {model_name: model_object}.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    model_modes : Dict[str, str]
        Dictionary indicating if model needs scaled data.
    figsize : Tuple[int, int]
        Figure size.
    """
    logger.info("Plotting ROC curves...")
    
    plt.figure(figsize=figsize)
    
    for name, model in models.items():
        # Use scaled or raw data based on model type
        if model_modes and name in model_modes and 'scaled' in model_modes[name]:
            X_use = X_test
        else:
            X_use = X_test
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_use)[:, 1]
        else:
            proba = model.decision_function(X_use)
        
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — All Models')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    
    # Save figure
    root = get_project_root()
    save_path = f"{root}/images/roc_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"ROC curves saved to {save_path}")
    plt.show()


def plot_precision_recall_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_modes: Dict[str, str] = None,
    figsize: Tuple[int, int] = (10, 7)
) -> None:
    """
    Plot Precision-Recall curves for multiple models.
    
    Parameters:
    -----------
    models : Dict[str, Any]
        Dictionary of {model_name: model_object}.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    model_modes : Dict[str, str]
        Dictionary indicating if model needs scaled data.
    figsize : Tuple[int, int]
        Figure size.
    """
    logger.info("Plotting Precision-Recall curves...")
    
    plt.figure(figsize=figsize)
    
    baseline = y_test.mean()
    plt.axhline(baseline, color='k', linestyle='--', label=f'Baseline (={baseline:.3f})')
    
    for name, model in models.items():
        # Use scaled or raw data
        if model_modes and name in model_modes and 'scaled' in model_modes[name]:
            X_use = X_test
        else:
            X_use = X_test
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_use)[:, 1]
        else:
            proba = model.decision_function(X_use)
        
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        
        plt.plot(rec, prec, label=f'{name} (AP={ap:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves — All Models')
    plt.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    
    # Save figure
    root = get_project_root()
    save_path = f"{root}/images/precision_recall_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Precision-Recall curves saved to {save_path}")
    plt.show()


def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_modes: Dict[str, str] = None,
    figsize: Tuple[int, int] = (16, 9)
) -> None:
    """
    Plot confusion matrices for multiple models.
    
    Parameters:
    -----------
    models : Dict[str, Any]
        Dictionary of {model_name: model_object}.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    model_modes : Dict[str, str]
        Dictionary indicating if model needs scaled data.
    figsize : Tuple[int, int]
        Figure size.
    """
    logger.info("Plotting confusion matrices...")
    
    n_models = len(models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        # Use scaled or raw data
        if model_modes and name in model_modes and 'scaled' in model_modes[name]:
            X_use = X_test
        else:
            X_use = X_test
        
        y_pred = model.predict(X_use)
        cm = confusion_matrix(y_test, y_pred)
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices — All Models', fontsize=13)
    plt.tight_layout()
    
    # Save figure
    root = get_project_root()
    save_path = f"{root}/images/confusion_matrices.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrices saved to {save_path}")
    plt.show()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : Any
        Trained model with feature_importances_ attribute.
    feature_names : List[str]
        List of feature names.
    top_n : int
        Number of top features to display.
    figsize : Tuple[int, int]
        Figure size.
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    logger.info(f"Plotting top {top_n} feature importances...")
    
    fi = pd.Series(model.feature_importances_, index=feature_names)
    fi = fi.sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    fi.plot(kind='barh', color='darkcyan')
    plt.gca().invert_yaxis()
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    # Save figure
    root = get_project_root()
    save_path = f"{root}/images/feature_importance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Feature importance plot saved to {save_path}")
    plt.show()


def compare_models(results_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Create comparison plots for all models.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with model names as index and metrics as columns.
    figsize : Tuple[int, int]
        Figure size.
    """
    logger.info("Creating model comparison plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ROC-AUC comparison
    results_df['ROC-AUC'].sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Model Comparison — ROC-AUC')
    axes[0].set_xlabel('ROC-AUC')
    axes[0].set_xlim(0.5, 1.0)
    
    # Average Precision comparison
    results_df['Avg Prec'].sort_values().plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_title('Model Comparison — Average Precision')
    axes[1].set_xlabel('Average Precision')
    axes[1].set_xlim(0, 1.0)
    
    plt.tight_layout()
    
    # Save figure
    root = get_project_root()
    save_path = f"{root}/images/model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Model comparison plot saved to {save_path}")
    plt.show()


def print_classification_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_mode: str = 'raw'
) -> None:
    """
    Print detailed classification report.
    
    Parameters:
    -----------
    model : Any
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    model_mode : str
        Whether model expects scaled data.
    """
    y_pred = model.predict(X_test)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, digits=4))
    print("="*60 + "\n")


def full_evaluation_pipeline(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    model_modes: Dict[str, str] = None,
    save_plots: bool = True
) -> pd.DataFrame:
    """
    Run complete evaluation pipeline on all models.
    
    Parameters:
    -----------
    models : Dict[str, Any]
        Dictionary of {model_name: model_object}.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    feature_names : List[str]
        List of feature names.
    model_modes : Dict[str, str]
        Dictionary indicating if model needs scaled data.
    save_plots : bool
        Whether to save plots to disk.
    
    Returns:
    --------
    pd.DataFrame with all model metrics
    """
    logger.info("="*60)
    logger.info("FULL EVALUATION PIPELINE")
    logger.info("="*60)
    
    # Evaluate each model
    all_metrics = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        all_metrics[name] = metrics
    
    results_df = pd.DataFrame(all_metrics).T.sort_values('ROC-AUC', ascending=False)
    
    # Print comparison
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    print(results_df.round(4))
    
    # Generate plots
    if save_plots:
        plot_roc_curves(models, X_test, y_test, model_modes)
        plot_precision_recall_curves(models, X_test, y_test, model_modes)
        plot_confusion_matrices(models, X_test, y_test, model_modes)
        compare_models(results_df)
        
        # Plot feature importance for best model
        best_model_name = results_df.index[0]
        best_model = models[best_model_name]
        if hasattr(best_model, 'feature_importances_'):
            plot_feature_importance(best_model, feature_names)
    
    logger.info("="*60)
    logger.info("EVALUATION COMPLETED")
    logger.info("="*60)
    
    return results_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Evaluation module loaded successfully")