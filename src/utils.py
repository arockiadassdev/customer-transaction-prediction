"""
Utility functions for Customer Transaction Prediction project.
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file.
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe.
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save processed data to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Processed dataframe to save.
    filepath : str
        Path to save the CSV file.
    """
    logger.info(f"Saving processed data to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Data saved successfully. Shape: {df.shape}")


def save_model(model: Any, filepath: str) -> None:
    """
    Save trained model using joblib.
    
    Parameters:
    -----------
    model : Any
        Trained model object.
    filepath : str
        Path to save the model.
    """
    logger.info(f"Saving model to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Model saved successfully")


def load_model(filepath: str) -> Any:
    """
    Load saved model from file.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model file.
    
    Returns:
    --------
    Any
        Loaded model object.
    """
    logger.info(f"Loading model from {filepath}")
    model = joblib.load(filepath)
    logger.info("Model loaded successfully")
    return model


def save_features(features: List[str], filepath: str) -> None:
    """
    Save selected features list to file.
    
    Parameters:
    -----------
    features : List[str]
        List of selected feature names.
    filepath : str
        Path to save the features list.
    """
    logger.info(f"Saving features to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(features, filepath)
    logger.info(f"Saved {len(features)} features")


def load_features(filepath: str) -> List[str]:
    """
    Load saved features list from file.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved features file.
    
    Returns:
    --------
    List[str]
        List of feature names.
    """
    logger.info(f"Loading features from {filepath}")
    features = joblib.load(filepath)
    logger.info(f"Loaded {len(features)} features")
    return features


def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
    --------
    str
        Absolute path to project root.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_directory_structure() -> None:
    """Create all required project directories."""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'reports',
        'images'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(get_project_root(), directory), exist_ok=True)
    
    logger.info("Project directory structure created/verified")


if __name__ == "__main__":
    create_directory_structure()