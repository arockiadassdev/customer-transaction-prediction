"""
Data preprocessing module for Customer Transaction Prediction.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import get_project_root

logger = logging.getLogger(__name__)


def load_raw_data(filename: str = 'train.csv') -> pd.DataFrame:
    """
    Load raw data from data/raw directory.
    
    Parameters:
    -----------
    filename : str
        Name of the CSV file in data/raw.
    
    Returns:
    --------
    pd.DataFrame
        Raw dataframe.
    """
    root = get_project_root()
    filepath = f"{root}/data/raw/{filename}"
    logger.info(f"Loading raw data from {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Raw data loaded. Shape: {df.shape}")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing unnecessary columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe.
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe.
    """
    logger.info("Starting data cleaning...")
    
    # Drop ID_code column (no predictive value)
    df_clean = df.drop(columns=['ID_code']).copy()
    
    # Log data quality metrics
    missing = df_clean.isnull().sum().sum()
    duplicates = df_clean.duplicated().sum()
    
    logger.info(f"Missing values: {missing}")
    logger.info(f"Duplicate rows: {duplicates}")
    logger.info(f"Cleaned data shape: {df_clean.shape}")
    
    return df_clean


def split_train_test(
    df: pd.DataFrame,
    target_col: str = 'target',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe.
    target_col : str
        Name of target column.
    test_size : float
        Proportion of data for test set.
    random_state : int
        Random seed for reproducibility.
    
    Returns:
    --------
    Tuple of X_train, X_test, y_train, y_test
    """
    logger.info("Splitting data into train and test sets...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Train class distribution:\n{y_train.value_counts(normalize=True).round(4)}")
    
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    
    Returns:
    --------
    Tuple of X_train_scaled, X_test_scaled, scaler
    """
    logger.info("Scaling features with StandardScaler...")
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    logger.info("Feature scaling completed")
    
    return X_train_scaled, X_test_scaled, scaler


def get_feature_columns(df: pd.DataFrame, target_col: str = 'target') -> List[str]:
    """
    Get list of feature column names.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe.
    target_col : str
        Name of target column to exclude.
    
    Returns:
    --------
    List[str]
        List of feature column names.
    """
    return [col for col in df.columns if col != target_col]


def preprocess_pipeline(
    filename: str = 'train.csv',
    target_col: str = 'target',
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    filename : str
        Raw data filename.
    target_col : str
        Target column name.
    test_size : float
        Test set proportion.
    random_state : int
        Random seed.
    scale : bool
        Whether to scale features.
    
    Returns:
    --------
    Dict containing all preprocessing outputs
    """
    logger.info("="*60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("="*60)
    
    # Load and clean data
    df = load_raw_data(filename)
    df_clean = clean_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_train_test(
        df_clean, target_col, test_size, random_state
    )
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_cols': get_feature_columns(df_clean, target_col),
        'df_clean': df_clean
    }
    
    # Scale features if requested
    if scale:
        X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
        result['X_train_scaled'] = X_train_sc
        result['X_test_scaled'] = X_test_sc
        result['scaler'] = scaler
    
    logger.info("="*60)
    logger.info("PREPROCESSING PIPELINE COMPLETED")
    logger.info("="*60)
    
    return result


if __name__ == "__main__":
    # Test the preprocessing pipeline
    logging.basicConfig(level=logging.INFO)
    results = preprocess_pipeline()
    print(f"Pipeline output keys: {results.keys()}")