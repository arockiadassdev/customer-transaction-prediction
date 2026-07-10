"""
Prediction module for Customer Transaction Prediction.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from typing import Dict, Any, Union, List

from src.utils import get_project_root, load_model, load_features

logger = logging.getLogger(__name__)


class TransactionPredictor:
    """
    Class for making predictions on customer transaction data.
    """
    
    def __init__(
        self,
        model_path: str = None,
        scaler_path: str = None,
        features_path: str = None
    ):
        """
        Initialize predictor with saved model artifacts.
        
        Parameters:
        -----------
        model_path : str
            Path to saved model file.
        scaler_path : str
            Path to saved scaler file.
        features_path : str
            Path to saved features list file.
        """
        self.model = None
        self.scaler = None
        self.selected_features = None
        
        # Default paths
        if model_path is None:
            root = get_project_root()
            # Try to find the best model (gradient boosting usually performs best)
            import os
            model_files = [f for f in os.listdir(f"{root}/models") if f.endswith('.pkl')]
            if model_files:
                # Prefer gradient boosting model
                gb_models = [f for f in model_files if 'gradient' in f.lower() or 'boosting' in f.lower()]
                if gb_models:
                    model_path = f"{root}/models/{gb_models[0]}"
                else:
                    model_path = f"{root}/models/{model_files[0]}"
            else:
                raise FileNotFoundError("No model files found in models directory")
        
        if scaler_path is None:
            scaler_path = f"{get_project_root()}/models/scaler.pkl"
        
        if features_path is None:
            features_path = f"{get_project_root()}/models/selected_features.pkl"
        
        # Load artifacts
        self._load_artifacts(model_path, scaler_path, features_path)
    
    def _load_artifacts(
        self,
        model_path: str,
        scaler_path: str,
        features_path: str
    ) -> None:
        """
        Load model, scaler, and features from disk.
        
        Parameters:
        -----------
        model_path : str
            Path to model file.
        scaler_path : str
            Path to scaler file.
        features_path : str
            Path to features file.
        """
        try:
            self.model = load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            try:
                self.scaler = load_model(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            except FileNotFoundError:
                logger.warning("Scaler not found - assuming model doesn't require scaling")
                self.scaler = None
            
            try:
                self.selected_features = load_features(features_path)
                logger.info(f"Features loaded from {features_path} ({len(self.selected_features)} features)")
            except FileNotFoundError:
                logger.warning("Features list not found - using all features")
                self.selected_features = None
                
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise
    
    def preprocess_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Raw input features.
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed features.
        """
        # Select features if specified
        if self.selected_features is not None:
            # Ensure all required features are present
            missing_features = set(self.selected_features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            X = X[self.selected_features]
        
        # Scale if scaler is available
        if self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        
        Returns:
        --------
        np.ndarray
            Binary predictions (0 or 1).
        """
        X_processed = self.preprocess_input(X)
        predictions = self.model.predict(X_processed)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        
        Returns:
        --------
        np.ndarray
            Probability of class 1 (transaction).
        """
        X_processed = self.preprocess_input(X)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_processed)[:, 1]
        else:
            # For models without predict_proba, use decision function
            decision = self.model.decision_function(X_processed)
            # Convert to probability using sigmoid
            proba = 1 / (1 + np.exp(-decision))
        
        return proba
    
    def predict_with_threshold(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Make predictions with custom threshold.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        threshold : float
            Classification threshold.
        
        Returns:
        --------
        np.ndarray
            Binary predictions.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def predict_single(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Make prediction for a single customer.
        
        Parameters:
        -----------
        features : Dict[str, float]
            Dictionary of feature names and values.
        
        Returns:
        --------
        Dict with prediction results
        """
        X = pd.DataFrame([features])
        
        prediction = self.predict(X)[0]
        probability = self.predict_proba(X)[0]
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'will_transact': bool(prediction == 1),
            'confidence': float(max(probability, 1 - probability))
        }
    
    def batch_predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for a batch of customers.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        return_proba : bool
            Whether to include probabilities.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions.
        """
        predictions = self.predict(X)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'will_transact': predictions == 1
        })
        
        if return_proba:
            results['probability'] = self.predict_proba(X)
            results['confidence'] = results['probability'].apply(
                lambda p: max(p, 1-p)
            )
        
        return results
    
    def predict_with_details(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions with detailed output.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions and probabilities.
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'will_transact': predictions == 1,
            'probability': probabilities,
            'confidence': [max(p, 1-p) for p in probabilities]
        })
        
        return results


def make_prediction(
    input_data: Union[pd.DataFrame, Dict],
    model_path: str = None,
    scaler_path: str = None,
    features_path: str = None,
    return_proba: bool = True
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Convenience function for making predictions.
    
    Parameters:
    -----------
    input_data : pd.DataFrame or Dict
        Input features.
    model_path : str
        Path to model file.
    scaler_path : str
        Path to scaler file.
    features_path : str
        Path to features file.
    return_proba : bool
        Whether to return probabilities.
    
    Returns:
    --------
    Predictions array or DataFrame
    """
    predictor = TransactionPredictor(model_path, scaler_path, features_path)
    
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    if return_proba:
        return predictor.predict_with_details(input_data)
    else:
        return predictor.predict(input_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Prediction module loaded successfully")