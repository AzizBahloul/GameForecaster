import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class PredictionResult:
    """Data class to store prediction results."""
    prediction: bool
    confidence: float
    feature_importance: Dict[str, float]
    top_influential_features: List[Tuple[str, float]]

class GamePredictor:
    """
    A class for predicting game success using trained ML models.
    
    Attributes:
        model_dir (Path): Directory containing model artifacts
        categorical_features (List[str]): List of categorical feature names
        numerical_features (List[str]): List of numerical feature names
        features (List[str]): Combined list of all features
    """
    
    def __init__(self, model_dir: Union[str, Path] = 'model_artifacts'):
        self.model_dir = Path(model_dir)
        self.categorical_features = ['Genre', 'Publisher', 'Developer', 'Rating']
        self.numerical_features = [
            'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',
            'Global_Sales', 'Critic_Score', 'Critic_Count',
            'User_Score', 'User_Count'
        ]
        self.features = self.categorical_features + self.numerical_features
        self._load_model()
        self._validate_model()

    def _load_model(self) -> None:
        """Load and validate all necessary model artifacts."""
        try:
            self.model = joblib.load(self.model_dir / 'model.pkl')
            self.label_encoders = joblib.load(self.model_dir / 'label_encoders.pkl')
            self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
            
            with open(self.model_dir / 'model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
                
        except FileNotFoundError as e:
            raise RuntimeError(f"Missing model artifact: {e.filename}")
        except json.JSONDecodeError:
            raise RuntimeError("Invalid model metadata file")
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

    def _validate_model(self) -> None:
        """Validate model artifacts and features."""
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model must support probability predictions")
            
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must support feature importances")
            
        # Validate label encoders
        missing_encoders = set(self.categorical_features) - set(self.label_encoders.keys())
        if missing_encoders:
            raise ValueError(f"Missing label encoders for: {missing_encoders}")

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna('unknown')
                
        for col in self.numerical_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(self.metadata.get(f'{col}_mean', 0))
                
        return df

    def _validate_input_features(self, game_features: Dict) -> None:
        """Validate input features."""
        missing_required = set(self.features) - set(game_features.keys())
        if missing_required:
            raise ValueError(f"Missing required features: {missing_required}")
            
        for feature, value in game_features.items():
            if feature in self.numerical_features and not isinstance(value, (int, float)):
                raise ValueError(f"Feature {feature} must be numeric, got {type(value)}")

    def preprocess_features(self, game_features: Dict) -> pd.DataFrame:
        """
        Preprocess input features for prediction.
        
        Args:
            game_features: Dictionary containing game features
            
        Returns:
            Preprocessed features as a DataFrame
            
        Raises:
            ValueError: If feature preprocessing fails
        """
        try:
            self._validate_input_features(game_features)
            df = pd.DataFrame([game_features])
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Scale numerical features
            df[self.numerical_features] = self.scaler.transform(df[self.numerical_features])
            
            # Encode categorical features
            for col in self.categorical_features:
                if col in df.columns:
                    try:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError as e:
                        raise ValueError(f"Unknown category in {col}: {df[col].iloc[0]}")
            
            return df[self.features]
            
        except Exception as e:
            raise ValueError(f"Feature preprocessing failed: {str(e)}")

    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get the top n most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples
        """
        importances = list(zip(self.features, self.model.feature_importances_))
        return sorted(importances, key=lambda x: x[1], reverse=True)[:n]

    def predict(self, game_features: Dict, threshold: float = 0.5) -> PredictionResult:
        """
        Make a prediction for a single game.
        
        Args:
            game_features: Dictionary containing game features
            threshold: Probability threshold for positive prediction
            
        Returns:
            PredictionResult object containing prediction details
        """
        processed_features = self.preprocess_features(game_features)
        prediction_prob = self.model.predict_proba(processed_features)[0]
        prediction = bool(prediction_prob[1] > threshold)
        confidence = float(prediction_prob[1] if prediction else prediction_prob[0])
        
        feature_importance = dict(zip(self.features, self.model.feature_importances_))
        top_features = self.get_top_features()
        
        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            feature_importance=feature_importance,
            top_influential_features=top_features
        )

    def batch_predict(self, games_features: List[Dict]) -> List[PredictionResult]:
        """
        Make predictions for multiple games.
        
        Args:
            games_features: List of dictionaries containing game features
            
        Returns:
            List of PredictionResult objects
        """
        return [self.predict(game_features) for game_features in games_features]