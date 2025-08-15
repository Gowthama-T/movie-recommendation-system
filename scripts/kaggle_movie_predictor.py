import pickle
import pandas as pd
import numpy as np
import os

class KaggleMoviePredictor:
    """Movie recommendation predictor using the trained Kaggle model"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.label_encoders = None
        self.load_model_artifacts()
    
    def load_model_artifacts(self):
        """Load all trained model artifacts"""
        try:
            # Load model
            with open(f'{self.models_dir}/kaggle_movie_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(f'{self.models_dir}/kaggle_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature columns
            with open(f'{self.models_dir}/kaggle_feature_cols.pkl', 'rb') as f:
                self.feature_cols = pickle.load(f)
            
            # Load label encoders
            with open(f'{self.models_dir}/kaggle_label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            print("✅ Model artifacts loaded successfully")
            
        except FileNotFoundError as e:
            print(f"❌ Model artifacts not found: {e}")
            print("Please run kaggle_movie_model.py first to train the model")
    
    def preprocess_movie_data(self, movie_data):
        """Preprocess movie data for prediction"""
        
        if not isinstance(movie_data, pd.DataFrame):
            movie_data = pd.DataFrame([movie_data])
        
        # Create a copy
        processed_data = movie_data.copy()
        
        # Apply label encoders to categorical columns
        for col, encoder in self.label_encoders.items():
            if col in processed_data.columns:
                # Handle unseen categories
                processed_data[col] = processed_data[col].astype(str)
                mask = processed_data[col].isin(encoder.classes_)
                processed_data.loc[mask, col] = encoder.transform(processed_data.loc[mask, col])
                processed_data.loc[~mask, col] = -1  # Unknown category
        
        # Ensure all feature columns are present
        for col in self.feature_cols:
            if col not in processed_data.columns:
                processed_data[col] = 0  # Default value for missing features
        
        # Select only the feature columns used in training
        processed_data = processed_data[self.feature_cols]
        
        return processed_data
    
    def predict_rating(self, movie_data):
        """Predict rating for given movie(s)"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")
        
        # Preprocess the data
        processed_data = self.preprocess_movie_data(movie_data)
        
        # Scale the features
        scaled_data = self.scaler.transform(processed_data)
        
        # Make predictions
        predictions = self.model.predict(scaled_data)
        
        return predictions
    
    def get_movie_recommendations(self, movies_df, top_n=10):
        """Get top N movie recommendations from a dataset"""
        
        # Predict ratings for all movies
        predicted_ratings = self.predict_rating(movies_df)
        
        # Add predictions to the dataframe
        movies_with_predictions = movies_df.copy()
        movies_with_predictions['predicted_rating'] = predicted_ratings
        
        # Sort by predicted rating and return top N
        recommendations = movies_with_predictions.nlargest(top_n, 'predicted_rating')
        
        return recommendations

def test_predictor():
    """Test the movie predictor with sample data"""
    
    predictor = KaggleMoviePredictor()
    
    if predictor.model is None:
        print("Model not available for testing")
        return
    
    # Create sample movie data
    sample_movies = pd.DataFrame([
        {'genre': 'Action', 'year': 2020, 'budget': 100000000},
        {'genre': 'Comedy', 'year': 2019, 'budget': 50000000},
        {'genre': 'Drama', 'year': 2021, 'budget': 20000000},
    ])
    
    try:
        predictions = predictor.predict_rating(sample_movies)
        print("Sample predictions:")
        for i, pred in enumerate(predictions):
            print(f"  Movie {i+1}: {pred:.2f}")
    
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    test_predictor()
