import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer

class MovieRecommendationEngine:
    def __init__(self):
        """Initialize the recommendation engine with trained models and data"""
        # Determine base project directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from scripts/
        
        # Paths
        self.model_path = os.path.join(self.base_dir, 'models')
        self.data_path = os.path.join(self.model_path)  # models folder also has CSVs
        
        self.model = None
        self.scaler = None
        self.vectorizer = None
        self.feature_columns = None
        self.movies_db = None
        self.user_data = None
        
        self.load_models()
        self.load_data()
    
    def load_models(self):
        """Load the trained models and preprocessors"""
        try:
            self.model = joblib.load(os.path.join(self.model_path, 'model.joblib'))
            self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.joblib'))
            self.vectorizer = joblib.load(os.path.join(self.model_path, 'vectorizer.joblib'))
            self.feature_columns = joblib.load(os.path.join(self.model_path, 'feature_columns.joblib'))
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def load_data(self):
        """Load the movie database and user data"""
        try:
            movies_csv = os.path.join(self.data_path, 'movies_database.csv')
            self.movies_db = pd.read_csv(movies_csv)
            
            ratings_csv = os.path.join(self.base_dir, 'data', 'ml-latest-small', 'ratings.csv')
            movies_info_csv = os.path.join(self.base_dir, 'data', 'ml-latest-small', 'movies.csv')
            
            if os.path.exists(ratings_csv) and os.path.exists(movies_info_csv):
                ratings = pd.read_csv(ratings_csv)
                movies_info = pd.read_csv(movies_info_csv)
                self.user_data = pd.merge(ratings, movies_info, on='movieId')
            
            print("Data loaded successfully!")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_user_profile(self, user_id):
        """Get user profile statistics"""
        if self.user_data is None:
            return None
        
        user_ratings = self.user_data[self.user_data['userId'] == user_id]
        if len(user_ratings) == 0:
            return None
        
        profile = {
            'user_avg_rating': user_ratings['rating'].mean(),
            'user_rating_count': len(user_ratings),
            'user_like_ratio': (user_ratings['rating'] >= 4.0).mean(),
            'favorite_genres': user_ratings['genres'].str.split('|').explode().value_counts().head(3).to_dict()
        }
        return profile
    
    def prepare_movie_features(self, movie_row, user_profile):
        """Prepare features for a single movie prediction"""
        genre_features = self.vectorizer.transform([movie_row['genres']]).toarray()[0]
        
        features = {}
        for i, genre in enumerate(self.vectorizer.get_feature_names_out()):
            features[f'genre_{genre}'] = genre_features[i]
        
        # Movie statistics
        features['rating_normalized'] = (movie_row['avg_rating'] - 0.5) / 4.5
        features['movie_avg_rating'] = movie_row['avg_rating']
        features['movie_rating_count'] = movie_row['rating_count']
        features['movie_like_ratio'] = movie_row.get('like_probability', 0.5)
        
        # User statistics
        if user_profile:
            features['user_avg_rating'] = user_profile['user_avg_rating']
            features['user_rating_count'] = user_profile['user_rating_count']
            features['user_like_ratio'] = user_profile['user_like_ratio']
        else:
            features['user_avg_rating'] = 3.5
            features['user_rating_count'] = 50
            features['user_like_ratio'] = 0.6
        
        # Ensure all required features are present
        feature_vector = [features.get(col, 0) for col in self.feature_columns]
        
        return np.array(feature_vector).reshape(1, -1)
    
    def get_recommendations(self, user_id=None, num_recommendations=10, min_rating=3.5):
        """Get movie recommendations for a user"""
        try:
            user_profile = self.get_user_profile(user_id) if user_id else None
            candidate_movies = self.movies_db[self.movies_db['avg_rating'] >= min_rating].copy()
            
            if user_id and self.user_data is not None:
                user_movies = self.user_data[self.user_data['userId'] == user_id]['movieId'].tolist()
                candidate_movies = candidate_movies[~candidate_movies['movieId'].isin(user_movies)]
            
            predictions = []
            for _, movie in candidate_movies.iterrows():
                features = self.prepare_movie_features(movie, user_profile)
                features_scaled = self.scaler.transform(features)
                like_probability = self.model.predict_proba(features_scaled)[0][1]
                
                predictions.append({
                    'movieId': movie['movieId'],
                    'title': movie['title'],
                    'genres': movie['genres'],
                    'avg_rating': movie['avg_rating'],
                    'like_probability': like_probability,
                    'rating_count': movie['rating_count']
                })
            
            predictions.sort(key=lambda x: x['like_probability'], reverse=True)
            
            recommendations = []
            for pred in predictions[:num_recommendations]:
                primary_genre = pred['genres'].split('|')[0]
                poster_url = f"/placeholder.svg?height=300&width=200"
                recommendations.append({
                    'title': pred['title'],
                    'genre': primary_genre,
                    'rating': round(pred['avg_rating'], 1),
                    'like_probability': round(pred['like_probability'], 3),
                    'poster_url': poster_url,
                    'rating_count': pred['rating_count']
                })
            
            return {
                'user_id': user_id,
                'user_profile': user_profile,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return {'user_id': user_id, 'error': str(e), 'recommendations': []}


# Test the engine
if __name__ == "__main__":
    engine = MovieRecommendationEngine()
    result = engine.get_recommendations(user_id=1, num_recommendations=5)
    print("Recommendations for User 1:")
    for rec in result['recommendations']:
        print(f"- {rec['title']} ({rec['genre']}) - Rating: {rec['rating']}, Probability: {rec['like_probability']}")
    
    result = engine.get_recommendations(num_recommendations=5)
    print("\nGeneral Recommendations:")
    for rec in result['recommendations']:
        print(f"- {rec['title']} ({rec['genre']}) - Rating: {rec['rating']}, Probability: {rec['like_probability']}")
