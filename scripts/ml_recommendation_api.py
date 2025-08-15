import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class MovieRecommendationEngine:
    def __init__(self):
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load trained ML models and data"""
        try:
            models_dir = 'models'
            
            # Load models
            with open(f'{models_dir}/kmeans_model.pkl', 'rb') as f:
                self.kmeans = pickle.load(f)
            
            with open(f'{models_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(f'{models_dir}/rf_classifier.pkl', 'rb') as f:
                self.rf_classifier = pickle.load(f)
            
            with open(f'{models_dir}/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            with open(f'{models_dir}/tfidf_genre.pkl', 'rb') as f:
                self.tfidf_genre = pickle.load(f)
            
            # Load movie database
            with open(f'{models_dir}/movie_database.json', 'r') as f:
                self.movies_data = json.load(f)
            
            self.movies_df = pd.DataFrame(self.movies_data)
            self.models_loaded = True
            print(f"✅ Loaded {len(self.movies_data)} movies and ML models")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.models_loaded = False
    
    def get_movie_features(self, movie_data):
        """Extract features for a movie"""
        # Genre features
        genre_features = self.tfidf_genre.transform([movie_data.get('genre', '')])
        
        # Numerical features
        numerical_features = np.array([[
            movie_data.get('rating', 7.0),
            len(movie_data.get('title', '')),
            movie_data.get('year', 2000)
        ]])
        
        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([numerical_features, genre_features])
        
        return combined_features
    
    def get_cluster_recommendations(self, user_preferences, limit=10, min_rating=0):
        """Get recommendations using clustering"""
        if not self.models_loaded:
            return []
        
        try:
            # Filter movies by minimum rating
            filtered_movies = [m for m in self.movies_data if m.get('rating', 0) >= min_rating]
            
            if not filtered_movies:
                return []
            
            # If user preferences provided, find similar cluster
            if user_preferences:
                # Create user feature vector
                user_features = self.get_movie_features(user_preferences)
                user_features_scaled = self.scaler.transform(user_features)
                user_cluster = self.kmeans.predict(user_features_scaled)[0]
                
                # Get movies from same cluster
                cluster_movies = []
                for movie in filtered_movies:
                    movie_features = self.get_movie_features(movie)
                    movie_features_scaled = self.scaler.transform(movie_features)
                    movie_cluster = self.kmeans.predict(movie_features_scaled)[0]
                    
                    if movie_cluster == user_cluster:
                        cluster_movies.append(movie)
                
                # If not enough movies in cluster, add from other clusters
                if len(cluster_movies) < limit:
                    other_movies = [m for m in filtered_movies if m not in cluster_movies]
                    cluster_movies.extend(other_movies[:limit - len(cluster_movies)])
                
                recommendations = cluster_movies[:limit]
            else:
                # No user preferences, return top-rated movies
                recommendations = sorted(filtered_movies, key=lambda x: x.get('rating', 0), reverse=True)[:limit]
            
            return recommendations
            
        except Exception as e:
            print(f"Error in clustering recommendations: {e}")
            return []
    
    def get_classification_recommendations(self, target_category='High', limit=10, min_rating=0):
        """Get recommendations using classification"""
        if not self.models_loaded:
            return []
        
        try:
            # Filter movies by minimum rating
            filtered_movies = [m for m in self.movies_data if m.get('rating', 0) >= min_rating]
            
            if not filtered_movies:
                return []
            
            # Predict categories for all movies
            movie_predictions = []
            for movie in filtered_movies:
                movie_features = self.get_movie_features(movie)
                predicted_category = self.rf_classifier.predict(movie_features)[0]
                predicted_proba = self.rf_classifier.predict_proba(movie_features)[0]
                
                category_name = self.label_encoder.inverse_transform([predicted_category])[0]
                confidence = max(predicted_proba)
                
                movie_predictions.append({
                    'movie': movie,
                    'predicted_category': category_name,
                    'confidence': confidence
                })
            
            # Filter by target category and sort by confidence
            target_movies = [
                mp for mp in movie_predictions 
                if mp['predicted_category'] == target_category
            ]
            
            # Sort by confidence and rating
            target_movies.sort(key=lambda x: (x['confidence'], x['movie'].get('rating', 0)), reverse=True)
            
            # If not enough movies in target category, add from other categories
            if len(target_movies) < limit:
                other_movies = [mp for mp in movie_predictions if mp not in target_movies]
                other_movies.sort(key=lambda x: (x['confidence'], x['movie'].get('rating', 0)), reverse=True)
                target_movies.extend(other_movies[:limit - len(target_movies)])
            
            recommendations = [mp['movie'] for mp in target_movies[:limit]]
            return recommendations
            
        except Exception as e:
            print(f"Error in classification recommendations: {e}")
            return []
    
    def get_hybrid_recommendations(self, user_id=None, limit=10, min_rating=0):
        """Get hybrid recommendations combining clustering and classification"""
        if not self.models_loaded:
            return self.get_fallback_recommendations(limit, min_rating)
        
        try:
            # Get clustering recommendations
            user_prefs = None
            if user_id:
                # Simulate user preferences based on user_id
                user_prefs = {
                    'genre': 'Drama, Action' if int(user_id) % 2 == 0 else 'Comedy, Romance',
                    'rating': 8.0 + (int(user_id) % 3) * 0.5,
                    'year': 2000 + (int(user_id) % 20),
                    'title': f'User{user_id} Preference'
                }
            
            cluster_recs = self.get_cluster_recommendations(user_prefs, limit//2, min_rating)
            
            # Get classification recommendations
            target_category = 'Excellent' if min_rating >= 8 else 'High'
            class_recs = self.get_classification_recommendations(target_category, limit//2, min_rating)
            
            # Combine and deduplicate
            all_recs = cluster_recs + class_recs
            seen_titles = set()
            unique_recs = []
            
            for movie in all_recs:
                if movie['title'] not in seen_titles:
                    seen_titles.add(movie['title'])
                    unique_recs.append(movie)
            
            # Add like probability based on user preferences
            for movie in unique_recs:
                if user_id:
                    # Simple probability calculation based on rating and user_id
                    base_prob = min(movie.get('rating', 7) / 10, 0.95)
                    user_factor = (int(user_id) % 10) / 10 * 0.2
                    movie['like_probability'] = min(base_prob + user_factor, 0.98)
                else:
                    movie['like_probability'] = min(movie.get('rating', 7) / 10, 0.95)
            
            return unique_recs[:limit]
            
        except Exception as e:
            print(f"Error in hybrid recommendations: {e}")
            return self.get_fallback_recommendations(limit, min_rating)
    
    def get_fallback_recommendations(self, limit=10, min_rating=0):
        """Fallback recommendations when ML models fail"""
        fallback_movies = [
            {
                "title": "The Shawshank Redemption",
                "genre": "Drama",
                "rating": 9.3,
                "poster_url": "/shawshank-redemption-poster.png",
                "like_probability": 0.95
            },
            {
                "title": "The Godfather",
                "genre": "Crime, Drama",
                "rating": 9.2,
                "poster_url": "/classic-mob-poster.png",
                "like_probability": 0.92
            },
            {
                "title": "The Dark Knight",
                "genre": "Action, Crime, Drama",
                "rating": 9.0,
                "poster_url": "/dark-knight-poster.png",
                "like_probability": 0.90
            },
            {
                "title": "Pulp Fiction",
                "genre": "Crime, Drama",
                "rating": 8.9,
                "poster_url": "/pulp-fiction-poster.png",
                "like_probability": 0.89
            },
            {
                "title": "Forrest Gump",
                "genre": "Drama, Romance",
                "rating": 8.8,
                "poster_url": "/forrest-gump-poster.png",
                "like_probability": 0.88
            }
        ]
        
        # Filter by minimum rating and limit
        filtered = [m for m in fallback_movies if m['rating'] >= min_rating]
        return filtered[:limit]

# Global instance
recommendation_engine = MovieRecommendationEngine()

def get_recommendations(user_id=None, limit=10, min_rating=0):
    """Main function to get movie recommendations"""
    return recommendation_engine.get_hybrid_recommendations(user_id, limit, min_rating)

if __name__ == "__main__":
    # Test the recommendation engine
    print("Testing recommendation engine...")
    
    # Test without user ID
    recs = get_recommendations(limit=5, min_rating=8.0)
    print(f"General recommendations: {len(recs)} movies")
    
    # Test with user ID
    user_recs = get_recommendations(user_id="123", limit=8, min_rating=7.0)
    print(f"User 123 recommendations: {len(user_recs)} movies")
