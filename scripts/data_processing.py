import pandas as pd
import numpy as np
import requests
import zipfile
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def download_movielens_data():
    """Download and extract MovieLens dataset"""
    print("Downloading MovieLens dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download the small dataset (100k ratings)
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    response = requests.get(url)
    
    with open('data/ml-latest-small.zip', 'wb') as f:
        f.write(response.content)
    
    # Extract the zip file
    with zipfile.ZipFile('data/ml-latest-small.zip', 'r') as zip_ref:
        zip_ref.extractall('data/')
    
    print("Dataset downloaded and extracted successfully!")

def load_and_preprocess_data():
    """Load and preprocess the MovieLens data"""
    print("Loading and preprocessing data...")
    
    # Load the datasets
    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    movies = pd.read_csv('data/ml-latest-small/movies.csv')
    
    # Merge ratings with movies
    data = pd.merge(ratings, movies, on='movieId')
    
    # Handle missing values
    data = data.dropna()
    
    # Create binary target for classification (rating >= 4 is "like")
    data['liked'] = (data['rating'] >= 4.0).astype(int)
    
    # Process genres using Count Vectorizer
    vectorizer = CountVectorizer(token_pattern=r'[^|]+')
    genre_features = vectorizer.fit_transform(data['genres']).toarray()
    
    # Create genre feature columns
    genre_columns = [f'genre_{genre}' for genre in vectorizer.get_feature_names_out()]
    genre_df = pd.DataFrame(genre_features, columns=genre_columns, index=data.index)
    
    # Combine with original data
    data = pd.concat([data, genre_df], axis=1)
    
    # Normalize ratings between 0 and 1
    data['rating_normalized'] = (data['rating'] - data['rating'].min()) / (data['rating'].max() - data['rating'].min())
    
    # Create user and movie statistics
    user_stats = data.groupby('userId').agg({
        'rating': ['mean', 'count'],
        'liked': 'mean'
    }).round(3)
    user_stats.columns = ['user_avg_rating', 'user_rating_count', 'user_like_ratio']
    
    movie_stats = data.groupby('movieId').agg({
        'rating': ['mean', 'count'],
        'liked': 'mean'
    }).round(3)
    movie_stats.columns = ['movie_avg_rating', 'movie_rating_count', 'movie_like_ratio']
    
    # Merge statistics back to main data
    data = data.merge(user_stats, on='userId', how='left')
    data = data.merge(movie_stats, on='movieId', how='left')
    
    print(f"Data preprocessing complete. Shape: {data.shape}")
    print(f"Positive samples (liked): {data['liked'].sum()}")
    print(f"Negative samples (disliked): {len(data) - data['liked'].sum()}")
    
    return data, vectorizer

def train_recommendation_model(data, vectorizer):
    """Train the recommendation model using Random Forest"""
    print("Training recommendation model...")
    
    # Select features for training
    feature_columns = [col for col in data.columns if col.startswith('genre_')]
    feature_columns.extend(['rating_normalized', 'user_avg_rating', 'user_rating_count', 
                           'user_like_ratio', 'movie_avg_rating', 'movie_rating_count', 'movie_like_ratio'])
    
    X = data[feature_columns]
    y = data['liked']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and preprocessors
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    
    # Save feature columns for later use
    joblib.dump(feature_columns, 'models/feature_columns.joblib')
    
    print("Model and preprocessors saved successfully!")
    
    return model, scaler, feature_columns

def create_movie_database(data):
    """Create a clean movie database for recommendations"""
    print("Creating movie database...")
    
    # Create unique movies dataset
    movies_db = data.groupby(['movieId', 'title', 'genres']).agg({
        'rating': 'mean',
        'liked': 'mean',
        'movie_rating_count': 'first'
    }).round(3).reset_index()
    
    movies_db.columns = ['movieId', 'title', 'genres', 'avg_rating', 'like_probability', 'rating_count']
    
    # Filter movies with at least 10 ratings for better recommendations
    movies_db = movies_db[movies_db['rating_count'] >= 10]
    
    # Save the movie database
    movies_db.to_csv('models/movies_database.csv', index=False)
    
    print(f"Movie database created with {len(movies_db)} movies")
    
    return movies_db

if __name__ == "__main__":
    # Download data if not exists
    if not os.path.exists('data/ml-latest-small'):
        download_movielens_data()
    
    # Load and preprocess data
    data, vectorizer = load_and_preprocess_data()
    
    # Train model
    model, scaler, feature_columns = train_recommendation_model(data, vectorizer)
    
    # Create movie database
    movies_db = create_movie_database(data)
    
    print("\nData processing and model training completed successfully!")
    print("Files created:")
    print("- models/model.joblib")
    print("- models/scaler.joblib") 
    print("- models/vectorizer.joblib")
    print("- models/feature_columns.joblib")
    print("- models/movies_database.csv")
