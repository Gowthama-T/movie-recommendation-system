import kagglehub
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
import pickle
import json
import os

def download_and_process_kaggle_data():
    """Download and process the Kaggle movie recommendation dataset"""
    print("Downloading Kaggle dataset...")
    
    # Download the dataset
    path = kagglehub.dataset_download("parasharmanas/movie-recommendation-system")
    print(f"Dataset downloaded to: {path}")
    
    # Find CSV files in the downloaded path
    csv_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"Found CSV files: {csv_files}")
    
    # Load the main dataset (assuming it's the largest CSV file)
    if csv_files:
        main_file = max(csv_files, key=os.path.getsize)
        print(f"Loading main dataset: {main_file}")
        df = pd.read_csv(main_file)
    else:
        print("No CSV files found, creating sample dataset...")
        # Create a sample dataset if no files found
        df = create_sample_dataset()
    
    return df, path

def create_sample_dataset():
    """Create a sample movie dataset if Kaggle data is not available"""
    movies_data = {
        'title': [
            'The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction',
            'Forrest Gump', 'Inception', 'The Matrix', 'Goodfellas', 'The Silence of the Lambs',
            'Saving Private Ryan', 'Interstellar', 'The Lord of the Rings', 'Fight Club',
            'Parasite', 'Spirited Away', 'Avengers: Endgame', 'Titanic', 'The Lion King',
            'Gladiator', 'The Departed', 'Casablanca', 'Citizen Kane', 'Vertigo', 'Psycho',
            'Alien', 'Apocalypse Now', 'Taxi Driver', 'Raging Bull', 'The Wizard of Oz',
            'Singin\' in the Rain', 'Some Like It Hot', 'Lawrence of Arabia', 'Sunset Boulevard',
            'On the Waterfront', 'The Treasure of the Sierra Madre', 'The Best Years of Our Lives',
            'Double Indemnity', 'The Maltese Falcon', 'To Kill a Mockingbird', 'It\'s a Wonderful Life'
        ],
        'genre': [
            'Drama', 'Crime, Drama', 'Action, Crime, Drama', 'Crime, Drama',
            'Drama, Romance', 'Action, Sci-Fi, Thriller', 'Action, Sci-Fi', 'Biography, Crime, Drama',
            'Crime, Drama, Thriller', 'Drama, War', 'Adventure, Drama, Sci-Fi', 'Adventure, Drama, Fantasy',
            'Drama', 'Comedy, Drama, Thriller', 'Animation, Adventure, Family', 'Action, Adventure, Drama',
            'Drama, Romance', 'Animation, Adventure, Drama', 'Action, Adventure, Drama', 'Crime, Drama, Thriller',
            'Drama, Romance, War', 'Drama', 'Mystery, Romance, Thriller', 'Horror, Mystery, Thriller',
            'Horror, Sci-Fi', 'Drama, War', 'Crime, Drama', 'Biography, Drama, Sport',
            'Adventure, Family, Fantasy', 'Comedy, Musical, Romance', 'Comedy, Romance', 'Adventure, Biography, Drama',
            'Drama, Film-Noir', 'Crime, Drama', 'Adventure, Drama, Western', 'Drama, Romance, War',
            'Crime, Drama, Film-Noir', 'Crime, Drama, Film-Noir', 'Crime, Drama', 'Drama, Family'
        ],
        'rating': [
            9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.6, 8.6, 8.9, 8.8,
            8.6, 9.2, 8.4, 7.8, 8.5, 8.5, 8.5, 8.5, 8.3, 8.3, 8.5, 8.4, 8.4,
            8.3, 8.2, 8.1, 8.3, 8.2, 8.3, 8.4, 8.1, 8.0, 8.1, 8.4, 7.9, 8.3, 8.6
        ],
        'year': [
            1994, 1972, 2008, 1994, 1994, 2010, 1999, 1990, 1991, 1998, 2014, 2003, 1999,
            2019, 2001, 2019, 1997, 1994, 2000, 2006, 1942, 1941, 1958, 1960, 1979, 1979,
            1976, 1980, 1939, 1952, 1959, 1962, 1950, 1954, 1948, 1946, 1944, 1941, 1962, 1946
        ],
        'director': [
            'Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan', 'Quentin Tarantino',
            'Robert Zemeckis', 'Christopher Nolan', 'Lana Wachowski', 'Martin Scorsese', 'Jonathan Demme',
            'Steven Spielberg', 'Christopher Nolan', 'Peter Jackson', 'David Fincher',
            'Bong Joon Ho', 'Hayao Miyazaki', 'Anthony Russo', 'James Cameron', 'Roger Allers',
            'Ridley Scott', 'Martin Scorsese', 'Michael Curtiz', 'Orson Welles', 'Alfred Hitchcock',
            'Alfred Hitchcock', 'Ridley Scott', 'Francis Ford Coppola', 'Martin Scorsese', 'Martin Scorsese',
            'Victor Fleming', 'Gene Kelly', 'Billy Wilder', 'David Lean', 'Billy Wilder',
            'Elia Kazan', 'John Huston', 'William Wyler', 'Billy Wilder', 'John Huston', 'Robert Mulligan', 'Frank Capra'
        ]
    }
    
    return pd.DataFrame(movies_data)

def preprocess_data(df):
    """Preprocess the movie data for ML models"""
    print("Preprocessing data...")
    
    # Handle missing values
    df = df.fillna('')
    
    # Ensure required columns exist
    required_columns = ['title', 'genre', 'rating']
    for col in required_columns:
        if col not in df.columns:
            if col == 'rating':
                df[col] = np.random.uniform(6.0, 9.0, len(df))
            else:
                df[col] = 'Unknown'
    
    # Create features for clustering and classification
    # Genre encoding using TF-IDF
    tfidf_genre = TfidfVectorizer(max_features=50, stop_words='english')
    genre_features = tfidf_genre.fit_transform(df['genre'].astype(str))
    
    # Additional features
    df['rating_category'] = pd.cut(df['rating'], bins=[0, 6, 7, 8, 10], labels=['Low', 'Medium', 'High', 'Excellent'])
    df['title_length'] = df['title'].str.len()
    
    # Create feature matrix
    feature_columns = ['rating', 'title_length']
    if 'year' in df.columns:
        feature_columns.append('year')
    
    numerical_features = df[feature_columns].fillna(df[feature_columns].mean())
    
    # Combine features
    from scipy.sparse import hstack
    all_features = hstack([numerical_features.values, genre_features])
    
    return df, all_features, tfidf_genre

def train_clustering_model(features, n_clusters=8):
    """Train K-means clustering model"""
    print(f"Training clustering model with {n_clusters} clusters...")
    
    # Standardize features
    scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse matrices
    features_scaled = scaler.fit_transform(features)
    
    # Train K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    
    return kmeans, scaler, cluster_labels

def train_classification_model(df, features):
    """Train classification model to predict rating category"""
    print("Training classification model...")
    
    # Prepare target variable
    y = df['rating_category']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return rf, le

def save_models_and_data(df, kmeans, scaler, rf, le, tfidf_genre):
    """Save trained models and processed data"""
    print("Saving models and data...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save models
    with open('models/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/rf_classifier.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    with open('models/tfidf_genre.pkl', 'wb') as f:
        pickle.dump(tfidf_genre, f)
    
    # Save processed movie data
    movie_data = df.to_dict('records')
    with open('models/movie_database.json', 'w') as f:
        json.dump(movie_data, f, indent=2, default=str)
    
    print("Models and data saved successfully!")

def main():
    """Main function to process Kaggle data and train models"""
    try:
        # Download and load data
        df, dataset_path = download_and_process_kaggle_data()
        print(f"Loaded dataset with {len(df)} movies")
        print(f"Columns: {df.columns.tolist()}")
        
        # Preprocess data
        df_processed, features, tfidf_genre = preprocess_data(df)
        
        # Train clustering model
        kmeans, scaler, cluster_labels = train_clustering_model(features)
        df_processed['cluster'] = cluster_labels
        
        # Train classification model
        rf, le = train_classification_model(df_processed, features)
        
        # Save everything
        save_models_and_data(df_processed, kmeans, scaler, rf, le, tfidf_genre)
        
        print("✅ Kaggle movie recommendation system setup complete!")
        print(f"📊 Processed {len(df_processed)} movies")
        print(f"🎯 Created {len(set(cluster_labels))} movie clusters")
        print(f"🏷️ Classification categories: {le.classes_}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Creating fallback sample dataset...")
        
        # Fallback to sample data
        df = create_sample_dataset()
        df_processed, features, tfidf_genre = preprocess_data(df)
        kmeans, scaler, cluster_labels = train_clustering_model(features)
        df_processed['cluster'] = cluster_labels
        rf, le = train_classification_model(df_processed, features)
        save_models_and_data(df_processed, kmeans, scaler, rf, le, tfidf_genre)
        
        print("✅ Fallback system setup complete!")

if __name__ == "__main__":
    main()
