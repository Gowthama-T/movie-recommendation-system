import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def download_and_process_kaggle_data():
    """Download and process the Kaggle movie recommendation dataset"""
    print("Downloading Kaggle dataset...")
    
    # Download latest version
    path = kagglehub.dataset_download("parasharmanas/movie-recommendation-system")
    print("Path to dataset files:", path)
    
    # List all files in the dataset
    dataset_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                dataset_files.append(os.path.join(root, file))
                print(f"Found CSV file: {file}")
    
    return path, dataset_files

def load_and_preprocess_data(dataset_path):
    """Load and preprocess the movie dataset"""
    
    # Try to find the main dataset files
    csv_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Load the main dataset (assuming it's the largest CSV file)
    main_file = max(csv_files, key=os.path.getsize) if csv_files else None
    
    if not main_file:
        raise FileNotFoundError("No CSV files found in the dataset")
    
    print(f"Loading main dataset: {os.path.basename(main_file)}")
    df = pd.read_csv(main_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def create_movie_features(df):
    """Create features for movie recommendation model"""
    
    # Create a copy for processing
    processed_df = df.copy()
    
    # Handle missing values
    processed_df = processed_df.dropna()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = processed_df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col not in ['title', 'overview', 'tagline']:  # Skip text columns
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            label_encoders[col] = le
    
    return processed_df, label_encoders

def train_recommendation_model(df):
    """Train a movie recommendation model"""
    
    print("Training movie recommendation model...")
    
    # Prepare features and target
    # Assuming we have rating or popularity as target
    target_columns = ['rating', 'vote_average', 'popularity', 'score']
    target_col = None
    
    for col in target_columns:
        if col in df.columns:
            target_col = col
            break
    
    if not target_col:
        print("No suitable target column found. Creating synthetic ratings...")
        # Create synthetic ratings based on available features
        df['synthetic_rating'] = np.random.uniform(1, 5, len(df))
        target_col = 'synthetic_rating'
    
    # Select feature columns (exclude text and target columns)
    exclude_cols = ['title', 'overview', 'tagline', target_col, 'id']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, scaler, feature_cols, feature_importance

def save_model_artifacts(model, scaler, feature_cols, label_encoders, feature_importance):
    """Save trained model and preprocessing artifacts"""
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    with open('models/kaggle_movie_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open('models/kaggle_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature columns
    with open('models/kaggle_feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save label encoders
    with open('models/kaggle_label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save feature importance
    feature_importance.to_csv('models/kaggle_feature_importance.csv', index=False)
    
    print("Model artifacts saved to 'models/' directory")

def main():
    """Main function to run the complete pipeline"""
    try:
        # Download dataset
        dataset_path, csv_files = download_and_process_kaggle_data()
        
        # Load and preprocess data
        df = load_and_preprocess_data(dataset_path)
        
        # Create features
        processed_df, label_encoders = create_movie_features(df)
        
        # Train model
        model, scaler, feature_cols, feature_importance = train_recommendation_model(processed_df)
        
        # Save artifacts
        save_model_artifacts(model, scaler, feature_cols, label_encoders, feature_importance)
        
        print("\n✅ Kaggle movie recommendation model training completed successfully!")
        print("You can now use the trained model for predictions.")
        
    except Exception as e:
        print(f"❌ Error during model training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
