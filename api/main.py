from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os
from typing import Optional, List, Dict, Any
import traceback

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.recommendation_engine import MovieRecommendationEngine
except ImportError as e:
    print(f"Import error: {e}")
    MovieRecommendationEngine = None

app = FastAPI(
    title="Movie Recommendation API",
    description="A machine learning-powered movie recommendation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store the recommendation engine
recommendation_engine = None

def get_recommendation_engine():
    """Get or initialize the recommendation engine"""
    global recommendation_engine
    
    if recommendation_engine is None:
        try:
            # Try to initialize the recommendation engine
            recommendation_engine = MovieRecommendationEngine()
            print("Recommendation engine initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize recommendation engine: {e}")
            print("Traceback:", traceback.format_exc())
            recommendation_engine = None
    
    return recommendation_engine

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "/recommend": "Get movie recommendations",
            "/health": "Health check",
            "/user/{user_id}/profile": "Get user profile"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    engine = get_recommendation_engine()
    
    return {
        "status": "healthy" if engine is not None else "unhealthy",
        "model_loaded": engine is not None,
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/recommend")
async def get_recommendations(
    user_id: Optional[int] = Query(None, description="User ID for personalized recommendations"),
    num_recommendations: int = Query(10, ge=1, le=50, description="Number of recommendations to return"),
    min_rating: float = Query(3.5, ge=0.5, le=5.0, description="Minimum movie rating filter")
):
    """
    Get movie recommendations
    
    - **user_id**: Optional user ID for personalized recommendations
    - **num_recommendations**: Number of movies to recommend (1-50)
    - **min_rating**: Minimum rating filter (0.5-5.0)
    """
    try:
        engine = get_recommendation_engine()
        
        if engine is None:
            raise HTTPException(
                status_code=503, 
                detail="Recommendation engine is not available. Please ensure the model files are present."
            )
        
        # Get recommendations
        result = engine.get_recommendations(
            user_id=user_id,
            num_recommendations=num_recommendations,
            min_rating=min_rating
        )
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        print("Traceback:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/user/{user_id}/profile")
async def get_user_profile(user_id: int):
    """
    Get user profile information
    
    - **user_id**: User ID to get profile for
    """
    try:
        engine = get_recommendation_engine()
        
        if engine is None:
            raise HTTPException(
                status_code=503, 
                detail="Recommendation engine is not available"
            )
        
        profile = engine.get_user_profile(user_id)
        
        if profile is None:
            raise HTTPException(
                status_code=404, 
                detail=f"User {user_id} not found"
            )
        
        return JSONResponse(content={
            "user_id": user_id,
            "profile": profile
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_user_profile: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/movies/popular")
async def get_popular_movies(
    limit: int = Query(20, ge=1, le=100, description="Number of popular movies to return")
):
    """
    Get popular movies based on rating and rating count
    
    - **limit**: Number of movies to return (1-100)
    """
    try:
        engine = get_recommendation_engine()
        
        if engine is None:
            raise HTTPException(
                status_code=503, 
                detail="Recommendation engine is not available"
            )
        
        # Get popular movies (high rating + high rating count)
        movies_db = engine.movies_db.copy()
        
        # Calculate popularity score (weighted average of rating and normalized rating count)
        max_count = movies_db['rating_count'].max()
        movies_db['popularity_score'] = (
            movies_db['avg_rating'] * 0.7 + 
            (movies_db['rating_count'] / max_count) * 5 * 0.3
        )
        
        # Sort by popularity and get top movies
        popular_movies = movies_db.nlargest(limit, 'popularity_score')
        
        result = []
        for _, movie in popular_movies.iterrows():
            primary_genre = movie['genres'].split('|')[0]
            result.append({
                'title': movie['title'],
                'genre': primary_genre,
                'rating': round(movie['avg_rating'], 1),
                'rating_count': movie['rating_count'],
                'poster_url': f"/placeholder.svg?height=300&width=200",
                'popularity_score': round(movie['popularity_score'], 2)
            })
        
        return JSONResponse(content={
            "popular_movies": result,
            "total_count": len(result)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_popular_movies: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/genres")
async def get_available_genres():
    """Get list of available movie genres"""
    try:
        engine = get_recommendation_engine()
        
        if engine is None:
            raise HTTPException(
                status_code=503, 
                detail="Recommendation engine is not available"
            )
        
        # Extract all unique genres
        all_genres = set()
        for genres_str in engine.movies_db['genres']:
            genres = genres_str.split('|')
            all_genres.update(genres)
        
        # Remove empty strings and sort
        genres_list = sorted([genre for genre in all_genres if genre.strip()])
        
        return JSONResponse(content={
            "genres": genres_list,
            "total_count": len(genres_list)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_available_genres: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Initialize the recommendation engine on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation engine on startup"""
    print("Starting up Movie Recommendation API...")
    get_recommendation_engine()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
