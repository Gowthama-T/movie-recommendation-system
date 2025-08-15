import requests
import json

def test_api_endpoints():
    """Test the FastAPI endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing Movie Recommendation API...")
    print("=" * 50)
    
    # Test root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    
    # Test health check
    print("2. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    
    # Test recommendations without user ID
    print("3. Testing general recommendations...")
    try:
        response = requests.get(f"{base_url}/recommend?num_recommendations=5")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"User ID: {data.get('user_id')}")
            print(f"Number of recommendations: {len(data.get('recommendations', []))}")
            for i, rec in enumerate(data.get('recommendations', [])[:3]):
                print(f"  {i+1}. {rec['title']} ({rec['genre']}) - Rating: {rec['rating']}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    
    # Test recommendations with user ID
    print("4. Testing personalized recommendations...")
    try:
        response = requests.get(f"{base_url}/recommend?user_id=1&num_recommendations=5")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"User ID: {data.get('user_id')}")
            print(f"Number of recommendations: {len(data.get('recommendations', []))}")
            for i, rec in enumerate(data.get('recommendations', [])[:3]):
                print(f"  {i+1}. {rec['title']} ({rec['genre']}) - Rating: {rec['rating']}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    
    # Test user profile
    print("5. Testing user profile...")
    try:
        response = requests.get(f"{base_url}/user/1/profile")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            profile = data.get('profile', {})
            print(f"User {data.get('user_id')} profile:")
            print(f"  Average rating: {profile.get('user_avg_rating', 'N/A')}")
            print(f"  Rating count: {profile.get('user_rating_count', 'N/A')}")
            print(f"  Like ratio: {profile.get('user_like_ratio', 'N/A')}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    
    # Test popular movies
    print("6. Testing popular movies...")
    try:
        response = requests.get(f"{base_url}/movies/popular?limit=5")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Popular movies ({data.get('total_count')}):")
            for i, movie in enumerate(data.get('popular_movies', [])[:3]):
                print(f"  {i+1}. {movie['title']} ({movie['genre']}) - Rating: {movie['rating']}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    
    # Test genres
    print("7. Testing available genres...")
    try:
        response = requests.get(f"{base_url}/genres")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            genres = data.get('genres', [])
            print(f"Available genres ({data.get('total_count')}):")
            print(f"  {', '.join(genres[:10])}{'...' if len(genres) > 10 else ''}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api_endpoints()
