"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { MovieCard } from "@/components/movie-card"
import { UserProfile } from "@/components/user-profile"
import { LoadingSpinner } from "@/components/loading-spinner"
import { Film, Star, TrendingUp, User, Search, Filter, Heart } from "lucide-react"

interface Movie {
  title: string
  genre: string
  rating: number
  like_probability?: number
  poster_url: string
  rating_count?: number
  popularity_score?: number
}

interface RecommendationResponse {
  user_id?: number
  user_profile?: any
  recommendations: Movie[]
}

const GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation", "Documentary"]

export default function MovieRecommendationApp() {
  const [userId, setUserId] = useState<string>("")
  const [searchQuery, setSearchQuery] = useState<string>("")
  const [selectedGenres, setSelectedGenres] = useState<string[]>([])
  const [favorites, setFavorites] = useState<Set<string>>(new Set())
  const [recommendations, setRecommendations] = useState<Movie[]>([])
  const [popularMovies, setPopularMovies] = useState<Movie[]>([])
  const [filteredMovies, setFilteredMovies] = useState<Movie[]>([])
  const [userProfile, setUserProfile] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>("")
  const [numRecommendations, setNumRecommendations] = useState([10])
  const [minRating, setMinRating] = useState([7.0])
  const [activeTab, setActiveTab] = useState("popular")
  const [numRecAnimating, setNumRecAnimating] = useState(false)
  const [minRatingAnimating, setMinRatingAnimating] = useState(false)

  useEffect(() => {
    fetchPopularMovies()
  }, [])

  useEffect(() => {
    let filtered = activeTab === "popular" ? popularMovies : recommendations

    if (searchQuery) {
      filtered = filtered.filter(
        (movie) =>
          movie.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
          movie.genre.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    }

    if (selectedGenres.length > 0) {
      filtered = filtered.filter((movie) =>
        selectedGenres.some((genre) => movie.genre.toLowerCase().includes(genre.toLowerCase())),
      )
    }

    setFilteredMovies(filtered)
  }, [searchQuery, selectedGenres, popularMovies, recommendations, activeTab])

  const fetchPopularMovies = async () => {
    try {
      const response = await fetch("/api/movies/popular?limit=12")
      if (response.ok) {
        const data = await response.json()
        setPopularMovies(data.popular_movies || [])
      }
    } catch (error) {
      console.error("Error fetching popular movies:", error)
    }
  }

  const fetchRecommendations = async () => {
    setLoading(true)
    setError("")

    try {
      const userIdParam = userId ? `user_id=${userId}&` : ""
      const searchParam = searchQuery ? `search=${encodeURIComponent(searchQuery)}&` : ""
      const genresParam = selectedGenres.length > 0 ? `genres=${encodeURIComponent(selectedGenres.join(","))}&` : ""
      const url = `/api/recommend?${userIdParam}${searchParam}${genresParam}num_recommendations=${numRecommendations[0]}&min_rating=${minRating[0]}`

      console.log("[v0] Fetching recommendations with URL:", url)

      const response = await fetch(url)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: RecommendationResponse = await response.json()

      console.log("[v0] Received recommendations:", data.recommendations?.length || 0)

      setRecommendations(data.recommendations || [])
      setUserProfile(data.user_profile)
      if (data.recommendations && data.recommendations.length > 0) {
        setActiveTab("recommendations")
      }
    } catch (error) {
      console.error("Error fetching recommendations:", error)
      setError("Failed to fetch recommendations. Please try again.")
    } finally {
      setLoading(false)
    }
  }

  const fetchUserProfile = async () => {
    if (!userId) return

    try {
      const response = await fetch(`/api/user/${userId}/profile`)
      if (response.ok) {
        const data = await response.json()
        setUserProfile(data.profile)
      }
    } catch (error) {
      console.error("Error fetching user profile:", error)
    }
  }

  const toggleGenre = (genre: string) => {
    setSelectedGenres((prev) => (prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre]))
  }

  const toggleFavorite = (movieTitle: string) => {
    setFavorites((prev) => {
      const newFavorites = new Set(prev)
      if (newFavorites.has(movieTitle)) {
        newFavorites.delete(movieTitle)
      } else {
        newFavorites.add(movieTitle)
      }
      return newFavorites
    })
  }

  const handleNumRecommendationsChange = (value: number[]) => {
    setNumRecommendations(value)
    setNumRecAnimating(true)
    setTimeout(() => setNumRecAnimating(false), 300)
  }

  const handleMinRatingChange = (value: number[]) => {
    setMinRating(value)
    setMinRatingAnimating(true)
    setTimeout(() => setMinRatingAnimating(false), 300)
  }

  const currentMovies =
    filteredMovies.length > 0 ? filteredMovies : activeTab === "popular" ? popularMovies : recommendations

  return (
    <div className="min-h-screen" style={{ backgroundColor: "#f9fafb" }}>
      {/* Header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-md sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 float-animation">
              <Film className="h-8 w-8 text-primary" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                MovieAI
              </h1>
            </div>
            <div className="flex items-center gap-4">
              <Badge variant="secondary" className="bg-primary/10 text-primary">
                Powered by ML
              </Badge>
              {favorites.size > 0 && (
                <Badge variant="outline" className="flex items-center gap-1">
                  <Heart className="h-3 w-3 text-red-500" />
                  {favorites.size} Favorites
                </Badge>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <section className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent float-animation">
            Discover Your Next Favorite Movie
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Get personalized movie recommendations powered by machine learning. Search, filter, and save your favorites.
          </p>
        </section>

        {/* Search and Filter Section */}
        <Card className="mb-8 movie-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5 text-primary" />
              Search & Filter
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search movies by title or genre..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 search-input transition-all duration-300"
              />
            </div>

            <div className="space-y-2">
              <Label className="flex items-center gap-2">
                <Filter className="h-4 w-4" />
                Filter by Genre
              </Label>
              <div className="flex flex-wrap gap-2">
                {GENRES.map((genre) => (
                  <Button
                    key={genre}
                    variant={selectedGenres.includes(genre) ? "default" : "outline"}
                    size="sm"
                    onClick={() => toggleGenre(genre)}
                    className={`genre-filter ${selectedGenres.includes(genre) ? "active" : ""}`}
                  >
                    {genre}
                  </Button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Recommendation Controls */}
        <Card className="mb-8 movie-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Star className="h-5 w-5 text-primary" />
              Get Recommendations
            </CardTitle>
            <CardDescription>Customize your movie discovery experience</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <Label htmlFor="userId" className="flex items-center gap-2">
                <User className="h-4 w-4" />
                User ID (Optional)
              </Label>
              <div className="flex gap-2">
                <Input
                  id="userId"
                  type="number"
                  placeholder="Enter your user ID for personalized recommendations"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  className="bg-input border-border transition-all duration-300"
                />
                {userId && (
                  <Button variant="outline" onClick={fetchUserProfile} className="bg-transparent whitespace-nowrap">
                    Load Profile
                  </Button>
                )}
              </div>
            </div>

            <div className="p-6 bg-gradient-to-br from-primary/5 via-accent/5 to-secondary/5 rounded-xl border-2 border-primary/20 slider-controls-container">
              <div className="text-center mb-6">
                <h3 className="text-lg font-semibold text-primary mb-2">Recommendation Settings</h3>
                <p className="text-sm text-muted-foreground">Adjust your preferences below</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-4 p-4 rounded-lg" style={{ backgroundColor: "#f0f4ff" }}>
                  <div className="text-center">
                    <Label className="text-sm font-medium" style={{ color: "#1e3a8a" }}>
                      Number of Movies
                    </Label>
                    <div
                      className={`mt-2 inline-flex items-center justify-center w-16 h-16 rounded-full font-bold text-lg shadow-lg transition-all duration-300 ${numRecAnimating ? "animate-bounce scale-110" : ""}`}
                      style={{ backgroundColor: "#1e3a8a", color: "white" }}
                    >
                      {numRecommendations[0]}
                    </div>
                  </div>
                  <Slider
                    value={numRecommendations}
                    onValueChange={handleNumRecommendationsChange}
                    max={20}
                    min={5}
                    step={1}
                    className="w-full slider-enhanced-prominent slider-blue"
                  />
                  <div className="flex justify-between text-xs" style={{ color: "#1e3a8a" }}>
                    <span className="px-2 py-1 rounded" style={{ backgroundColor: "#dbeafe" }}>
                      5 movies
                    </span>
                    <span className="px-2 py-1 rounded" style={{ backgroundColor: "#dbeafe" }}>
                      20 movies
                    </span>
                  </div>
                </div>

                <div className="space-y-4 p-4 rounded-lg" style={{ backgroundColor: "#fff4f0" }}>
                  <div className="text-center">
                    <Label className="text-sm font-medium" style={{ color: "#7c2d12" }}>
                      Minimum Rating
                    </Label>
                    <div
                      className={`mt-2 inline-flex items-center justify-center w-16 h-16 rounded-full font-bold text-sm shadow-lg transition-all duration-300 ${minRatingAnimating ? "animate-bounce scale-110" : ""}`}
                      style={{ backgroundColor: "#7c2d12", color: "white" }}
                    >
                      {minRating[0]}★
                    </div>
                  </div>
                  <Slider
                    value={minRating}
                    onValueChange={handleMinRatingChange}
                    max={10}
                    min={1}
                    step={0.1}
                    className="w-full slider-enhanced-prominent slider-orange"
                  />
                  <div className="flex justify-between text-xs" style={{ color: "#7c2d12" }}>
                    <span className="px-2 py-1 rounded" style={{ backgroundColor: "#ffedd5" }}>
                      1.0★
                    </span>
                    <span className="px-2 py-1 rounded" style={{ backgroundColor: "#ffedd5" }}>
                      10.0★
                    </span>
                  </div>
                </div>
              </div>

              {/* Live Preview of Current Settings */}
              <div className="mt-6 p-4 bg-card/50 rounded-lg border border-border/50">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground mb-2">Current Settings:</p>
                  <div className="flex items-center justify-center gap-4 text-sm">
                    <Badge variant="secondary" className="bg-primary/10 text-primary px-3 py-1">
                      {numRecommendations[0]} Movies
                    </Badge>
                    <Badge variant="secondary" className="bg-accent/10 text-accent px-3 py-1">
                      {minRating[0]}★ Minimum
                    </Badge>
                  </div>
                </div>
              </div>
            </div>

            <Button
              onClick={fetchRecommendations}
              disabled={loading}
              className={`w-full text-white transition-all duration-300 ${!loading ? "pulse-button" : ""}`}
              style={{
                backgroundColor: "#2563eb",
                borderColor: "#2563eb",
              }}
              onMouseEnter={(e) => {
                if (!loading) {
                  e.currentTarget.style.backgroundColor = "#1e40af"
                }
              }}
              onMouseLeave={(e) => {
                if (!loading) {
                  e.currentTarget.style.backgroundColor = "#2563eb"
                }
              }}
            >
              {loading ? <LoadingSpinner /> : "Get Recommendations"}
            </Button>

            {error && <div className="text-destructive text-sm mt-2 p-3 bg-destructive/10 rounded-md">{error}</div>}
          </CardContent>
        </Card>

        {/* User Profile */}
        {userProfile && <UserProfile profile={userProfile} userId={userId} />}

        {/* Results Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="popular" className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Popular Movies ({popularMovies.length})
            </TabsTrigger>
            <TabsTrigger value="recommendations" className="flex items-center gap-2">
              <Star className="h-4 w-4" />
              Recommendations {recommendations.length > 0 && `(${recommendations.length})`}
              {recommendations.length > 0 && (
                <Badge variant="secondary" className="ml-1 bg-primary/20 text-primary">
                  New!
                </Badge>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="popular" className="tab-content">
            {(searchQuery || selectedGenres.length > 0) && (
              <div className="mb-4 text-sm text-muted-foreground">
                Showing {currentMovies.length} of {popularMovies.length} movies
                {searchQuery && ` matching "${searchQuery}"`}
                {selectedGenres.length > 0 && ` in ${selectedGenres.join(", ")}`}
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {currentMovies.map((movie, index) => (
                <div key={`popular-${index}`} className="movie-grid-item">
                  <MovieCard
                    movie={movie}
                    showPopularity={true}
                    isFavorite={favorites.has(movie.title)}
                    onToggleFavorite={() => toggleFavorite(movie.title)}
                  />
                </div>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="recommendations" className="tab-content">
            {recommendations.length > 0 ? (
              <>
                {(searchQuery || selectedGenres.length > 0) && (
                  <div className="mb-4 text-sm text-muted-foreground">
                    Showing {currentMovies.length} of {recommendations.length} recommendations
                    {searchQuery && ` matching "${searchQuery}"`}
                    {selectedGenres.length > 0 && ` in ${selectedGenres.join(", ")}`}
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {currentMovies.map((movie, index) => (
                    <div key={`rec-${index}`} className="movie-grid-item">
                      <MovieCard
                        movie={movie}
                        showLikeProbability={true}
                        isFavorite={favorites.has(movie.title)}
                        onToggleFavorite={() => toggleFavorite(movie.title)}
                      />
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <Card className="text-center py-12">
                <CardContent>
                  <Film className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-xl font-semibold mb-2">No Recommendations Yet</h3>
                  <p className="text-muted-foreground">
                    Click "Get Recommendations" above to discover movies you'll love!
                  </p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
