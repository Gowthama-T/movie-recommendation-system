import { NextResponse } from "next/server"

// Extended movie database with more variety
const movieDatabase = [
  {
    title: "The Shawshank Redemption",
    genre: "Drama",
    rating: 9.3,
    poster_url: "/shawshank-redemption-poster.png",
    year: 1994,
    director: "Frank Darabont",
  },
  {
    title: "The Godfather",
    genre: "Crime, Drama",
    rating: 9.2,
    poster_url: "/classic-mob-poster.png",
    year: 1972,
    director: "Francis Ford Coppola",
  },
  {
    title: "The Dark Knight",
    genre: "Action, Crime, Drama",
    rating: 9.0,
    poster_url: "/dark-knight-poster.png",
    year: 2008,
    director: "Christopher Nolan",
  },
  {
    title: "Pulp Fiction",
    genre: "Crime, Drama",
    rating: 8.9,
    poster_url: "/pulp-fiction-poster.png",
    year: 1994,
    director: "Quentin Tarantino",
  },
  {
    title: "Forrest Gump",
    genre: "Drama, Romance",
    rating: 8.8,
    poster_url: "/forrest-gump-poster.png",
    year: 1994,
    director: "Robert Zemeckis",
  },
  {
    title: "Inception",
    genre: "Action, Sci-Fi, Thriller",
    rating: 8.8,
    poster_url: "/inception-movie-poster.png",
    year: 2010,
    director: "Christopher Nolan",
  },
  {
    title: "The Matrix",
    genre: "Action, Sci-Fi",
    rating: 8.7,
    poster_url: "/matrix-inspired-poster.png",
    year: 1999,
    director: "Lana Wachowski",
  },
  {
    title: "Goodfellas",
    genre: "Biography, Crime, Drama",
    rating: 8.7,
    poster_url: "/goodfellas-poster.png",
    year: 1990,
    director: "Martin Scorsese",
  },
  {
    title: "The Silence of the Lambs",
    genre: "Crime, Drama, Thriller",
    rating: 8.6,
    poster_url: "/silence-of-the-lambs-inspired-poster.png",
    year: 1991,
    director: "Jonathan Demme",
  },
  {
    title: "Saving Private Ryan",
    genre: "Drama, War",
    rating: 8.6,
    poster_url: "/private-ryan-inspired-poster.png",
    year: 1998,
    director: "Steven Spielberg",
  },
  {
    title: "Interstellar",
    genre: "Adventure, Drama, Sci-Fi",
    rating: 8.6,
    poster_url: "/interstellar-inspired-poster.png",
    year: 2014,
    director: "Christopher Nolan",
  },
  {
    title: "The Lord of the Rings: The Return of the King",
    genre: "Adventure, Drama, Fantasy",
    rating: 8.9,
    poster_url: "/return-of-the-king-poster.png",
    year: 2003,
    director: "Peter Jackson",
  },
  {
    title: "Fight Club",
    genre: "Drama",
    rating: 8.8,
    poster_url: "/fight-club-poster.png",
    year: 1999,
    director: "David Fincher",
  },
  {
    title: "Parasite",
    genre: "Comedy, Drama, Thriller",
    rating: 8.6,
    poster_url: "/parasite-movie-poster.png",
    year: 2019,
    director: "Bong Joon Ho",
  },
  {
    title: "Spirited Away",
    genre: "Animation, Adventure, Family",
    rating: 9.2,
    poster_url: "/spirited-away-poster.png",
    year: 2001,
    director: "Hayao Miyazaki",
  },
  {
    title: "Avengers: Endgame",
    genre: "Action, Adventure, Drama",
    rating: 8.4,
    poster_url: "/generic-superhero-team-poster.png",
    year: 2019,
    director: "Anthony Russo",
  },
  {
    title: "Titanic",
    genre: "Drama, Romance",
    rating: 7.8,
    poster_url: "/titanic-poster.png",
    year: 1997,
    director: "James Cameron",
  },
  {
    title: "The Lion King",
    genre: "Animation, Adventure, Drama",
    rating: 8.5,
    poster_url: "/lion-king-poster.png",
    year: 1994,
    director: "Roger Allers",
  },
  {
    title: "Gladiator",
    genre: "Action, Adventure, Drama",
    rating: 8.5,
    poster_url: "/gladiator-poster.png",
    year: 2000,
    director: "Ridley Scott",
  },
  {
    title: "The Departed",
    genre: "Crime, Drama, Thriller",
    rating: 8.5,
    poster_url: "/departed-poster.png",
    year: 2006,
    director: "Martin Scorsese",
  },
  {
    title: "Casablanca",
    genre: "Drama, Romance, War",
    rating: 8.5,
    poster_url: "/casablanca-poster.png",
    year: 1942,
    director: "Michael Curtiz",
  },
  {
    title: "Citizen Kane",
    genre: "Drama",
    rating: 8.3,
    poster_url: "/citizen-kane-poster.png",
    year: 1941,
    director: "Orson Welles",
  },
  {
    title: "Vertigo",
    genre: "Mystery, Romance, Thriller",
    rating: 8.3,
    poster_url: "/vertigo-poster.png",
    year: 1958,
    director: "Alfred Hitchcock",
  },
  {
    title: "Psycho",
    genre: "Horror, Mystery, Thriller",
    rating: 8.5,
    poster_url: "/psycho-poster.png",
    year: 1960,
    director: "Alfred Hitchcock",
  },
  {
    title: "Alien",
    genre: "Horror, Sci-Fi",
    rating: 8.4,
    poster_url: "/alien-poster.png",
    year: 1979,
    director: "Ridley Scott",
  },
]

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const userId = searchParams.get("user_id")
    const limit = Number.parseInt(searchParams.get("num_recommendations") || searchParams.get("limit") || "10")
    const minRating = Number.parseFloat(searchParams.get("min_rating") || "0")
    const searchQuery = searchParams.get("search") || ""
    const genresFilter = searchParams.get("genres") || ""

    console.log("[v0] API called with params:", { userId, limit, minRating, searchQuery, genresFilter })

    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 800))

    const recommendations = await getMLRecommendations(userId, limit, minRating, searchQuery, genresFilter)

    console.log("[v0] Returning recommendations:", recommendations.length)

    return NextResponse.json({
      recommendations,
      user_id: userId,
      total: recommendations.length,
      filters: {
        min_rating: minRating,
        search: searchQuery,
        genres: genresFilter,
        limit,
      },
      ml_powered: true,
    })
  } catch (error) {
    console.error("Error generating recommendations:", error)
    return NextResponse.json({ error: "Failed to generate recommendations" }, { status: 500 })
  }
}

async function getMLRecommendations(
  userId: string | null,
  limit: number,
  minRating: number,
  searchQuery = "",
  genresFilter = "",
) {
  let filteredMovies = [...movieDatabase]

  // Apply filters progressively, keeping track of results
  if (minRating > 1) {
    const ratingFiltered = filteredMovies.filter((movie) => movie.rating >= minRating)
    if (ratingFiltered.length >= limit) {
      filteredMovies = ratingFiltered
    }
    // If rating filter reduces results too much, relax it slightly
    else if (ratingFiltered.length < limit / 2) {
      filteredMovies = movieDatabase.filter((movie) => movie.rating >= Math.max(1, minRating - 0.5))
    } else {
      filteredMovies = ratingFiltered
    }
  }

  if (searchQuery) {
    const query = searchQuery.toLowerCase()
    const searchFiltered = filteredMovies.filter(
      (movie) =>
        movie.title.toLowerCase().includes(query) ||
        movie.genre.toLowerCase().includes(query) ||
        movie.director.toLowerCase().includes(query),
    )
    if (searchFiltered.length >= limit || searchFiltered.length >= filteredMovies.length / 2) {
      filteredMovies = searchFiltered
    }
  }

  if (genresFilter) {
    const selectedGenres = genresFilter.split(",").map((g) => g.trim().toLowerCase())
    const genreFiltered = filteredMovies.filter((movie) =>
      selectedGenres.some((genre) => movie.genre.toLowerCase().includes(genre)),
    )

    // If genre filter reduces results too much, include partial matches
    if (genreFiltered.length < limit && genreFiltered.length < filteredMovies.length / 3) {
      // Add movies that match at least one genre or have similar themes
      const partialMatches = movieDatabase.filter((movie) => {
        const movieGenres = movie.genre.toLowerCase()
        return selectedGenres.some(
          (genre) =>
            movieGenres.includes(genre) ||
            (genre === "action" && (movieGenres.includes("adventure") || movieGenres.includes("thriller"))) ||
            (genre === "drama" && (movieGenres.includes("romance") || movieGenres.includes("biography"))) ||
            (genre === "sci-fi" && movieGenres.includes("thriller")),
        )
      })
      filteredMovies = [...new Set([...genreFiltered, ...partialMatches])]
    } else {
      filteredMovies = genreFiltered
    }
  }

  // ML-based personalization simulation
  if (userId) {
    const userSeed = Number.parseInt(userId) || 1

    // Simulate user preferences based on user ID
    const userPreferences = {
      preferredGenres: userSeed % 2 === 0 ? ["Action", "Sci-Fi", "Thriller"] : ["Drama", "Romance", "Comedy"],
      preferredDecade: userSeed % 3 === 0 ? 1990 : userSeed % 3 === 1 ? 2000 : 2010,
      ratingBias: (userSeed % 5) * 0.1, // 0.0 to 0.4 bias
    }

    // Score movies based on user preferences
    filteredMovies = filteredMovies.map((movie) => {
      let score = movie.rating

      // Genre preference scoring
      const movieGenres = movie.genre.toLowerCase()
      const genreMatch = userPreferences.preferredGenres.some((genre) => movieGenres.includes(genre.toLowerCase()))
      if (genreMatch) score += 1.0

      // Decade preference scoring
      const movieDecade = Math.floor(movie.year / 10) * 10
      if (Math.abs(movieDecade - userPreferences.preferredDecade) <= 10) {
        score += 0.5
      }

      // Add rating bias
      score += userPreferences.ratingBias

      // Calculate like probability
      const likeProbability = Math.min(0.98, Math.max(0.6, score / 10 + (userSeed % 10) * 0.02))

      return {
        ...movie,
        score,
        like_probability: Math.round(likeProbability * 100) / 100,
      }
    })

    // Sort by score (highest first)
    filteredMovies.sort((a, b) => b.score - a.score)
  } else {
    // No user ID - add general like probability based on rating
    filteredMovies = filteredMovies.map((movie) => ({
      ...movie,
      like_probability: Math.min(0.95, Math.max(0.7, movie.rating / 10)),
    }))

    // Sort by rating (highest first)
    filteredMovies.sort((a, b) => b.rating - a.rating)
  }

  if (filteredMovies.length < limit) {
    const remaining = limit - filteredMovies.length
    const usedTitles = new Set(filteredMovies.map((m) => m.title))
    const additionalMovies = movieDatabase
      .filter((movie) => !usedTitles.has(movie.title))
      .sort((a, b) => b.rating - a.rating)
      .slice(0, remaining)
      .map((movie) => ({
        ...movie,
        like_probability: Math.min(0.95, Math.max(0.7, movie.rating / 10)),
      }))

    filteredMovies = [...filteredMovies, ...additionalMovies]
  }

  // Return limited results
  return filteredMovies.slice(0, limit)
}
