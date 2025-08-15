import { NextResponse } from "next/server"

// Mock movie data for demonstration
const mockMovies = [
  {
    id: 1,
    title: "The Shawshank Redemption",
    genre: "Drama",
    rating: 9.3,
    poster_url: "/shawshank-redemption-poster.png",
  },
  {
    id: 2,
    title: "The Godfather",
    genre: "Crime, Drama",
    rating: 9.2,
    poster_url: "/classic-mob-poster.png",
  },
  {
    id: 3,
    title: "The Dark Knight",
    genre: "Action, Crime, Drama",
    rating: 9.0,
    poster_url: "/dark-knight-poster.png",
  },
  {
    id: 4,
    title: "Pulp Fiction",
    genre: "Crime, Drama",
    rating: 8.9,
    poster_url: "/pulp-fiction-poster.png",
  },
  {
    id: 5,
    title: "Forrest Gump",
    genre: "Drama, Romance",
    rating: 8.8,
    poster_url: "/forrest-gump-poster.png",
  },
  {
    id: 6,
    title: "Inception",
    genre: "Action, Sci-Fi, Thriller",
    rating: 8.8,
    poster_url: "/inception-movie-poster.png",
  },
  {
    id: 7,
    title: "The Matrix",
    genre: "Action, Sci-Fi",
    rating: 8.7,
    poster_url: "/matrix-movie-poster.png",
  },
  {
    id: 8,
    title: "Goodfellas",
    genre: "Biography, Crime, Drama",
    rating: 8.7,
    poster_url: "/goodfellas-poster.png",
  },
]

export async function GET() {
  try {
    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 500))

    return NextResponse.json({
      movies: mockMovies,
      total: mockMovies.length,
    })
  } catch (error) {
    console.error("Error fetching popular movies:", error)
    return NextResponse.json({ error: "Failed to fetch popular movies" }, { status: 500 })
  }
}
