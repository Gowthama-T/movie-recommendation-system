import { NextResponse } from "next/server"

// Mock user profiles
const userProfiles = {
  "1": {
    id: 1,
    name: "Alex Johnson",
    favorite_genres: ["Action", "Sci-Fi", "Thriller"],
    total_ratings: 127,
    average_rating: 4.2,
    joined_date: "2022-03-15",
  },
  "2": {
    id: 2,
    name: "Sarah Chen",
    favorite_genres: ["Drama", "Romance", "Comedy"],
    total_ratings: 89,
    average_rating: 4.5,
    joined_date: "2021-11-08",
  },
  "3": {
    id: 3,
    name: "Mike Rodriguez",
    favorite_genres: ["Horror", "Thriller", "Mystery"],
    total_ratings: 203,
    average_rating: 3.8,
    joined_date: "2020-07-22",
  },
}

export async function GET(request: Request, { params }: { params: { userId: string } }) {
  try {
    const userId = params.userId

    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 300))

    const profile = userProfiles[userId as keyof typeof userProfiles]

    if (!profile) {
      // Generate a default profile for unknown users
      const defaultProfile = {
        id: Number.parseInt(userId),
        name: `User ${userId}`,
        favorite_genres: ["Drama", "Action"],
        total_ratings: Math.floor(Math.random() * 100) + 20,
        average_rating: Math.round((Math.random() * 2 + 3) * 10) / 10,
        joined_date: "2023-01-01",
      }

      return NextResponse.json(defaultProfile)
    }

    return NextResponse.json(profile)
  } catch (error) {
    console.error("Error fetching user profile:", error)
    return NextResponse.json({ error: "Failed to fetch user profile" }, { status: 500 })
  }
}
