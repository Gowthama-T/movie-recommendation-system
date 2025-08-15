import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { User, Star, Heart, BarChart3 } from "lucide-react"

interface UserProfileProps {
  profile: {
    user_avg_rating: number
    user_rating_count: number
    user_like_ratio: number
    favorite_genres?: Record<string, number>
  }
  userId: string
}

export function UserProfile({ profile, userId }: UserProfileProps) {
  return (
    <Card className="mb-8 movie-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <User className="h-5 w-5 text-primary" />
          User Profile - ID: {userId}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="flex items-center gap-3 p-4 bg-muted/20 rounded-lg">
            <Star className="h-8 w-8 text-yellow-400" />
            <div>
              <p className="text-sm text-muted-foreground">Average Rating</p>
              <p className="text-2xl font-bold">{profile.user_avg_rating.toFixed(1)}</p>
            </div>
          </div>

          <div className="flex items-center gap-3 p-4 bg-muted/20 rounded-lg">
            <BarChart3 className="h-8 w-8 text-accent" />
            <div>
              <p className="text-sm text-muted-foreground">Movies Rated</p>
              <p className="text-2xl font-bold">{profile.user_rating_count}</p>
            </div>
          </div>

          <div className="flex items-center gap-3 p-4 bg-muted/20 rounded-lg">
            <Heart className="h-8 w-8 text-primary" />
            <div>
              <p className="text-sm text-muted-foreground">Like Ratio</p>
              <p className="text-2xl font-bold">{Math.round(profile.user_like_ratio * 100)}%</p>
            </div>
          </div>
        </div>

        {profile.favorite_genres && (
          <div className="mt-6">
            <h4 className="text-sm font-medium text-muted-foreground mb-3">Favorite Genres</h4>
            <div className="flex flex-wrap gap-2">
              {Object.entries(profile.favorite_genres).map(([genre, count]) => (
                <Badge key={genre} variant="secondary" className="bg-primary/10 text-primary">
                  {genre} ({count})
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
