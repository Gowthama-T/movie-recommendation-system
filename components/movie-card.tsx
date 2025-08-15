"use client"

import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Star, Heart, TrendingUp } from "lucide-react"

interface Movie {
  title: string
  genre: string
  rating: number
  like_probability?: number
  poster_url: string
  rating_count?: number
  popularity_score?: number
}

interface MovieCardProps {
  movie: Movie
  showLikeProbability?: boolean
  showPopularity?: boolean
  isFavorite?: boolean
  onToggleFavorite?: () => void
}

export function MovieCard({
  movie,
  showLikeProbability = false,
  showPopularity = false,
  isFavorite = false,
  onToggleFavorite,
}: MovieCardProps) {
  return (
    <Card className="movie-card bg-card border-border overflow-hidden group">
      <div className="aspect-[2/3] relative overflow-hidden">
        <img
          src={movie.poster_url || "/placeholder.svg"}
          alt={movie.title}
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
        />

        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

        <div className="absolute top-2 right-2">
          <Badge variant="secondary" className="bg-background/90 backdrop-blur-sm">
            <Star className="h-3 w-3 mr-1 fill-yellow-400 text-yellow-400" />
            {movie.rating}
          </Badge>
        </div>

        {showLikeProbability && movie.like_probability && (
          <div className="absolute top-2 left-2">
            <Badge variant="default" className="bg-primary/90 backdrop-blur-sm">
              <Heart className="h-3 w-3 mr-1" />
              {Math.round(movie.like_probability * 100)}%
            </Badge>
          </div>
        )}

        {showPopularity && movie.popularity_score && (
          <div className="absolute top-2 left-2">
            <Badge variant="default" className="bg-accent/90 backdrop-blur-sm">
              <TrendingUp className="h-3 w-3 mr-1" />
              {movie.popularity_score}
            </Badge>
          </div>
        )}

        {onToggleFavorite && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleFavorite}
            className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-all duration-300 bg-background/80 backdrop-blur-sm hover:bg-background/90"
          >
            <Heart
              className={`h-4 w-4 favorite-heart ${isFavorite ? "active fill-red-500 text-red-500" : "text-muted-foreground"}`}
            />
          </Button>
        )}
      </div>

      <CardHeader className="pb-2">
        <h3 className="font-semibold text-card-foreground line-clamp-2 leading-tight group-hover:text-primary transition-colors duration-300">
          {movie.title}
        </h3>
      </CardHeader>

      <CardContent className="pt-0">
        <div className="flex items-center justify-between">
          <Badge variant="outline" className="text-xs">
            {movie.genre}
          </Badge>
          {movie.rating_count && <span className="text-xs text-muted-foreground">{movie.rating_count} ratings</span>}
        </div>
      </CardContent>
    </Card>
  )
}
