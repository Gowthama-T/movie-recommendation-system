export function LoadingSpinner() {
  return (
    <div className="flex items-center gap-2">
      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground"></div>
      <span>Loading...</span>
    </div>
  )
}
