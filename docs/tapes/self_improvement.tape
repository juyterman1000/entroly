Output docs/assets/self_improvement.gif
Output docs/assets/self_improvement.mp4

Require entroly

Set FontSize 16
Set Width 1400
Set Height 900
Set Padding 40
Set Theme "TokyoNight"
Set WindowBar Colorful
Set Framerate 30

# Title
Type "# Entroly Self-Improvement: Watch the context engine learn your codebase"
Enter
Sleep 2s

# Show initial PRISM weights (day 1)
Type "entroly status --weights"
Enter
Sleep 3s

# Run a coding session — Entroly observes and learns
Type 'entroly wrap claude -p "Find all authentication endpoints and explain the auth flow"'
Enter
Sleep 15s

# Show PRISM weights again — they've shifted
Type "entroly status --weights"
Enter
Sleep 3s

# Run autotune to show the learning loop
Type "entroly autotune --show-delta"
Enter
Sleep 8s

# Show the dreaming loop kicking in
Type "entroly daemon --dream-once"
Enter
Sleep 10s

# Final status — weights have improved
Type "entroly status --weights"
Enter
Sleep 3s

# Show the improvement summary
Type "entroly digest --since 1h"
Enter
Sleep 5s
