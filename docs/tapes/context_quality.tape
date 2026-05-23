Output docs/assets/context_quality.gif
Output docs/assets/context_quality.mp4

Require entroly

Set FontSize 16
Set Width 1400
Set Height 900
Set Padding 40
Set Theme "TokyoNight"
Set WindowBar Colorful
Set Framerate 30

# Title
Type "# Entroly Context Quality: Before vs After — watch quality improve over time"
Enter
Sleep 2s

# Show initial context quality (before learning)
Type "entroly health"
Enter
Sleep 5s

# Run benchmark to establish baseline
Type "entroly benchmark --baseline"
Enter
Sleep 10s

# Now run several requests to let PRISM learn
Type 'entroly wrap claude -p "What modules depend on depgraph.rs?"'
Enter
Sleep 12s

Type 'entroly wrap claude -p "Show me all error handling patterns in the proxy"'
Enter
Sleep 12s

# Run benchmark again — quality has improved
Type "entroly benchmark --compare-baseline"
Enter
Sleep 8s

# Show the health grade improvement
Type "entroly health"
Enter
Sleep 5s

# Show explain output — why each fragment was included
Type "entroly explain --last-request"
Enter
Sleep 5s

# Final comparison
Type "echo '--- Context quality improved: PRISM weights optimized from usage ---'"
Enter
Sleep 3s
