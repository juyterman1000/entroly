Output docs/assets/token_savings.gif
Output docs/assets/token_savings.mp4

Require entroly

Set FontSize 16
Set Width 1400
Set Height 900
Set Padding 40
Set Theme "TokyoNight"
Set WindowBar Colorful
Set Framerate 30

# Title
Type "# Entroly Token Savings: See the cost reduction in real time"
Enter
Sleep 2s

# Run the demo to show before/after on this repo
Type "entroly demo"
Enter
Sleep 10s

# Show live savings dashboard
Type "entroly dashboard --snapshot"
Enter
Sleep 5s

# Run a real request through the proxy and show savings
Type "entroly proxy --quality balanced &"
Enter
Sleep 3s

Type 'curl -s http://localhost:9377/v1/chat/completions -H "Content-Type: application/json" -d '"'"'{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Explain the knapsack solver in entroly-core"}]}'"'"' | jq .usage'
Enter
Sleep 10s

# Show cumulative savings
Type "entroly status --savings"
Enter
Sleep 5s

# Show the digest with dollar amounts
Type "entroly digest"
Enter
Sleep 5s
