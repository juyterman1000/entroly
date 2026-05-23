Output entroly_claude_demo.gif
Output entroly_claude_demo.mp4

Require entroly
Require claude

Set FontSize 16
Set Width 1400
Set Height 900
Set Padding 40
Set Theme "TokyoNight"
Set WindowBar Colorful
Set Framerate 30

# Switch to the Langfuse monorepo directory
Type "cd C:\Users\abhis\langfuse\langfuse"
Enter
Sleep 1.5s

# Run the real Entroly wrapper with Claude Code
Type 'entroly wrap claude -p "Explain how trace ingestion works in the worker package"'
Sleep 1s
Enter

# Wait for Entroly to boot, Claude to analyze the repo, intercept the massive context, compress it, and stream the answer.
Sleep 35s

# Pause at the end so people can read the savings summary
Sleep 5s
