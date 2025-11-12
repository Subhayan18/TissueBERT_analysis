#!/bin/bash

# Set changelog file
CHANGELOG="CHANGELOG.md"

# Get current date
DATE=$(date +"%Y-%m-%d %H:%M:%S")

# Get the *entire* status output
STATUS=$(git status --porcelain)

# If no changes (output is empty), exit
if [ -z "$STATUS" ]; then
  echo "No changes to commit."
  exit 0
fi

# Get just the file names for the log
FILES_FOR_LOG=$(echo "$STATUS" | awk '{print $2}')

# Create changelog entry
echo -e "\n## [$DATE] Auto Commit\n" >> $CHANGELOG
for FILE in $FILES_FOR_LOG; do
  echo "- Found: $FILE" >> $CHANGELOG
done

# Stage all changes (this will add untracked '??' files)
git add .

# Commit with message
git commit -m "Auto commit on $DATE"

# Push to GitHub
git push origin main

echo "Changes committed and pushed successfully."
