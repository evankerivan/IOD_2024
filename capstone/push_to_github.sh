#!/bin/bash

# Navigate to the project directory where week_data.csv is located
cd /Desktop/IOD_2024/capstone

# Git commands to add, commit, and push the changes
git add week_data.csv
git commit -m "Update week_data with latest air quality data"  # Customize the commit message if needed
git push origin main  # Replace 'main' with your branch if different
