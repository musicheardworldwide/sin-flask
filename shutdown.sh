#!/bin/bash

# Kill the API process
pkill -f app.py

# Kill the code interpreter process
pkill -f interpreter

# Kill the Ollama process
pkill -f ollama

# Kill the HTTP server process
pkill -f http.server

echo "All components shut down!"
