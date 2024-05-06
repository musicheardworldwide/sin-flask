#!/bin/bash

# Start Ollama
ollama serve &
ollama run phi3:latest &

# Start the API
python3 app.py &

# Start code interpreter
python3 -m interpreter &

# Start the chat interface
python3 -m http.server 8000 &
