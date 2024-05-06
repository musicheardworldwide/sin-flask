import requests
import time

# Set the API endpoint URL
api_url = "http://localhost:5000/interpreter"

# Set the code to execute in the Open Interpreter
code = "print('Hello, Ollama!')"

while True:
    # Send a POST request to the Open Interpreter endpoint
    response = requests.post(api_url, json={"code": code})

    # Check if the response was successful
    if response.status_code == 200:
        print("Open Interpreter executed successfully!")
        print("Output:", response.json().get("output"))
    else:
        print("Error executing Open Interpreter:", response.text)

    # Wait for 1 second before executing again
    time.sleep(1)
