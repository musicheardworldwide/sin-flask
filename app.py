from flask import Flask
from routes import api

app = Flask(__name__)
app.config["BASE_URL"] = "http://cleardiamondmedia.com/api"  # or "example.com/api" if you want to use a domain

app.register_blueprint(api, url_prefix=app.config["BASE_URL"])

if __name__ == "__main__":
    app.run(debug=True, port=3333)  # run on port 3333
