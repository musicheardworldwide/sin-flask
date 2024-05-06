# Install Python packages
pip install -r requirements.txt

# Install necessary NLTK data
python3 -m nltk.downloader punkt wordnet

# Install spaCy models
python3 -m spacy download en_core_web_sm

echo "Installation complete!"