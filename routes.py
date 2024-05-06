from flask import Blueprint, request, jsonify
from models import OllamaContainer
from nltk.tokenize import word_tokenize
import openai

api = Blueprint('api', __name__)

ollama_container = OllamaContainer()

@api.route('/classify_request', methods=['POST'])
def classify_request():
    data = request.get_json()
    request_type = ollama_container.determine_tool(data)
    if request_type == 'lemmatizer':
        result = ollama_container.lemmatize(data)
    elif request_type == 'tokenizer':
        result = [{'token': token} for token in word_tokenize(data)]
    elif request_type == 'SentimentAnalyzer':
        result = ollama_container.sentiment_analysis(data)
    elif request_type == 'natural_language_processing':
        result = ollama_container.natural_language_processing(data)
    elif request_type == 'pos_tag':
        result = ollama_container.pos_tag(data)
    elif request_type == 'ner':
        result = ollama_container.ner(data)
    elif request_type == 'classify_text':
        result = ollama_container.classify_text(data)
    elif request_type == 'interpreter':
        result = ollama_container.open_interpreter(data)
    else:
        result = {'error': 'Invalid request'}
    return jsonify({'result': result})
