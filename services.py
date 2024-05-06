from models import OllamaModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
import spacy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import MySQLdb
import csv
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC

import torch.nn as nn
import torch.optim as optim

class OllamaContainer:
    def __init__(self):
        self.train_data = None
        self.use_tool_data = None
        self.model = None
        self.tokenizer = None
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = MultinomialNB()
        self.vectorizer = TfidfVectorizer()
        self.lemmatizer = WordNetLemmatizer()
        self.interpreter = openai.Interpreter()
        self.db = MySQLdb.connect(host="your_host", user="your_user", passwd="your_passwd", db="your_db")

    def use_tool(self, data):
        self.use_tool_data = data
        tool_type = data['tool_type']
        if tool_type == 'api_call':
            api_url = data['api_url']
            response = requests.get(api_url)
            return {'result': response.json()}
        elif tool_type == 'db_query':
            query = data['query']
            cursor = self.db.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            return {'result': results}
        else:
            return {'error': 'Invalid tool type'}

    def train(self, data):
        self.train_data = data
        X, y = self.prepare_data(data)
        self.model = OllamaModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(5):  # train for 5 epochs
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        return {'message': 'Model trained successfully'}

    def prepare_data(self, data):
        X = []
        y = []
        for example in data:
            text = example['text']
            label = example['label']
            tokens = word_tokenize(text)
            X.append(' '.join(tokens))
            y.append(label)
        le = LabelEncoder()
        y = le.fit_transform(y)
        return X, y

    def classify_text(self, text):
        tokens = word_tokenize(text)
        X = [' '.join(tokens)]
        X_vectors = self.vectorizer.transform(X)
        prediction = self.sentiment_analyzer.predict(X_vectors)
        return {'result': prediction[0]}

    def sentiment_analysis(self, text):
        tokens = word_tokenize(text)
        X = [' '.join(tokens)]
        X_vectors = self.vectorizer.transform(X)
        prediction = self.sentiment_analyzer.predict(X_vectors)
        if prediction[0] == 0:
            return {'result': 'Negative'}
        elif prediction[0] == 1:
            return {'result': 'Positive'}
        else:
            return {'result': 'Neutral'}

    def natural_language_processing(self, text):
        doc = self.spacy_nlp(text)
        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        return {'result': entities}

    def open_interpreter(self, code):
        result = {}
        try:
            exec(code, globals(), result)
            return {'output': result}
        except Exception as e:
            return {'error': str(e)}

    def pos_tag(self, text):
        doc = self.spacy_nlp(text)
        pos_tags = [{'word': token.text, 'pos': token.pos_} for token in doc]
        return {'result': pos_tags}

    def ner(self, text):
        doc = self.spacy_nlp(text)
        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        return {'result': entities}

    def lemmatize(self, text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return {'result': lemmatized_tokens}

    def interpreter_run(self, code):
        result = self.interpreter.run(code)
        return {'output': result}

class IntentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.clf = SVC(kernel='linear', C=1.0)

    def train(self, X, y):
        X_vectors = self.vectorizer.fit_transform(X)
        self.clf.fit(X_vectors, y)


    def predict(self, message):
        message_vector = self.vectorizer.transform([message])
        return self.clf.predict(message_vector)[0]

def load_labeled_data(filename):
    X, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            X.append(row[0])  # message text
            y.append(row[1])  # intent label
    return X, y

def classify_intent(self, message):
    intentclf = IntentClassifier()
    X_train, y_train = load_labeled_data('intent_data.csv')
    intentclf.train(X_train, y_train)
    return intentclf.predict(message)

def get_conversation_summary(self, conversation_id):
    summary = self.cursor.execute("SELECT summary_text FROM Message_Summaries WHERE conversation_id = %s", (conversation_id,)).fetchone()
    if summary:
        return summary[0]
    else:
        return "No summary available"

def cosine_similarity(messages):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(messages)
    similarity = cosine_similarity(vectors)
    return similarity
