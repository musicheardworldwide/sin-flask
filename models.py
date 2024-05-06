from sklearn.naive_bayes import MultinomialNB
import spacy
import nltk
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_sm")  # Load the English model

class OllamaContainer:  
    def __init__(self): 
        self.text_classifier = None
        self.vectorizer = None  
        

    def determine_tool(self, text):
        if 'lemmatize' in text:
            return 'lemmatizer'
        elif 'tokenize' in text:
            return 'tokenizer'
        elif 'classify' in text:
            return 'SentimentAnalyzer'
        else:
            return 'ner'

    def pos_tag(self, text):
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        return [{'word': word, 'pos': pos} for word, pos in pos_tags]

    def ner(self, text):
        doc = nlp(text)
        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        return entities

    def train_text_classifier(self, data):
        X = [example['text'] for example in data]
        y = [example['label'] for example in data]

        vectorizer = TfidfVectorizer()
        X_vectors = vectorizer.fit_transform(X)

        clf = MultinomialNB()
        clf.fit(X_vectors, y)

        self.text_classifier = clf
        self.vectorizer = vectorizer

    def classify_text(self, text):
        vector = self.vectorizer.transform([text])
        prediction = self.text_classifier.predict(vector)
        return prediction[0]

    def open_interpreter(self, code):
        result = {}
        try:
            exec(code, globals(), result)
            return {'output': result}
        except Exception as e:
            return {'error': str(e)}
