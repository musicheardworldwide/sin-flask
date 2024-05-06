import mysql.connector
from nltk.tokenize import word_tokenize
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

class DB:
    def __init__(self):
        self.cnx = mysql.connector.connect(
            user='your_user',
            password='your_passwd',
            host='your_host',
            database='your_db'
        )
        self.cursor = self.cnx.cursor()

    def create_tables(self):
        # Create Conversations table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS Conversations (
                conversation_id INT AUTO_INCREMENT,
                user_id INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (conversation_id)
            );
        """)

        # Create Messages table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS Messages (
                message_id INT AUTO_INCREMENT,
                conversation_id INT,
                text TEXT,
                is_user_input BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (message_id),
                FOREIGN KEY (conversation_id) REFERENCES Conversations(conversation_id)
            );
        """)

        # Create Message_Summaries table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS Message_Summaries (
                summary_id INT AUTO_INCREMENT,
                conversation_id INT,
                summary_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (summary_id),
                FOREIGN KEY (conversation_id) REFERENCES Conversations(conversation_id)
            );
        """)

    def populate_initial_data(self):
        # Insert sample conversations, messages, and summaries
        pass

    def summarize_conversation(self, conversation_id):
        # Retrieve conversation messages
        messages = self.cursor.execute("SELECT text, is_user_input FROM Messages WHERE conversation_id = %s", (conversation_id,)).fetchall()

        # Initialize graph for TextRank
        graph = nx.Graph()

        # Add nodes for each message
        for message in messages:
            graph.add_node(message[0], text=message[0], is_user_input=message[1])

        # Add edges between messages based on cosine similarity
        for i in range(len(messages)):
            for j in range(i+1, len(messages)):
                cosine_sim = cosine_similarity([messages[i][0], messages[j][0]])
                if cosine_sim > 0.5:
                    graph.add_edge(messages[i][0], messages[j][0], weight=cosine_sim)

        # Compute TextRank scores
        scores = nx.pagerank(graph, alpha=0.85, personalization=None, max_iter=100, tol=1e-6)

        # Identify top-scoring sentences (summaries)
            def summarize_conversation(self, conversation_id):
        # Retrieve conversation messages
        messages = self.cursor.execute("SELECT text, is_user_input FROM Messages WHERE conversation_id = %s", (conversation_id,)).fetchall()

        # Initialize graph for TextRank
        graph = nx.Graph()

        # Add nodes for each message
        for message in messages:
            graph.add_node(message[0], text=message[0], is_user_input=message[1])

        # Add edges between messages based on cosine similarity
        for i in range(len(messages)):
            for j in range(i+1, len(messages)):
                cosine_sim = cosine_similarity([messages[i][0], messages[j][0]])
                if cosine_sim > 0.5:
                    graph.add_edge(messages[i][0], messages[j][0], weight=cosine_sim)

        # Compute TextRank scores
        scores = nx.pagerank(graph, alpha=0.85, personalization=None, max_iter=100, tol=1e-6)

        # Identify top-scoring sentences (summaries)
        summaries = [(node, score) for node, score in scores.items()]
        summaries.sort(key=lambda x: x[1], reverse=True)

        # Identify intent for each message
        intents = []
        for message in messages:
            intent = self.classify_intent(message[0])
            intents.append((message[0], intent))

        # Store summary and intents in Message_Summaries table
        summary_text = '\n'.join([summary[0] for summary in summaries[:3]])
        self.cursor.execute("INSERT INTO Message_Summaries (conversation_id, summary_text, intents) VALUES (%s, %s, %s)", (conversation_id, summary_text, json.dumps(intents)))
        self.cnx.commit()

    def get_conversation_summary(self, conversation_id):
        # Retrieve summary from Message_Summaries table
        summary = self.cursor.execute("SELECT summary_text FROM Message_Summaries WHERE conversation_id = %s", (conversation_id,)).fetchone()
        if summary:
            return summary[0]
        else:
            return "No summary available"

    def close(self):
        # Close database connection
        self.cursor.close()
        self.cnx.close()

    def populate_initial_data(self):
        conversations = [
            {'user_id': 1, 'created_at': '2022-01-01 00:00:00', 'updated_at': '2022-01-01 00:00:00'},
            {'user_id': 2, 'created_at': '2022-01-05 00:00:00', 'updated_at': '2022-01-05 00:00:00'}
        ]

        messages = [
            {'conversation_id': 1, 'text': 'Hello, how are you?', 'is_user_input': True},
            {'conversation_id': 1, 'text': 'I\'m fine, thank you.', 'is_user_input': False},
            {'conversation_id': 2, 'text': 'What\'s your favorite food?', 'is_user_input': True},
            {'conversation_id': 2, 'text': 'I love pizza!', 'is_user_input': False}
        ]

        summaries = [
            {'conversation_id': 1, 'summary_text': 'Hello, how are you?'},
            {'conversation_id': 2, 'summary_text': 'What\'s your favorite food?'}
        ]

        self.cursor.executemany("INSERT INTO Conversations (user_id, created_at, updated_at) VALUES (%(user_id)s, %(created_at)s, %(updated_at)s)", conversations)
        self.cursor.executemany("INSERT INTO Messages (conversation_id, text, is_user_input) VALUES (%(conversation_id)s, %(text)s, %(is_user_input)s)", messages)
        self.cursor.executemany("INSERT INTO Message_Summaries (conversation_id, summary_text) VALUES (%(conversation_id)s, %(summary_text)s)", summaries)
        self.cnx.commit()

