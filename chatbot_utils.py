import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MHTCETChatbot:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.df['query'])

    def get_response(self, user_input):
        user_vec = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vec, self.X)
        max_index = np.argmax(similarities)
        confidence = np.max(similarities)

        if confidence < 0.3:
            return "Sorry, I couldn’t understand that. Can you please rephrase?"

        return self.df.iloc[max_index]['response']
