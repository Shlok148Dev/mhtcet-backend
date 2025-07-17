from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
df = pd.read_csv("mhtcet_chatbot_dataset.csv")  # your downloaded CSV

questions = df['question'].tolist()
answers = df['answer'].tolist()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.get_json()["query"]
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    best_match_idx = similarities.argmax()
    response = answers[best_match_idx]
    return jsonify({"response": response})

app.run(host="0.0.0.0", port=10000)
