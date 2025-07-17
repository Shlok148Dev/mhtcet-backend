from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset
df = pd.read_csv("mhtcet_chatbot_dataset.csv")

# Prepare questions and answers
questions = df["question"].fillna("").tolist()
answers = df["answer"].fillna("").tolist()

# Vectorize questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Route for chatbot query
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_query = data.get("question", "")

    if not user_query:
        return jsonify({"answer": "Please ask a valid question."})

    # Transform user query into vector
    user_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vec, X)
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    # Set a similarity threshold
    if best_score < 0.3:
        return jsonify({"answer": "Sorry, I couldn't find a good match. Please rephrase your question."})

    return jsonify({"answer": answers[best_match_index]})

# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

