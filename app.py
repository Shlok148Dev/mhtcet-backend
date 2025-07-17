from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Initialize app
app = Flask(__name__)
CORS(app)

# Load dataset
data = pd.read_csv('mhtcet_chatbot_dataset.csv')

# Prepare data
questions = data['question']
answers = data['answer']

# Vectorize questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Train model
model = LogisticRegression()
model.fit(X, answers)

# Route to test chatbot
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('question')
    if not user_input:
        return jsonify({'error': 'No question provided'}), 400

    input_vec = vectorizer.transform([user_input])
    response = model.predict(input_vec)[0]

    return jsonify({'answer': response})

# Test route
@app.route('/')
def index():
    return "MHT-CET Chatbot Backend is Live!"

if __name__ == '__main__':
    app.run(debug=True)
