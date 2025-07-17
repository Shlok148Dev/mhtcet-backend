from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_utils import get_response, train_model

app = Flask(__name__)
CORS(app)

# Train model once at start
train_model()

@app.route('/')
def index():
    return "✅ MHT-CET Chatbot Backend is Live!"

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_input = data.get('question', '')

    if not user_input:
        return jsonify({'error': 'No question provided'}), 400

    response = get_response(user_input)
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)

