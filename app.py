from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_utils import MHTCETChatbot

app = Flask(__name__)
CORS(app)

chatbot = MHTCETChatbot("dataset.csv")

@app.route("/")
def home():
    return "MHT-CET Chatbot backend is live!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    response = chatbot.get_response(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
