import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load and preprocess dataset
def load_data():
    df = pd.read_csv('mhtcet_chatbot_dataset.csv')
    questions = df['question']
    answers = df['answer']
    return questions, answers

# Train model and save vectorizer and model
def train_model():
    questions, answers = load_data()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)
    model = LogisticRegression()
    model.fit(X, answers)

    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(model, 'chatbot_model.pkl')

# Predict response
def get_response(user_input):
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('chatbot_model.pkl')
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    return prediction
