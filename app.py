from flask import Flask, request, jsonify, render_template
from preprocess import preprocess_text
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and vectorizer
model = joblib.load("models/Random_Forest_best_model.pkl")
vectorizer = joblib.load("models/Random_Forest_best_vectorizer.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    text = request.form['text']
    preprocessed_text = preprocess_text(text, vectorizer)
    prediction = model.predict(preprocessed_text)
    probability = model.predict_proba(preprocessed_text)
    prediction_value = int(prediction[0])  # Convert to native Python 
    
    probability_real= float(probability[0][1])  # Convert to native Python float for fake
    probability_fake= float(probability[0][0])  # Convert to native Python float for real
    return render_template('result.html', prediction=prediction_value, probability_fake=probability_fake, probability_real=probability_real)

if __name__ == '__main__':
    app.run(debug=True)
