from flask import Flask, render_template, request
import pickle
import re

# Load the model and vectorizer
with open('spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    if request.method == 'POST':
        message = request.form['message']
        cleaned = clean_text(message)
        vect = vectorizer.transform([cleaned]).toarray()
        result = model.predict(vect)[0]
        prediction = "Spam" if result == 1 else "Ham"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
