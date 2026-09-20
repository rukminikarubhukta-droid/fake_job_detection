from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("fake_job_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    accuracy = "≈ 95%"

    if request.method == 'POST':
        job_text = request.form['job_description']
        cleaned = clean_text(job_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            result = "❌ Fake Job Posting"
        else:
            result = "✅ Real Job Posting"

    return render_template('index.html', result=result, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
