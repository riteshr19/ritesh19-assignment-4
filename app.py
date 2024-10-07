from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Fetch dataset, initialize vectorizer and LSA here
newsgroups = fetch_20newsgroups(subset='all')
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(newsgroups.data)
lsa = TruncatedSVD(n_components=100)
X_reduced = lsa.fit_transform(X)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    query_vec = vectorizer.transform([query])
    query_reduced = lsa.transform(query_vec)
    similarities = cosine_similarity(query_reduced, X_reduced).flatten()
    indices = similarities.argsort()[-5:][::-1]
    documents = [newsgroups.data[i] for i in indices]
    similarities = similarities[indices].tolist()  # Convert to list
    indices = indices.tolist()  # Convert to list
    return documents, similarities, indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({
        'documents': documents,
        'similarities': similarities,
        'indices': indices
    })

if __name__ == '__main__':
    app.run(debug=True, port=3000)