from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def make_model():
    return Pipeline([
        ("tfidf_vectorizer", TfidfVectorizer(stop_words=stopwords.words('french'), strip_accents='unicode')),
        ("random_forest", RandomForestClassifier()),
    ])
