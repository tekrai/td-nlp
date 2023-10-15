from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.features.text_features import stem_tokenizer


def make_model():
    return Pipeline([
        ("tfidf_vectorizer", TfidfVectorizer(tokenizer=stem_tokenizer, stop_words=stopwords.words('french'), strip_accents='unicode')),
        ("random_forest", LogisticRegression()),
    ])
