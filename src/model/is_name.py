from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline


def make_model():
    return Pipeline([
        ('dict_vect', DictVectorizer(sparse=False)),  # Convert feature dicts to vectors
        ('rf', RandomForestClassifier(random_state=42, verbose=2))  # Classifier
    ])
