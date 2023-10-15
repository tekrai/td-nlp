from nltk.stem.snowball import FrenchStemmer

stemmer = FrenchStemmer()


def stem_tokenizer(text):
    """
    Custom tokenizer for stemming French words.
    Keeps only the root of a word example (eated, eaten) becomes eat
    """
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens
