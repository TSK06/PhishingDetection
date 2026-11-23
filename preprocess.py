import pickle
from pathlib import Path

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r"[A-Za-z]+")
stemmer = SnowballStemmer("english")

with (Path(__file__).resolve().parent / "vectorizer.pkl").open("rb") as file:
    cvectorizer = pickle.load(file)


def prepare_data(text: str):
    tokens = tokenizer.tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return cvectorizer.transform([stemmed_text])
