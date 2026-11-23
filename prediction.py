import pickle
<<<<<<< HEAD
from pathlib import Path

from preprocess import prepare_data


with (Path(__file__).resolve().parent / "PhishingDomainDetection.pkl").open("rb") as file:
    classifier = pickle.load(file)


def predictor(url: str) -> int:
    result = classifier.predict(prepare_data(url))
    return int(result[0])
=======
from preprocess import prepare_data

with open("PhishingDomainDetection.pkl", "rb") as file:
    classifier = pickle.load(file)


def predictor(URL):
    result = classifier.predict(prepare_data(URL))
    return result[0]
>>>>>>> cbbf5d3c2d9da247cbfc2a4f115d31cdac9691a8
