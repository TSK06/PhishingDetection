import pickle
from pathlib import Path

from preprocess import prepare_data


with (Path(__file__).resolve().parent / "PhishingDomainDetection.pkl").open("rb") as file:
    classifier = pickle.load(file)


def predictor(url: str) -> int:
    result = classifier.predict(prepare_data(url))
    return int(result[0])