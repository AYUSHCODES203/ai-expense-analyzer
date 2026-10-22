
from pathlib import Path

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path(__file__).parent / "data" / "transactions_training.csv"


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["description", "category"])
    df["description"] = df["description"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    df = df[
        (df["description"] != "")
        & (df["category"] != "")
    ]

    return df


def build_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                strip_accents="unicode",
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])


def evaluate_model():
    df = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["description"],
        df["category"],
        test_size=0.25,
        random_state=42,
        stratify=df["category"],
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    return accuracy, report


def train_full_model():
    df = load_dataset()

    model = build_model()
    model.fit(df["description"], df["category"])

    return model


def predict_category(model, description):
    description = description.strip()

    if not description:
        raise ValueError("Description cannot be empty.")

    predicted_category = model.predict([description])[0]

    probabilities = model.predict_proba([description])[0]
    confidence = float(probabilities.max())

    return predicted_category, confidence
