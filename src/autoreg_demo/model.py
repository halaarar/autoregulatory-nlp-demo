from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic_autoreg.csv"

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(
            f"{DATA} not found. Run: python -m src.autoreg_demo.synth_data"
        )
    df = pd.read_csv(DATA)
    return df["text"].tolist(), df["label"].tolist()

def train_and_eval():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")

    return pipe

if __name__ == "__main__":
    train_and_eval()
