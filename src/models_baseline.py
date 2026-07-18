import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

MODEL_DIR = "models"

def train_baseline(X_train, y_train, X_test, y_test):
    print("\nОбучение Baseline (TF-IDF + LogisticRegression):")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=500, n_jobs=-1, C=1.0, solver='lbfgs')
    model.fit(X_train_tfidf, y_train)

    joblib.dump(model, os.path.join(MODEL_DIR, "baseline_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"Baseline Test Accuracy: {acc:.4f}, Macro F1: {f1_macro:.4f}")

    return model, vectorizer, y_pred
