import os
import math
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())

def max_consecutive_consonants(s: str) -> int:
    consonants = set("bcdfghjklmnpqrstvwxyz")
    max_c, curr_c = 0, 0
    for char in s.lower():
        if char in consonants:
            curr_c += 1
            max_c = max(max_c, curr_c)
        else:
            curr_c = 0
    return max_c

class LexicalFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        vowels = set("aeiou")
        digits = set("0123456789")
        for domain in X:
            d = str(domain).lower().strip()
            parts = d.split(".")
            sld = parts[-2] if len(parts) >= 2 else parts[0]
            len_sld = len(sld)
            features.append([
                len(d),
                len_sld,
                shannon_entropy(sld),
                max(0, len(parts) - 2),
                (sum(1 for c in sld if c in digits) / len_sld) if len_sld > 0 else 0.0,
                (sum(1 for c in sld if c in vowels) / len_sld) if len_sld > 0 else 0.0,
                d.count("-"),
                max_consecutive_consonants(sld),
            ])
        return np.array(features)

if __name__ == "__main__":
    df = pd.read_csv("data/domains.csv").dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        df["domain"].astype(str), df["label"].astype(int),
        test_size=0.25, random_state=42, stratify=df["label"]
    )

    features_pipeline = FeatureUnion([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=5000, sublinear_tf=True)),
        ("lexical", Pipeline([("extractor", LexicalFeatureExtractor()), ("scaler", StandardScaler())]))
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1)
    }

    results = {}
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    for name, clf in models.items():
        print(f"[*] Entraînement : {name}...")
        pipe = Pipeline([("features", features_pipeline), ("clf", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        results[name] = {"pipeline": pipe, "accuracy": acc, "precision": p, "recall": r, "f1": f1, "cm": confusion_matrix(y_test, preds)}
        print(f"    -> Exactitude: {acc*100:.2f}% | Précision: {p*100:.2f}% | Rappel: {r*100:.2f}% | F1: {f1*100:.2f}%")

    best_name = max(results, key=lambda k: results[k]["f1"])
    joblib.dump(results[best_name]["pipeline"], "models/best_domain_detector.pkl")
    print(f"\n[+] Meilleur modèle ({best_name}) sauvegardé dans models/best_domain_detector.pkl")

    # Graphique des matrices
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, (name, m) in enumerate(results.items()):
        sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["Bénin", "Malveillant"], yticklabels=["Bénin", "Malveillant"])
        axes[idx].set_title(name)
    plt.tight_layout()
    plt.savefig("reports/matrices.png", dpi=200)
    plt.close()

    # Rapport PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Rapport d'Evaluation : Detection de Domaines Malveillants", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(230, 235, 245)
    for col, w in [("Modele", 50), ("Exactitude", 35), ("Precision", 35), ("Rappel", 35), ("F1-Score", 35)]:
        pdf.cell(w, 7, col, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    for name, m in results.items():
        pdf.cell(50, 6, name, border=1)
        pdf.cell(35, 6, f"{m['accuracy']*100:.2f}%", border=1)
        pdf.cell(35, 6, f"{m['precision']*100:.2f}%", border=1)
        pdf.cell(35, 6, f"{m['recall']*100:.2f}%", border=1)
        pdf.cell(35, 6, f"{m['f1']*100:.2f}%", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.image("reports/matrices.png", x=10, w=190)
    pdf.output("reports/rapport_final.pdf")
    print("[+] Rapport généré dans reports/rapport_final.pdf")