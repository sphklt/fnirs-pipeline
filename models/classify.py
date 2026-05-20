"""
fNIRS Mental State Classification
-----------------------------------
Trains and evaluates classifiers to decode mental state from fNIRS features.

Models compared:
  - Linear Discriminant Analysis (LDA)  — BCI community standard, interpretable
  - Random Forest                        — handles non-linear interactions
  - SVM (RBF kernel)                    — strong baseline for small-N BCI datasets
  - Gradient Boosting                   — often best performance

Evaluation:
  - Stratified K-Fold cross-validation (5-fold)
  - Accuracy, F1-macro, confusion matrix
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings("ignore")


CLASS_NAMES = ["REST", "MENTAL", "MOTOR"]

MODELS = {
    "LDA": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]),
}


def evaluate_all_models(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """
    Cross-validate all models and return results dict.

    Returns:
        results: {model_name: {"acc_mean", "acc_std", "f1_mean", "f1_std"}}
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for name, pipeline in MODELS.items():
        scores = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring=["accuracy", "f1_macro"],
            return_train_score=False,
        )
        results[name] = {
            "acc_mean": scores["test_accuracy"].mean(),
            "acc_std":  scores["test_accuracy"].std(),
            "f1_mean":  scores["test_f1_macro"].mean(),
            "f1_std":   scores["test_f1_macro"].std(),
            "acc_per_fold": scores["test_accuracy"].tolist(),
        }
        print(f"{name:20s}  acc={results[name]['acc_mean']:.3f}±{results[name]['acc_std']:.3f}  "
              f"f1={results[name]['f1_mean']:.3f}±{results[name]['f1_std']:.3f}")

    return results


def train_best_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Train best model (Gradient Boosting) on full dataset.
    Returns (fitted_pipeline, confusion_matrix, report_str).
    Used for the Streamlit demo.
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    best = MODELS["Gradient Boosting"]
    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nBest model test accuracy: {acc:.3f}")
    print(report)

    return best, cm, report, acc


def get_feature_importances(pipeline: Pipeline, top_n: int = 15) -> list:
    """Extract top feature importances from Random Forest / GB model."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:top_n]
        return [(int(i), float(importances[i])) for i in top_idx]
    return []


if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from data.generate import generate_dataset
    from utils.preprocess import preprocess_dataset
    from utils.features import build_feature_matrix

    print("Generating data...")
    raw = generate_dataset()
    proc = preprocess_dataset(raw)
    X, y = build_feature_matrix(proc)

    print("\n--- Cross-validation results ---")
    results = evaluate_all_models(X, y)

    print("\n--- Training best model ---")
    model, cm, report, acc = train_best_model(X, y)
