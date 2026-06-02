"""
Entrena XGBoost + Logistic Regression con split temporal.
Guarda modelos en models/ para uso en predicciones.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from features import to_diff_matrix

DATA_DIR = os.path.join(_SRC, "..", "data")
MODEL_DIR = os.path.join(_SRC, "..", "models")

TRAIN_CUTOFF = "2023-01-01"


class CalibratedXGB:
    """XGBoost con Platt scaling para probabilidades calibradas."""
    def __init__(self, xgb_model, cal_lr):
        self.xgb = xgb_model
        self.cal = cal_lr

    def predict_proba(self, X):
        raw = self.xgb.predict_proba(X)[:, 1].reshape(-1, 1)
        p1 = self.cal.predict_proba(raw)[:, 1]
        return np.column_stack([1 - p1, p1])


def train_tour(tour: str):
    feat_path = os.path.join(DATA_DIR, f"features_{tour}.parquet")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"No encontrado: {feat_path}\nEjecuta primero: python src/features.py"
        )

    print(f"\n[{tour.upper()}] Cargando features...")
    feat_df = pd.read_parquet(feat_path)
    feat_df["tourney_date"] = pd.to_datetime(feat_df["tourney_date"])

    train_df = feat_df[feat_df["tourney_date"] < TRAIN_CUTOFF]
    test_df = feat_df[feat_df["tourney_date"] >= TRAIN_CUTOFF]
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,} | Corte: {TRAIN_CUTOFF}")

    X_train, y_train = to_diff_matrix(train_df, augment=True)
    X_test, y_test = to_diff_matrix(test_df, augment=True)
    feature_names = X_train.columns.tolist()

    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    print(f"  Features: {len(feature_names)} | X_train: {X_train.shape} | X_test: {X_test.shape}")

    # XGBoost
    print("\nEntrenando XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )

    # Calibración Platt scaling: LR sobre las probabilidades brutas de XGBoost
    raw_probs_train = xgb_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
    cal_lr = LogisticRegression(C=1e4, max_iter=1000)
    cal_lr.fit(raw_probs_train, y_train)

    xgb_cal = CalibratedXGB(xgb_model, cal_lr)

    # Logistic Regression (baseline calibrado)
    print("\nEntrenando Logistic Regression...")
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])
    lr.fit(X_train, y_train)

    # Evaluación
    print(f"\n{'—'*55}")
    print(f"  Evaluación en test ({TRAIN_CUTOFF} en adelante)")
    print(f"{'—'*55}")
    for name, model in [("XGBoost (calibrado)", xgb_cal), ("Logistic Regression", lr)]:
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        print(f"  {name}")
        print(f"    Accuracy : {accuracy_score(y_test, preds)*100:.2f}%")
        print(f"    Log-loss : {log_loss(y_test, probs):.4f}")
        print(f"    Brier    : {brier_score_loss(y_test, probs):.4f}")

    # Alta confianza
    probs_xgb = xgb_cal.predict_proba(X_test)[:, 1]
    for thresh in [0.60, 0.65, 0.70]:
        mask = (probs_xgb >= thresh) | (probs_xgb <= (1 - thresh))
        if mask.sum() > 0:
            correct = (
                ((probs_xgb >= thresh) & (y_test == 1)) |
                ((probs_xgb <= (1 - thresh)) & (y_test == 0))
            ).sum()
            print(f"\n  Confianza >={thresh*100:.0f}%: {mask.sum()} partidos | "
                  f"Accuracy: {correct/mask.sum()*100:.2f}%")
    print(f"{'—'*55}")

    # Importancia de features (XGBoost)
    imp = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\n  Top 10 features más importantes:")
    for feat, val in imp.head(10).items():
        print(f"    {feat:<30} {val:.4f}")

    # Guardar
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(xgb_cal.xgb, os.path.join(MODEL_DIR, f"xgb_{tour}.joblib"))
    joblib.dump(xgb_cal.cal, os.path.join(MODEL_DIR, f"xgb_cal_{tour}.joblib"))
    joblib.dump(lr, os.path.join(MODEL_DIR, f"lr_{tour}.joblib"))
    joblib.dump(medians, os.path.join(MODEL_DIR, f"medians_{tour}.joblib"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, f"feature_names_{tour}.joblib"))
    print(f"\n  Modelos guardados en models/")


if __name__ == "__main__":
    for tour in ["atp", "wta"]:
        train_tour(tour)
    print("\nSiguiente paso: python src/predict.py --help")
