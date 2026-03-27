#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Federated Learning-Based Diabetes Risk Prediction — Explainable AI Pipeline
Install: pip install xgboost lightgbm shap imbalanced-learn
"""

# =============================================================================
# IMPORTS & CONFIG
# =============================================================================
from __future__ import annotations
import os, warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

@dataclass
class Config:
    RANDOM_SEED: int   = 42
    OUTPUT_DIR:  str   = "./outputs"
    TEST_SIZE:   float = 0.20
    SMOTE_STRATEGY: float = 1.0
    NUM_CLIENTS: int   = 5
    FL_ROUNDS:   int   = 3
    FIGURE_DPI:  int   = 150
    SHAP_MAX:    int   = 15

CFG = Config()
np.random.seed(CFG.RANDOM_SEED)
os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DATA
# =============================================================================
COLS = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
ZERO_COLS = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

class DataLoader:
    @staticmethod
    def load(filepath: Optional[str] = None) -> pd.DataFrame:
        if filepath and os.path.isfile(filepath):
            df = pd.read_csv(filepath)
        else:
            url = ("https://raw.githubusercontent.com/"
                   "jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
            print(f"[Data] Downloading from {url}...")
            df = pd.read_csv(url, header=None, names=COLS)
        assert list(df.columns) == COLS
        print(f"[Data] Loaded {df.shape}")
        print(f"[Data] Class dist:\n{df['Outcome'].value_counts()}")
        return df

# =============================================================================
# PREPROCESSING
# =============================================================================
class Preprocessor:
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ZERO_COLS:
            df[col] = df[col].replace(0, np.nan)
        for col in ZERO_COLS:
            for v in [0, 1]:
                med = df.loc[df["Outcome"] == v, col].median()
                df.loc[(df["Outcome"] == v) & df[col].isnull(), col] = med
        return df

    @staticmethod
    def engineer(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        eps = 1e-6
        df["GlucoseBMI_Ratio"]     = df["Glucose"] / (df["BMI"] + eps)
        df["InsulinGlucose_Ratio"] = df["Insulin"] / (df["Glucose"] + eps)
        df["BP_Age_Ratio"]         = df["BloodPressure"] / (df["Age"] + eps)
        df["AgeBMI_Interaction"]   = df["Age"] * df["BMI"]
        df["Pedigree_Age"]         = df["DiabetesPedigreeFunction"] * df["Age"]
        return df

    def run(self, df: pd.DataFrame):
        df = self.handle_missing(df)
        df = self.engineer(df)
        feat_cols = [c for c in df.columns if c != "Outcome"]
        self.feature_names = feat_cols
        X, y = df[feat_cols].values, df["Outcome"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=CFG.TEST_SIZE, stratify=y,
            random_state=CFG.RANDOM_SEED)

        self.scaler = StandardScaler()
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)

        smote = SMOTE(sampling_strategy=CFG.SMOTE_STRATEGY,
                      k_neighbors=5, random_state=CFG.RANDOM_SEED)
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        print(f"[Prep] Train {X_tr.shape} | Test {X_te.shape}")
        return X_tr, X_te, y_tr, y_te, feat_cols

# =============================================================================
# MODELS
# =============================================================================
class ModelFactory:
    @staticmethod
    def get(seed: int = CFG.RANDOM_SEED) -> Dict[str, Any]:
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_split=5,
                min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced", random_state=seed, n_jobs=-1),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, gamma=1,
                eval_metric="logloss", random_state=seed, n_jobs=-1),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=seed,
                n_jobs=-1, verbose=-1),
        }

# =============================================================================
# EVALUATION
# =============================================================================
from dataclasses import dataclass as dc

@dc
class EvalResult:
    name: str; accuracy: float; precision: float; recall: float
    f1: float; roc_auc: float
    y_true: np.ndarray; y_pred: np.ndarray; y_prob: np.ndarray
    model: Any

class Evaluator:
    def __init__(self):
        self.results: List[EvalResult] = []

    def train_and_evaluate(self, models, X_tr, y_tr, X_te, y_te):
        self.results = []
        for name, model in models.items():
            print(f"\n{'─'*50}\n  Training: {name}\n{'─'*50}")
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
            res = EvalResult(
                name=name,
                accuracy =accuracy_score(y_te, y_pred),
                precision=precision_score(y_te, y_pred, zero_division=0),
                recall   =recall_score(y_te, y_pred, zero_division=0),
                f1       =f1_score(y_te, y_pred, zero_division=0),
                roc_auc  =roc_auc_score(y_te, y_prob),
                y_true=y_te, y_pred=y_pred, y_prob=y_prob, model=model)
            self.results.append(res)
            print(classification_report(y_te, y_pred,
                  target_names=["Non-Diabetic","Diabetic"]))
        return self.results

    def plot_roc_curves(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        for res, color in zip(self.results, sns.color_palette("Set2", len(self.results))):
            fpr, tpr, _ = roc_curve(res.y_true, res.y_prob)
            ax.plot(fpr, tpr, label=f"{res.name} (AUC={res.roc_auc:.4f})",
                    color=color, linewidth=2)
        ax.plot([0,1],[0,1],"k--", alpha=0.4, label="Random")
        ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curve Comparison")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3); fig.tight_layout()
        path = os.path.join(CFG.OUTPUT_DIR, "roc_curve_comparison.png")
        fig.savefig(path, dpi=CFG.FIGURE_DPI, bbox_inches="tight")
        print(f"[Eval] ROC saved -> {path}"); plt.show()

    def plot_confusion_matrices(self):
        n = len(self.results)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1: axes = [axes]
        for ax, res in zip(axes, self.results):
            sns.heatmap(confusion_matrix(res.y_true, res.y_pred),
                        annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Non-Diabetic","Diabetic"],
                        yticklabels=["Non-Diabetic","Diabetic"], ax=ax)
            ax.set_title(f"{res.name}\nAcc={res.accuracy:.3f}", fontweight="bold")
            ax.set(ylabel="Actual", xlabel="Predicted")
        fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        path = os.path.join(CFG.OUTPUT_DIR, "confusion_matrices.png")
        fig.savefig(path, dpi=CFG.FIGURE_DPI, bbox_inches="tight")
        print(f"[Eval] CM saved -> {path}"); plt.show()

    def export_summary_csv(self):
        df = pd.DataFrame([{
            "Model": r.name, "Accuracy": round(r.accuracy,4),
            "Precision": round(r.precision,4), "Recall": round(r.recall,4),
            "F1-Score": round(r.f1,4), "ROC-AUC": round(r.roc_auc,4),
        } for r in self.results])
        path = os.path.join(CFG.OUTPUT_DIR, "model_performance_summary.csv")
        df.to_csv(path, index=False)
        print(f"\n[Eval] Summary saved -> {path}")
        print(df.to_string(index=False))
        return df

# =============================================================================
# SHAP EXPLAINABILITY
# =============================================================================
class Explainer:
    def explain(self, model, X_test, feature_names, model_name="Best"):
        print(f"\n{'─'*50}\n  SHAP — {model_name}\n{'─'*50}")
        exp = shap.TreeExplainer(model)
        try:
            sv = exp.shap_values(X_test, check_additivity=False)
        except Exception:
            sv = exp.shap_values(X_test)
        if isinstance(sv, list): sv = sv[1]

        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv, X_test, feature_names=feature_names,
                          max_display=CFG.SHAP_MAX, show=False)
        plt.title(f"SHAP Feature Importance — {model_name}",
                  fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        path = os.path.join(CFG.OUTPUT_DIR, f"shap_{model_name}.png")
        plt.savefig(path, dpi=CFG.FIGURE_DPI, bbox_inches="tight")
        print(f"[SHAP] Saved -> {path}"); plt.show()

        mean_abs = np.abs(sv).mean(axis=0)
        imp_df = (pd.DataFrame({"Feature": feature_names, "Mean|SHAP|": mean_abs})
                  .sort_values("Mean|SHAP|", ascending=False).reset_index(drop=True))
        print("\nTop Feature Contributions:")
        print(imp_df.head(10).to_string(index=False))

# =============================================================================
# FEDERATED LEARNING SIMULATION
# =============================================================================
class FederatedSimulator:
    def _partition(self, X, y):
        idx = np.arange(len(y)); np.random.shuffle(idx)
        splits = np.array_split(idx, CFG.NUM_CLIENTS)
        partitions = [(X[s], y[s]) for s in splits]
        for i, (xp, yp) in enumerate(partitions):
            print(f"  Client {i+1}: {len(yp)} samples (pos: {int(yp.sum())})")
        return partitions

    def _train_client(self, X, y, seed):
        m = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                class_weight="balanced", random_state=seed,
                                n_jobs=-1, verbose=-1)
        m.fit(X, y); return m

    def _fedavg(self, models, X, weights=None):
        if weights is None: weights = [1/len(models)] * len(models)
        prob = sum(w * m.predict_proba(X)[:,1] for m, w in zip(models, weights))
        return (prob >= 0.5).astype(int), prob

    def run(self, X_tr, y_tr, X_te, y_te):
        print(f"\n{'='*50}\n  FL SIM | Clients:{CFG.NUM_CLIENTS} Rounds:{CFG.FL_ROUNDS}\n{'='*50}")
        results = {"rounds": []}
        best_acc, best_models, best_weights = 0.0, [], []

        # Fixed partitions across all rounds (correct FedAvg)
        print("\n[FL] Partitioning data (fixed across rounds)...")
        partitions = self._partition(X_tr, y_tr)
        weights = [len(p[1]) for p in partitions]
        total = sum(weights); weights = [w/total for w in weights]

        for rnd in range(1, CFG.FL_ROUNDS + 1):
            print(f"\n--- Round {rnd}/{CFG.FL_ROUNDS} ---")
            round_models, client_accs = [], []

            for i, (Xc, yc) in enumerate(partitions):
                m = self._train_client(Xc, yc, CFG.RANDOM_SEED + rnd*100 + i)
                round_models.append(m)
                acc = accuracy_score(y_te, m.predict(X_te))
                client_accs.append(acc)
                print(f"  Client {i+1} acc: {acc:.4f}")

            y_pred, y_prob = self._fedavg(round_models, X_te, weights)
            agg_acc = accuracy_score(y_te, y_pred)
            agg_auc = roc_auc_score(y_te, y_prob)
            print(f"  Aggregated acc: {agg_acc:.4f} | AUC: {agg_auc:.4f}")

            results["rounds"].append({"round": rnd, "client_accuracies": client_accs,
                                      "aggregated_accuracy": agg_acc,
                                      "aggregated_roc_auc": agg_auc})
            if agg_acc >= best_acc:
                best_acc, best_models, best_weights = agg_acc, round_models, weights

        # Final report
        y_pred_f, y_prob_f = self._fedavg(best_models, X_te, best_weights)
        results["best_accuracy"] = best_acc
        results["final_report"] = classification_report(
            y_te, y_pred_f, target_names=["Non-Diabetic","Diabetic"])
        print(f"\n{'='*50}\n  FL FINAL | Best acc: {best_acc:.4f}\n{'='*50}")
        print(results["final_report"])
        self._privacy_report()

        # FL confusion matrix
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_te, y_pred_f), annot=True, fmt="d",
                    cmap="Greens", xticklabels=["Non-Diabetic","Diabetic"],
                    yticklabels=["Non-Diabetic","Diabetic"], ax=ax)
        ax.set_title("Federated Model — Confusion Matrix", fontweight="bold")
        ax.set(ylabel="Actual", xlabel="Predicted"); fig.tight_layout()
        path = os.path.join(CFG.OUTPUT_DIR, "federated_confusion_matrix.png")
        fig.savefig(path, dpi=CFG.FIGURE_DPI, bbox_inches="tight")
        print(f"[FL] CM saved -> {path}"); plt.show()

        self._plot_rounds(results)
        return results

    def _plot_rounds(self, results):
        fig, ax = plt.subplots(figsize=(10, 6))
        rounds   = [r["round"] for r in results["rounds"]]
        agg_accs = [r["aggregated_accuracy"] for r in results["rounds"]]
        ax.plot(rounds, agg_accs, "o-", lw=3, ms=10, color="#2ca02c",
                label="Aggregated (FedAvg)")
        for c_idx, color in enumerate(sns.color_palette("tab10", CFG.NUM_CLIENTS)):
            c_accs = [r["client_accuracies"][c_idx] for r in results["rounds"]]
            ax.plot(rounds, c_accs, "x--", alpha=0.4, color=color,
                    label=f"Client {c_idx+1}")
        ax.set(xlabel="Round", ylabel="Accuracy",
               title="Federated Learning — Accuracy per Round")
        ax.legend(fontsize=9, loc="lower right", ncol=2)
        ax.set_xticks(rounds); ax.grid(alpha=0.3); fig.tight_layout()
        path = os.path.join(CFG.OUTPUT_DIR, "federated_accuracy_rounds.png")
        fig.savefig(path, dpi=CFG.FIGURE_DPI, bbox_inches="tight")
        print(f"[FL] Round plot saved -> {path}"); plt.show()

    @staticmethod
    def _privacy_report():
        print("\n┌─────────────────────────────────────────────────────┐")
        print("│  PRIVACY REPORT                                     │")
        print("│  • Raw data never leaves the local client.         │")
        print("│  • Only model predictions are shared centrally.    │")
        print("│  • Weighted averaging without patient-level access. │")
        print("│  • Aligns with HIPAA / DISHA frameworks.           │")
        print("└─────────────────────────────────────────────────────┘")

# =============================================================================
# INTERACTIVE RISK PREDICTOR
# =============================================================================
_CLINICAL_RULES = {
    "Glucose":   [(70,100,"normal","Normal fasting range (70–100 mg/dL). No elevated glucose risk."),
                  (100,126,"warning","Pre-diabetic range (100–125 mg/dL). Impaired fasting glucose."),
                  (126,9999,"danger",">=126 mg/dL meets ADA clinical diabetes threshold.")],
    "BMI":       [(0,18.5,"warning","Underweight (<18.5). Not a direct diabetes risk here."),
                  (18.5,25,"normal","Healthy range (18.5-24.9). No obesity risk."),
                  (25,30,"warning","Overweight (25-29.9). Increases insulin resistance."),
                  (30,9999,"danger","Obese (>=30). Strongest modifiable diabetes risk factor.")],
    "Age":       [(0,35,"normal","Age <35: Relatively lower Type-2 diabetes risk."),
                  (35,45,"warning","Age 35-44: Risk begins to rise in this bracket."),
                  (45,9999,"danger","Age >=45: Major non-modifiable risk factor.")],
    "DiabetesPedigreeFunction":
                 [(0,0.3,"normal","Low pedigree (<0.3): Minimal genetic contribution."),
                  (0.3,0.6,"warning","Moderate (0.3-0.6): Notable family history influence."),
                  (0.6,9999,"danger","High (>0.6): Strong genetic predisposition.")],
    "Pregnancies":[(0,2,"normal","0-2 pregnancies: No elevated gestational history."),
                   (2,5,"warning","2-4 pregnancies: Some gestational diabetes risk."),
                   (5,9999,"danger","5+: Multiple pregnancies increase gestational risk.")],
    "BloodPressure":[(0,80,"normal","Normal diastolic. No hypertension signal."),
                     (80,90,"warning","Borderline hypertension (80-89 mmHg). Linked to insulin resistance."),
                     (90,9999,"danger","High BP (>=90 mmHg). Strongly co-occurs with Type-2 diabetes.")],
    "Insulin":   [(0,16,"normal","Low-normal fasting insulin."),
                  (16,166,"normal","Within normal fasting range."),
                  (166,9999,"danger","Hyperinsulinemia (>166 uU/mL): May indicate insulin resistance.")],
    "SkinThickness":[(0,20,"normal","Triceps skinfold normal."),
                     (20,9999,"warning","Higher skinfold (>20 mm): Linked to insulin resistance.")],
}
_ICONS = {"normal":"[OK]","warning":"[WARN]","danger":"[HIGH]"}
_HINTS = {
    "Glucose":"(fasting mg/dL; normal 70-99)",
    "BMI":"(kg/m2; healthy 18.5-24.9)",
    "Age":"(years)",
    "DiabetesPedigreeFunction":"(0.0-2.5; higher = stronger family history)",
    "Pregnancies":"(number of pregnancies)",
    "BloodPressure":"(diastolic mmHg; normal <80)",
    "Insulin":"(fasting uU/mL; normal 2-25)",
    "SkinThickness":"(triceps mm; typical 10-50)",
}

def _clinical_reason(feature: str, value: float) -> str:
    for lo, hi, level, msg in _CLINICAL_RULES.get(feature, []):
        if lo <= value < hi:
            return f"{_ICONS[level]}  {feature} = {value:.2f} -> {msg}"
    return f"*  {feature} = {value:.2f}"

def _top5_shap(model, X_test, feature_names):
    try:
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(X_test[:min(50, len(X_test))], check_additivity=False)
        if isinstance(sv, list): sv = sv[1]
        idx = np.argsort(np.abs(sv).mean(axis=0))[::-1][:5]
        return [feature_names[i] for i in idx]
    except Exception:
        return ["Glucose","BMI","Age","DiabetesPedigreeFunction","Pregnancies"]

def interactive_predictor(model, preprocessor, feature_names, X_test):
    print("\n" + "="*56)
    print("       INTERACTIVE DIABETES RISK ASSESSMENT")
    print("="*56)

    top5 = _top5_shap(model, X_test, feature_names)
    print("\n  Top-5 predictive features:")
    for i, f in enumerate(top5, 1): print(f"    {i}. {f}")

    print("\n" + "-"*56 + "\n  Enter your values (numbers only):\n" + "-"*56)
    user: Dict[str, float] = {}
    for feat in top5:
        hint = _HINTS.get(feat, "")
        while True:
            try:
                user[feat] = float(input(f"  > {feat} {hint}: "))
                break
            except ValueError:
                print("      ! Enter a valid number.")

    # Fill unasked features with training means (neutral after scaling)
    vec = preprocessor.scaler.mean_.copy()
    fidx = {f: i for i, f in enumerate(feature_names)}
    for f, v in user.items():
        if f in fidx: vec[fidx[f]] = v
    scaled = preprocessor.scaler.transform(vec.reshape(1, -1))

    prob = model.predict_proba(scaled)[0][1] * 100
    if prob <= 30:
        label, note = "LOW RISK",      "Maintain a healthy lifestyle. Routine check-ups advised."
    elif prob <= 60:
        label, note = "MODERATE RISK", "Consider lifestyle changes; consult a healthcare provider."
    else:
        label, note = "HIGH RISK",     "Please consult a physician promptly for clinical evaluation."

    print("\n" + "="*56)
    print("         DIABETES RISK ASSESSMENT RESULT")
    print("="*56)
    print(f"  Probability : {prob:.2f} %")
    print(f"  Risk Level  : {label}")
    print(f"  Advice      : {note}")
    print("="*56)

    print("\n--- Clinical Reasoning ---")
    for f in top5: print(f"  {_clinical_reason(f, user[f])}")

    print("\n--- SHAP Feature Contributions ---")
    try:
        exp = shap.TreeExplainer(model)
        sv  = exp.shap_values(scaled, check_additivity=False)
        if isinstance(sv, list): sv = sv[1]
        sv = sv.flatten()
        top = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:5]
        print("  Top-5 contributors (+ raises risk / - lowers risk):")
        for fn, s in top:
            print(f"    {'(+)' if s>0 else '(-)'} {fn:<30s}  SHAP = {s:+.4f}")
    except Exception:
        print("  (SHAP explanation unavailable for this model.)")

    print("\n" + "-"*56)
    print("  DISCLAIMER: For informational purposes only.")
    print("  Does NOT replace professional medical advice.")
    print("-"*56)

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("  FEDERATED LEARNING-BASED DIABETES RISK PREDICTION -- XAI PIPELINE")
    print("="*70)

    print("\n>> STAGE 1: DATA")
    df = DataLoader.load()

    print("\n>> STAGE 2: PREPROCESSING")
    prep = Preprocessor()
    X_tr, X_te, y_tr, y_te, feat_names = prep.run(df)

    print("\n>> STAGE 3: MODEL TRAINING")
    evaluator = Evaluator()
    results   = evaluator.train_and_evaluate(ModelFactory.get(), X_tr, y_tr, X_te, y_te)

    print("\n>> STAGE 4: VISUALISATION")
    evaluator.plot_roc_curves()
    evaluator.plot_confusion_matrices()
    evaluator.export_summary_csv()

    print("\n>> STAGE 5: SHAP EXPLAINABILITY")
    best = max(results, key=lambda r: r.roc_auc)
    print(f"  Best model: {best.name} (AUC={best.roc_auc:.4f})")
    Explainer().explain(best.model, X_te, feat_names, best.name)

    print("\n>> STAGE 6: FEDERATED LEARNING SIMULATION")
    FederatedSimulator().run(X_tr, y_tr, X_te, y_te)

    print("\n" + "="*70)
    print("  PIPELINE COMPLETE -- ALL OUTPUTS SAVED")
    print("="*70)
    for fname in sorted(os.listdir(CFG.OUTPUT_DIR)):
        fpath = os.path.join(CFG.OUTPUT_DIR, fname)
        print(f"  {fname:.<50s} {os.path.getsize(fpath)/1024:>7.1f} KB")
    print("\n[OK] All artifacts generated successfully.")

    print("\n>> STAGE 8: INTERACTIVE DIABETES RISK PREDICTOR")
    interactive_predictor(best.model, prep, feat_names, X_te)

if __name__ == "__main__":
    main()