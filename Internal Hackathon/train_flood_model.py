"""
train_flood_model.py
Universal, robust Machine Learning pipeline for Urban Flood Impact Classification.
Follows ML Best Practices:
- Auto-detection of target and feature columns (numeric vs categorical)
- Handles custom uploaded CSV or Excel datasets automatically
- Strict featurization ordering: Stratified Train/Test split BEFORE preprocessing
- Benchmarking multiple classifiers: Logistic Regression (baseline), Random Forest, HistGradientBoosting
- Comprehensive evaluation: Accuracy, Balanced Accuracy, Precision, Recall, Macro/Weighted F1, ROC-AUC
- Confusion matrix, feature importance, and threshold analysis plots
- Checkpoint persistence (joblib pipeline + metadata)
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg") # Non-interactive headless backend for server environments
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score


def detect_target_and_features(df: pd.DataFrame, target_col: str = None):
    """
    Auto-detects the target classification column and splits features into numeric and categorical.
    """
    if target_col and target_col in df.columns:
        target = target_col
    else:
        # Search candidate names
        candidates = [
            "flood_impact_level", "flood_impact", "impact_level", "flood_level", 
            "risk_level", "impact_category", "flood_risk", "inundation_level", 
            "target", "label", "class", "status"
        ]
        matched = [c for c in df.columns if c.lower() in candidates]
        if matched:
            target = matched[0]
        else:
            # Fallback to the last column
            target = df.columns[-1]

    print(f">> Identified Target Column: '{target}'")

    # Exclude ID or name columns that shouldn't be trained on
    id_like = [c for c in df.columns if any(k in c.lower() for k in ["id", "name", "date", "time", "timestamp", "label"]) and c != target]
    feature_candidates = [c for c in df.columns if c != target and c not in id_like]

    numeric_features = [c for c in feature_candidates if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in feature_candidates if c not in numeric_features]

    print(f">> Detected {len(numeric_features)} Numeric Features: {numeric_features}")
    print(f">> Detected {len(categorical_features)} Categorical Features: {categorical_features}")

    return target, numeric_features, categorical_features


def plot_confusion_matrix(cm, class_names, save_path):
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Urban Flood Impact - Confusion Matrix",
        ylabel="True Flood Level",
        xlabel="Predicted Flood Level"
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_feature_importance(model, feature_names, save_path):
    if not HAS_MATPLOTLIB:
        return
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        # Limit to top 15 features if large
        indices = indices[-15:]

        plt.figure(figsize=(10, 6))
        plt.title("Key Predictive Drivers of Flood Impact (Random Forest)")
        plt.barh(range(len(indices)), importances[indices], color="#0284c7", align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Gini Feature Importance")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Feature importance plot saved to: {save_path}")


def plot_threshold_sensitivity(pipeline, X_train, numeric_features, save_path):
    """Generates sensitivity response curve across varying rainfall values."""
    rain_cols = [c for c in numeric_features if "rain" in c.lower()]
    if not rain_cols:
        return
    rain_col = rain_cols[0]

    # Synthesize sensitivity spectrum using median values
    median_row = X_train.median(numeric_only=True).to_dict()
    for cat_col in X_train.select_dtypes(include=["object", "category"]).columns:
        median_row[cat_col] = X_train[cat_col].mode()[0]

    rain_values = np.linspace(5, 300, 60)
    sim_rows = []
    for r in rain_values:
        row = median_row.copy()
        row[rain_col] = r
        sim_rows.append(row)
    sim_df = pd.DataFrame(sim_rows)

    try:
        probs = pipeline.predict_proba(sim_df)
        n_classes = probs.shape[1]

        plt.figure(figsize=(9, 5))
        for i in range(n_classes):
            plt.plot(rain_values, probs[:, i], linewidth=2.2, label=f"Class {i} Probability")
        plt.title(f"Flood Risk Probability vs. {rain_col} (Threshold Sensitivity)")
        plt.xlabel(f"{rain_col}")
        plt.ylabel("Model Predicted Probability")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Threshold sensitivity analysis saved to: {save_path}")
    except Exception as e:
        print(f"Skipping threshold plot: {e}")


def train_ml_pipeline(data_path: str, target_col: str = None, output_dir: str = "output_models"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print(" URBAN FLOOD IMPACT CLASSIFICATION: ML ALGORITHM TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load dataset (CSV or Excel)
    if data_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded dataset from '{data_path}': {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Target and Feature Identification
    target, num_features, cat_features = detect_target_and_features(df, target_col)

    # Clean target
    df = df.dropna(subset=[target])
    y_raw = df[target]

    # Encode target labels if string
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = [str(c) for c in label_encoder.classes_]
    print(f"Target classes ({len(class_names)}): {class_names}")

    X = df[num_features + cat_features]

    # 3. Stratified Train/Test Split (ML Best Practice: featurize AFTER splitting)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nData split: {len(X_train)} Training samples, {len(X_test)} Testing samples")

    # 4. Robust Imputation & Encoding Preprocessor
    transformers = []
    if num_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, num_features))

    if cat_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, cat_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    preprocessor.fit(X_train)

    # Extract transformed feature names
    all_feature_names = []
    if num_features:
        all_feature_names.extend(num_features)
    if cat_features:
        cat_enc = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        all_feature_names.extend(list(cat_enc.get_feature_names_out(cat_features)))

    # 5. Multi-Model Benchmark
    classifiers = {
        "Logistic_Regression_Baseline": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest_Classifier": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        "Hist_Gradient_Boosting": HistGradientBoostingClassifier(max_iter=150, max_depth=8, random_state=42)
    }

    benchmark_results = {}
    best_name = None
    best_score = -1.0
    best_pipe = None

    print("\n--- Model Benchmark Performance (Hold-out Test Set) ---")
    for name, clf in classifiers.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        try:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr") if y_prob is not None else 0.0
        except Exception:
            auc = 0.0

        benchmark_results[name] = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "roc_auc": round(float(auc), 4)
        }

        print(f"[{name}] Acc: {acc:.4f} | Bal Acc: {bal_acc:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f} | ROC-AUC: {auc:.4f}")

        if weighted_f1 > best_score:
            best_score = weighted_f1
            best_name = name
            best_pipe = pipe

    print(f"\n>> BEST PERFORMING MODEL: {best_name} (Weighted F1: {best_score:.4f})")

    # 6. Detailed Evaluation of Best Model
    y_pred_best = best_pipe.predict(X_test)
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred_best, target_names=class_names, digits=4, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    # Feature Importance
    rf_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", classifiers["Random_Forest_Classifier"])])
    rf_pipe.fit(X_train, y_train)
    fi_path = os.path.join(output_dir, "feature_importance.png")
    plot_feature_importance(classifiers["Random_Forest_Classifier"], all_feature_names, fi_path)

    # Threshold Sensitivity Analysis
    thresh_path = os.path.join(output_dir, "threshold_analysis.png")
    plot_threshold_sensitivity(best_pipe, X_train, num_features, thresh_path)

    # 7. Persist Model and Metadata
    model_path = os.path.join(output_dir, "flood_classifier_pipeline.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"\nSaved production pipeline artifact to: {model_path}")

    meta = {
        "best_model_name": best_name,
        "class_names": class_names,
        "numeric_features": num_features,
        "categorical_features": cat_features,
        "all_feature_names": all_feature_names,
        "benchmark_results": benchmark_results,
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model metadata to: {meta_path}")

    return best_pipe, meta


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Urban Flood Impact ML Classifier.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to input CSV or Excel file")
    parser.add_argument("--target-col", type=str, default=None, help="Name of the classification target column")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save models and plots")

    args = parser.parse_args()

    data_path = args.data_path
    if not data_path:
        data_path = os.path.join(BASE_DIR, "urban_flood_dataset.csv")
    elif not os.path.exists(data_path) and os.path.exists(os.path.join(BASE_DIR, data_path)):
        data_path = os.path.join(BASE_DIR, data_path)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(BASE_DIR, "output_models")
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(BASE_DIR, output_dir)

    train_ml_pipeline(data_path, args.target_col, output_dir)
