#!/usr/bin/env python3
"""
tes_full.py
Fully-featured ML pipeline for UCI Bank Marketing dataset.

Features:
 - Load CSV (default ./bank-additional-full.csv or --data PATH)
 - Basic EDA (info, target dist, a few plots)
 - Preprocessing: OneHotEncoder for categoricals, StandardScaler for numeric
 - Uses imbalanced-learn Pipeline: preprocess -> SMOTE -> model
 - Models: LogisticRegression, DecisionTree, RandomForest (and XGBoost if available)
 - GridSearchCV for RandomForest (optional)
 - Metrics: accuracy, precision, recall, f1, ROC AUC, PR AUC
 - Visuals saved: ROC, PR, Confusion Matrices, Feature Importance
 - Optional SHAP explanations (if shap installed)
 - Saves model objects (joblib) and reports (JSON/CSV/PNG)
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# optional imports
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_INSTALLED = True
except Exception:
    IMBLEARN_INSTALLED = False

try:
    import xgboost as xgb
    XGBOOST_INSTALLED = True
except Exception:
    XGBOOST_INSTALLED = False

try:
    import shap
    SHAP_INSTALLED = True
except Exception:
    SHAP_INSTALLED = False

import joblib

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_feature_names_from_column_transformer(ct: ColumnTransformer, input_columns):
    """
    Get output feature names after a ColumnTransformer with sklearn >=1.0
    """
    feature_names = []
    for name, transformer, cols in ct.transformers_:
        if transformer == 'drop' or transformer is None:
            continue
        if name == "remainder":
            continue
        # handle pipelines
        if hasattr(transformer, "get_feature_names_out"):
            try:
                names = transformer.get_feature_names_out(cols)
            except Exception:
                # If transformer is a Pipeline, get last step
                if isinstance(transformer, Pipeline):
                    last = transformer.steps[-1][1]
                    if hasattr(last, "get_feature_names_out"):
                        names = last.get_feature_names_out(cols)
                    else:
                        names = cols if isinstance(cols, (list, tuple, np.ndarray)) else [cols]
                else:
                    names = cols if isinstance(cols, (list, tuple, np.ndarray)) else [cols]
        else:
            names = cols if isinstance(cols, (list, tuple, np.ndarray)) else [cols]
        feature_names.extend(list(names))
    return feature_names

# ---------------------------
# Plot helpers
# ---------------------------
def plot_and_save_roc(y_true, y_proba_dict, out_path):
    plt.figure(figsize=(8,6))
    for name, proba in y_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_and_save_pr(y_true, y_proba_dict, out_path):
    plt.figure(figsize=(8,6))
    for name, proba in y_proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (PR AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_confusion_matrix(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels, title=title, ylabel='True label', xlabel='Predicted label')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(data_path="bank-additional-full.csv",
                 outdir="outputs",
                 test_size=0.2,
                 random_state=42,
                 run_gridsearch=True,
                 use_xgboost=False,
                 use_shap=False):
    ensure_dir(outdir)
    print(f"[INFO] Loading data from: {data_path}")
    df = pd.read_csv(data_path, sep=';')
    print(df.head().to_string(index=False))
    print("\n[INFO] Data info:")
    print(df.info())

    # target distribution
    print("\n[INFO] Target distribution:")
    print(df['y'].value_counts(normalize=True).rename("proportion"))

    # Basic EDA saved figure: target count
    try:
        plt.figure(figsize=(5,4))
        df['y'].value_counts().plot(kind='bar')
        plt.title("Target counts")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "eda_target_counts.png"), dpi=150)
        plt.close()
    except Exception as e:
        print("[WARN] EDA save failed:", e)

    # Prepare X, y
    X = df.drop(columns=["y"])
    y = df["y"].map({"yes":1, "no":0})
    # detect columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n[INFO] Found {len(categorical_cols)} categorical cols and {len(numeric_cols)} numeric cols")

    # Build preprocessor
    # use sparse_output=False for OneHotEncoder to get dense array (compatibility)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()
    preprocessor = ColumnTransformer(transformers=[
        ("cat", ohe, categorical_cols),
        ("num", scaler, numeric_cols)
    ], remainder="drop")

    # Train-test split (on raw dataframe)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {len(X_train_df)}, Test size: {len(X_test_df)}")

    # Build models to evaluate
    models = {}
    models['LogisticRegression'] = LogisticRegression(max_iter=1000, random_state=random_state)
    models['DecisionTree'] = DecisionTreeClassifier(random_state=random_state)
    models['RandomForest'] = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    if use_xgboost and XGBOOST_INSTALLED:
        models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=-1)
    elif use_xgboost:
        print("[WARN] XGBoost requested but not installed. Skipping XGBoost.")

    # We'll use imblearn pipeline: preprocess -> SMOTE -> model
    if IMBLEARN_INSTALLED:
        pipeline_base = ImbPipeline
        print("[INFO] imblearn detected: using ImbPipeline with SMOTE")
    else:
        pipeline_base = Pipeline
        print("[WARN] imblearn not installed: SMOTE will not be applied. Install imbalanced-learn for SMOTE.")

    # Containers for results
    trained_pipelines = {}
    results_summary = {}
    y_proba_for_plots = {}

    # Train each model
    for name, estimator in models.items():
        print(f"\n[INFO] Training model: {name}")
        if IMBLEARN_INSTALLED:
            pipe = pipeline_base(steps=[
                ('preprocess', preprocessor),
                ('smote', SMOTE(random_state=random_state)),
                ('model', estimator)
            ])
            # fit on X_train_df (DataFrame) so ColumnTransformer can use column names
            pipe.fit(X_train_df, y_train)
        else:
            pipe = pipeline_base(steps=[
                ('preprocess', preprocessor),
                ('model', estimator)
            ])
            # If no SMOTE, we need to fit preprocessor+model on training DataFrame directly
            pipe.fit(X_train_df, y_train)

        # Predict on test set
        y_pred = pipe.predict(X_test_df)
        # try predict_proba
        try:
            y_proba = pipe.predict_proba(X_test_df)[:,1]
        except Exception:
            y_proba = None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True)

        cm = confusion_matrix(y_test, y_pred)

        # Save pipeline and metrics
        trained_pipelines[name] = pipe
        results_summary[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "has_proba": y_proba is not None
        }
        if y_proba is not None:
            y_proba_for_plots[name] = y_proba

        print(f"[RESULT] {name} — acc:{acc:.4f} prec:{prec:.4f} rec:{rec:.4f} f1:{f1:.4f}")
        # Save confusion matrix image
        cm_path = os.path.join(outdir, f"cm_{name}.png")
        plot_confusion_matrix(cm, labels=["no","yes"], title=f"Confusion Matrix - {name}", out_path=cm_path)

        # Save classification report text and JSON
        save_json(report, os.path.join(outdir, f"classif_report_{name}.json"))
        with open(os.path.join(outdir, f"classif_report_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, y_pred))

        # Save model object
        model_path = os.path.join(outdir, f"pipeline_{name}.joblib")
        joblib.dump(pipe, model_path)

    # Plot ROC & PR curves if any model supports probability
    if len(y_proba_for_plots) > 0:
        roc_path = os.path.join(outdir, "roc_comparison.png")
        plot_and_save_roc(y_test, y_proba_for_plots, roc_path)
        pr_path = os.path.join(outdir, "pr_comparison.png")
        plot_and_save_pr(y_test, y_proba_for_plots, pr_path)
    else:
        print("[WARN] No model provided probability estimates; skipping ROC/PR plots.")

    # GridSearch for RandomForest (optional)
    if run_gridsearch:
        print("\n[INFO] Running GridSearchCV for RandomForest (this may take time)...")
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5]
        }
        if IMBLEARN_INSTALLED:
            grid_pipe = ImbPipeline(steps=[
                ('preprocess', preprocessor),
                ('smote', SMOTE(random_state=random_state)),
                ('model', RandomForestClassifier(random_state=random_state, n_jobs=-1))
            ])
        else:
            grid_pipe = Pipeline(steps=[
                ('preprocess', preprocessor),
                ('model', RandomForestClassifier(random_state=random_state, n_jobs=-1))
            ])

        grid = GridSearchCV(grid_pipe, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        grid.fit(X_train_df, y_train)
        print("[INFO] GridSearch best params:", grid.best_params_)
        best = grid.best_estimator_
        # Evaluate best
        y_pred = best.predict(X_test_df)
        y_proba = None
        try:
            y_proba = best.predict_proba(X_test_df)[:,1]
        except Exception:
            y_proba = None
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        results_summary['RandomForest_GridSearch'] = {
            "best_params": grid.best_params_,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "has_proba": y_proba is not None
        }
        joblib.dump(best, os.path.join(outdir, "RandomForest_GridSearch.joblib"))
        save_json(report, os.path.join(outdir, "classif_report_RandomForest_GridSearch.json"))
        plot_confusion_matrix(cm, labels=["no","yes"], title="Confusion Matrix - RF GridSearch",
                             out_path=os.path.join(outdir, "cm_RandomForest_GridSearch.png"))
        if y_proba is not None:
            plot_and_save_roc(y_test, {'RandomForest_GridSearch': y_proba}, os.path.join(outdir, "roc_RF_gridsearch.png"))
            plot_and_save_pr(y_test, {'RandomForest_GridSearch': y_proba}, os.path.join(outdir, "pr_RF_gridsearch.png"))

    # Feature importance for RandomForest (best available RF)
    rf_key = None
    if 'RandomForest_GridSearch' in results_summary:
        rf_key = 'RandomForest_GridSearch'
        rf_pipe = joblib.load(os.path.join(outdir, "RandomForest_GridSearch.joblib"))
    elif 'RandomForest' in trained_pipelines:
        rf_key = 'RandomForest'
        rf_pipe = trained_pipelines['RandomForest']
    else:
        rf_pipe = None

    if rf_pipe is not None:
        print("\n[INFO] Extracting feature importances from RandomForest...")
        # get model and preprocessor inside pipeline
        if isinstance(rf_pipe, Pipeline) or (IMBLEARN_INSTALLED and isinstance(rf_pipe, ImbPipeline)):
            # If ImbPipeline: steps may be preprocess->smote->model
            # find preprocess step
            try:
                preproc = rf_pipe.named_steps.get('preprocess', None)
                if preproc is None:
                    # older imblearn pipeline might use index
                    preproc = rf_pipe.steps[0][1]
            except Exception:
                preproc = None
            # find model
            model_obj = rf_pipe.named_steps.get('model', None)
            if model_obj is None:
                # fallback: last step
                model_obj = rf_pipe.steps[-1][1]
        else:
            preproc = None
            model_obj = rf_pipe

        if preproc is not None and hasattr(model_obj, "feature_importances_"):
            # get feature names
            try:
                feat_names = get_feature_names_from_column_transformer(preproc, X.columns)
            except Exception:
                # fallback: try using get_feature_names_out
                try:
                    feat_names = preproc.get_feature_names_out()
                except Exception:
                    feat_names = [f"f_{i}" for i in range(len(model_obj.feature_importances_))]
            importances = model_obj.feature_importances_
            fi_df = pd.DataFrame({"feature": feat_names, "importance": importances})
            fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
            fi_csv = os.path.join(outdir, "feature_importances_randomforest.csv")
            fi_df.to_csv(fi_csv, index=False)
            # plot top 25
            topn = fi_df.head(25).iloc[::-1]
            plt.figure(figsize=(8,8))
            plt.barh(topn['feature'], topn['importance'])
            plt.title("Top 25 Feature Importances (RandomForest)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "feature_importances_randomforest.png"), dpi=150)
            plt.close()
            print(f"[INFO] Feature importance saved to {fi_csv} and PNG.")
        else:
            print("[WARN] Cannot extract feature importances (preprocessor or model missing).")

    # SHAP explanations (optional)
    if use_shap and SHAP_INSTALLED and rf_pipe is not None:
        try:
            print("\n[INFO] Computing SHAP values (may be slow)...")
            # use a small sample from training set to save time
            sample_X = X_train_df.sample(n=min(300, len(X_train_df)), random_state=random_state)
            # get transformed sample (preprocessor)
            preproc_for_shap = None
            if isinstance(rf_pipe, Pipeline) or (IMBLEARN_INSTALLED and isinstance(rf_pipe, ImbPipeline)):
                preproc_for_shap = rf_pipe.named_steps.get('preprocess', None)
            if preproc_for_shap is None:
                print("[WARN] Preprocessor not found for SHAP; skipping.")
            else:
                X_trans = preproc_for_shap.transform(sample_X)
                model_for_shap = None
                if isinstance(rf_pipe, Pipeline) or (IMBLEARN_INSTALLED and isinstance(rf_pipe, ImbPipeline)):
                    model_for_shap = rf_pipe.named_steps.get('model', None)
                    if model_for_shap is None:
                        model_for_shap = rf_pipe.steps[-1][1]
                else:
                    model_for_shap = rf_pipe
                expl = shap.TreeExplainer(model_for_shap)
                shap_vals = expl.shap_values(X_trans)
                # summary plot
                shap_summary_png = os.path.join(outdir, "shap_summary.png")
                shap.summary_plot(shap_vals, X_trans, show=False)
                plt.savefig(shap_summary_png, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[INFO] SHAP summary saved to {shap_summary_png}")
        except Exception as e:
            print("[WARN] SHAP computation failed:", e)
    elif use_shap and not SHAP_INSTALLED:
        print("[WARN] SHAP requested but shap package not installed. Skipping SHAP step.")

    # Save overall summary
    summary_path = os.path.join(outdir, f"summary_{timestamp_str()}.json")
    save_json(results_summary := results_summary if 'results_summary' in locals() else results_summary, summary_path)
    print(f"\n[INFO] All done. Outputs saved to folder: {outdir}")
    print(f"[INFO] Summary saved to: {summary_path}")

# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full ML pipeline for Bank Marketing dataset")
    parser.add_argument("--data", type=str, default="bank-additional-full.csv", help="Path to CSV dataset")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--no_grid", action="store_true", help="Skip GridSearchCV")
    parser.add_argument("--use_xgb", action="store_true", help="Include XGBoost model (if installed)")
    parser.add_argument("--use_shap", action="store_true", help="Compute SHAP explanations (if shap installed)")
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        outdir=args.outdir,
        test_size=args.test_size,
        random_state=42,
        run_gridsearch=not args.no_grid,
        use_xgboost=args.use_xgb,
        use_shap=args.use_shap
    )