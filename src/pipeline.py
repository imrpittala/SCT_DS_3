"""
Decision Tree Classifier Pipeline (Bank Marketing Term Deposit Prediction)
===========================================================================
Author: Raghavendra (imrpittala)
Project: SCT_DS_3 - SkillCraft Technology Data Science Internship
Objective: Build, prune, evaluate, and interpret a Decision Tree Classifier
           to predict customer term deposit subscriptions with hyperparameter
           tuning, ROC-AUC analysis, and visual tree extraction.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def set_visual_style():
    """Sets consistent publication-ready figure styling."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "figure.titlesize": 16,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def load_dataset(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    """Loads dataset and splits into feature matrix X and binary target vector y."""
    logger.info(f"Loading Bank Marketing dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} customer records with {df.shape[1]} columns.")
    
    # Binary conversion for target 'y' ('yes' -> 1, 'no' -> 0)
    y = df["y"].map({"yes": 1, "no": 0}).astype(int)
    X = df.drop(columns=["y"])
    
    logger.info(f"Target distribution: {dict(y.value_counts())}")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Performs stratified train-test split before any preprocessing
    to strictly prevent data leakage.
    """
    logger.info(f"Performing stratified split (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list, list]:
    """Constructs column transformer with OneHotEncoder for categoricals and StandardScaler for numericals."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    logger.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    logger.info(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor, numerical_cols, categorical_cols


def train_pruned_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer):
    """
    Fits and tunes a regularized DecisionTreeClassifier via grid search
    over max_depth and min_samples_split to prevent overfitting.
    """
    logger.info("Setting up Decision Tree training pipeline with hyperparameter pruning...")
    
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(criterion="gini", random_state=42, class_weight="balanced"))
    ])

    param_grid = {
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__min_samples_split": [10, 20, 50],
        "classifier__min_samples_leaf": [5, 10, 20]
    }

    grid_search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    logger.info(f"Optimal hyperparameters: {grid_search.best_params_}")
    logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def extract_feature_names(preprocessor: ColumnTransformer, numerical_cols: list, categorical_cols: list) -> list:
    """Extracts post-transformation feature names for tree visualization and importance analysis."""
    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    return numerical_cols + cat_feature_names


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes comprehensive evaluation metrics on held-out test data."""
    logger.info("Evaluating pruned Decision Tree on held-out test data...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    conf_mat = confusion_matrix(y_test, y_pred)
    cls_rep = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "average_precision_pr_auc": round(float(avg_precision), 4),
        "confusion_matrix": conf_mat.tolist(),
        "classification_report": cls_rep
    }

    logger.info(f"Model Metrics: Accuracy={acc:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")
    return metrics


def export_performance_visuals(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list, output_image_path: str):
    """
    Renders 4-panel executive performance dashboard:
    1. Confusion Matrix
    2. ROC Curve with AUC Score
    3. Precision-Recall Curve
    4. Top 12 Feature Importances
    """
    logger.info("Generating model performance visualization dashboard...")
    set_visual_style()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    fig.suptitle(
        "Bank Marketing Term Deposit Decision Tree Performance Dashboard\n"
        "SkillCraft Technology Data Science Internship | Task 3 (SCT_DS_3)",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0, 0],
        xticklabels=["No Deposit (0)", "Subscribed (1)"],
        yticklabels=["No Deposit (0)", "Subscribed (1)"]
    )
    axes[0, 0].set_title("Test Confusion Matrix", fontweight="bold")
    axes[0, 0].set_xlabel("Predicted Class")
    axes[0, 0].set_ylabel("Actual Class")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, color="#0d47a1", lw=2.5, label=f"Decision Tree (AUC = {auc_val:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], color="#9e9e9e", lw=1.8, linestyle="--", label="Random Classifier (AUC = 0.500)")
    axes[0, 1].set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
    axes[0, 1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0, 1].set_ylabel("True Positive Rate (Recall)")
    axes[0, 1].legend(loc="lower right", frameon=True)

    # 3. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    axes[1, 0].plot(rec, prec, color="#2e7d32", lw=2.5, label=f"PR Curve (AP = {pr_auc:.3f})")
    axes[1, 0].set_title("Precision-Recall (PR) Curve", fontweight="bold")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend(loc="upper right", frameon=True)

    # 4. Top Feature Importances
    dt_classifier = model.named_steps["classifier"]
    importances = dt_classifier.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=False).head(12)

    sns.barplot(
        data=feat_df,
        x="Importance",
        y="Feature",
        palette="viridis",
        ax=axes[1, 1],
        edgecolor="black"
    )
    axes[1, 1].set_title("Top 12 Predictive Features (Gini Importance)", fontweight="bold")
    axes[1, 1].set_xlabel("Relative Importance")

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Performance report saved to: {output_image_path}")


def export_tree_diagram(model: Pipeline, feature_names: list, output_tree_path: str):
    """Exports high-resolution visual decision tree structure."""
    logger.info("Exporting interpretable decision tree diagram...")
    dt_classifier = model.named_steps["classifier"]

    plt.figure(figsize=(26, 12))
    plot_tree(
        dt_classifier,
        feature_names=feature_names,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=3,  # Top levels for visual legibility
        precision=2
    )
    plt.title("Pruned Decision Tree Structural Diagram (Top 3 Levels)", fontsize=18, fontweight="bold", pad=20)

    os.makedirs(os.path.dirname(output_tree_path), exist_ok=True)
    plt.savefig(output_tree_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Tree diagram exported to: {output_tree_path}")


def main():
    """Main pipeline execution function for Decision Tree Classifier."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "dataset.csv")
    report_img_path = os.path.join(base_dir, "reports", "visual_insights.png")
    tree_img_path = os.path.join(base_dir, "reports", "decision_tree_structure.png")
    metrics_json_path = os.path.join(base_dir, "reports", "model_evaluation.json")

    logger.info("Starting SCT_DS_3 Decision Tree Classifier Pipeline...")
    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    best_model = train_pruned_model(X_train, y_train, preprocessor)

    # Feature names
    feature_names = extract_feature_names(
        best_model.named_steps["preprocessor"], num_cols, cat_cols
    )

    # Evaluation & Export
    metrics = evaluate_model(best_model, X_test, y_test)
    export_performance_visuals(best_model, X_test, y_test, feature_names, report_img_path)
    export_tree_diagram(best_model, feature_names, tree_img_path)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to: {metrics_json_path}")
    logger.info("SCT_DS_3 pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
