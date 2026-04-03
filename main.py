
"""
SNCF Train Delay Prediction — main.py 
    Run from terminal python main.py --data path/to/sncf_retardsetsans_avec_causes.csv
    Optional :
            you can skip all plots visualisation using --skip-plots
            you can set the directory to save plots --output-dir DIR    
            you can run only some models using --mode (clf for classification, reg for regression & cluster for unsupervised learning)
            you can run specific single model --mode MODE --model MODELNAME --params "YOURPARMATERS" 
    Example running only random forest
    python3 main.py --data "path/to/data.csv" --mode clf --model rf --params "n_estimators=500"

    Examlpe running all classification model
    python3 main.py --data "path/to/data.csv" --mode clf
"""

import argparse
import calendar
import os
import sys
import warnings
import ast

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

warnings.filterwarnings("ignore")


# ─────────────────────
# SETTING UP & CLEANING
# ─────────────────────
CAT_COLS = ["Gare de départ", "Gare d'arrivée", "zone"]
NUM_COLS = ["Annee", "Mois", "Jour", "weekend", "vacance", "tempête", "match"]

def parse_args():
    parser = argparse.ArgumentParser(description="SNCF Prediction Pipeline")
    parser.add_argument("--data", required=True, help="Path to csv")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--mode", choices=["all", "clf", "reg", "cluster"], default="all")
    parser.add_argument("--model", type=str, help="Ex: 'rf', 'lr', 'kmeans', 'ridge'")
    parser.add_argument("--params", type=str, help="Ex: 'n_estimators=100, max_depth=5'")
    return parser.parse_args()

def get_custom_params(param_str):
    if not param_str: return {}
    try:
        pairs = [p.split('=') for p in param_str.split(',')]
        return {k.strip(): ast.literal_eval(v.strip()) for k, v in pairs}
    except Exception as e:
        print(f"Error parsing params: {e}. Using default values.")
        return {}

def load_and_clean(path):
    print(f"\n Loading data")
    df = pd.read_csv(path, low_memory=False).dropna(subset=["retard moyen"])
    df["classe_retard"] = df["retard moyen"].apply(lambda x: 1 if x<5 else 2 if x<15 else 3 if x<30 else 4)
    return df

def get_preprocessor(df):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [c for c in CAT_COLS if c in df.columns]),
        ("num", StandardScaler(), [c for c in NUM_COLS if c in df.columns])
    ])

# ─────────────────────────────
# CLASSIFICATION- Supervisé
# ──────────────────────────
def run_single_clf(df, model_code, custom_p, skip_plots, output_dir):
    print(f"\n Running classification model: {model_code.upper()}")
    X = df[[c for c in CAT_COLS + NUM_COLS if c in df.columns]]
    y = df["classe_retard"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        'rf': (RandomForestClassifier, {"n_estimators": 100, "random_state": 42}),
        'lr': (LogisticRegression, {"max_iter": 1000}),
        'gb': (HistGradientBoostingClassifier, {"max_iter": 100}),
        'knn': (KNeighborsClassifier, {"n_neighbors": 5}),
        'lda': (LinearDiscriminantAnalysis, {"solver": 'svd'})
    }

    if model_code not in models:
        print(f"Unknown model, possible choices: {list(models.keys())}"); return

    clf_class, defaults = models[model_code]
    defaults.update(custom_p)
    model = clf_class(**defaults)

    pipe = Pipeline([("pre", get_preprocessor(df)), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(classification_report(y_test, y_pred))
    if not skip_plots:
        import matplotlib.pyplot as plt
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
        plt.savefig(f"{output_dir}/cm_{model_code}.png")
        plt.show()

# ─────────────────────────
# 2. REGRESSION- Supervised
# ─────────────────────────
def run_single_reg(df, model_code, custom_p, skip_plots, output_dir):
    print(f"\nRunning regression model:: {model_code.upper()}")
    X = df[[c for c in CAT_COLS + NUM_COLS if c in df.columns]]
    y = df["retard moyen"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'lin': (LinearRegression, {}),
        'rf': (RandomForestRegressor, {"n_estimators": 100, "random_state": 42}),
        'ridge': (Ridge, {"alpha": 1.0}),
        'lasso': (Lasso, {"alpha": 0.1})
    }

    if model_code not in models:
        print(f"Unknown model, possible choices: {list(models.keys())}"); return

    reg_class, defaults = models[model_code]
    defaults.update(custom_p)
    model = reg_class(**defaults)

    pipe = Pipeline([("pre", get_preprocessor(df)), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    print(f"R2: {r2_score(y_test, y_pred):.3f}")

# ───────────────────────────
# 3. CLUSTERING- Unsupervised
# ───────────────────────────

def run_single_cluster(df, model_code, custom_p, skip_plots, output_dir):
    print(f"\nRunning regression model: {model_code.upper()}")
    X = df[[c for c in CAT_COLS + NUM_COLS if c in df.columns]]
    
    # Data preprocessing for clustering
    preprocessor = get_preprocessor(df)
    X_processed = preprocessor.fit_transform(X)

    models = {
        'kmeans': (KMeans, {"n_clusters": 4, "random_state": 42, "n_init": 10}),
        'hierarchical': (AgglomerativeClustering, {"n_clusters": 4})
    }

    if model_code not in models:
        print(f"Unknown model, possible choices: {list(models.keys())}"); return

    cluster_class, defaults = models[model_code]
    defaults.update(custom_p)
    model = cluster_class(**defaults)

    clusters = model.fit_predict(X_processed)
    
    if len(np.unique(clusters)) > 1:
        score = silhouette_score(X_processed, clusters, sample_size=2000)
        print(f"Silhouette Score (sample): {score:.3f}")

    if not skip_plots:
        import matplotlib.pyplot as plt
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_processed)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
        plt.title(f"Clustering {model_code.upper()} (PCA 2D)")
        plt.savefig(f"{output_dir}/cluster_{model_code}.png")
        plt.show()

# ───────
# ROUTING
# ───────
def main():
    args = parse_args()
    if args.skip_plots: matplotlib.use("Agg")
    
    sncf = load_and_clean(args.data)
    custom_p = get_custom_params(args.params)

    if args.mode == "clf":
        run_single_clf(sncf, args.model, custom_p, args.skip_plots, args.output_dir)
    elif args.mode == "reg":
        run_single_reg(sncf, args.model, custom_p, args.skip_plots, args.output_dir)
    elif args.mode == "cluster":
        run_single_cluster(sncf, args.model, custom_p, args.skip_plots, args.output_dir)
    else:
        print("Specify a mode --mode (clf, reg, cluster) and a --model.")

if __name__ == "__main__":
    main()
