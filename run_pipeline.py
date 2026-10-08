"""
CardioMine — Full Pipeline Runner
End-to-end: ETL → Classification (RF, LR, J48, NB) → Regression → Clustering

After this script completes, launch the dashboard with:
  streamlit run dashboard/app.py
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def run_etl():
    from etl.extract import extract
    from etl.transform import transform
    from etl.load import save_processed, load as db_load

    logger.info("=" * 55)
    logger.info("STEP 1: ETL Pipeline")
    logger.info("=" * 55)
    raw = extract()
    clean = transform(raw)
    save_processed(clean)
    try:
        db_load(clean)
    except Exception as e:
        logger.warning(f"MySQL load skipped ({e}). Using CSV pipeline.")
    return clean


def run_classification(clean_df):
    from mining.preprocess import prepare_classification_data
    from mining.classification import (
        train_random_forest, train_logistic_regression,
        train_j48, train_naive_bayes,
        evaluate_model, get_feature_importance,
        get_j48_text, get_logistic_coefficients, get_nb_class_stats,
        save_model,
    )

    logger.info("=" * 55)
    logger.info("STEP 2: Classification — RF | LR | J48 | Naive Bayes")
    logger.info("=" * 55)

    X_train, X_test, y_train, y_test, feature_names = prepare_classification_data(clean_df)

    rf  = train_random_forest(X_train, y_train)
    lr  = train_logistic_regression(X_train, y_train)
    j48 = train_j48(X_train, y_train)
    nb  = train_naive_bayes(X_train, y_train)

    all_metrics = {
        "random_forest":       evaluate_model(rf,  X_test, y_test, "Random Forest"),
        "logistic_regression": evaluate_model(lr,  X_test, y_test, "Logistic Regression"),
        "j48":                 evaluate_model(j48, X_test, y_test, "J48 Decision Tree"),
        "naive_bayes":         evaluate_model(nb,  X_test, y_test, "Naive Bayes"),
    }

    fi = get_feature_importance(rf, feature_names)
    fi_j48 = get_feature_importance(j48, feature_names)
    all_metrics["random_forest"]["feature_importance"]    = fi.to_dict("records")
    all_metrics["j48"]["tree_text"]                       = get_j48_text(j48, feature_names)
    all_metrics["j48"]["feature_importance"]              = fi_j48.to_dict("records")
    all_metrics["logistic_regression"]["coefficients"]    = get_logistic_coefficients(lr, feature_names).to_dict("records")
    all_metrics["naive_bayes"]["class_stats"]             = get_nb_class_stats(nb, feature_names)

    save_model(rf, lr, j48, nb, feature_names, all_metrics)

    logger.info("\n── Classifier Summary ──────────────────────────────")
    for name, m in all_metrics.items():
        logger.info(f"  {m['model_name']:25s} Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")


def run_regression(clean_df):
    from mining.regression import train_regression

    logger.info("=" * 55)
    logger.info("STEP 3: Regression — Linear Regression & Ridge")
    logger.info("=" * 55)
    train_regression(clean_df)


def run_clustering(clean_df):
    from mining.preprocess import prepare_clustering_data
    from mining.clustering import (
        elbow_analysis, train_kmeans,
        get_cluster_profiles, save_cluster_model,
    )

    logger.info("=" * 55)
    logger.info("STEP 4: Clustering — K-Means")
    logger.info("=" * 55)

    X_scaled, scaler, feature_names = prepare_clustering_data(clean_df)
    elbow = elbow_analysis(X_scaled)
    km = train_kmeans(X_scaled, n_clusters=3)
    profiles = get_cluster_profiles(clean_df, km.labels_, feature_names)
    save_cluster_model(km, scaler, feature_names, profiles, elbow)


if __name__ == "__main__":
    clean = run_etl()
    run_classification(clean)
    run_regression(clean)
    run_clustering(clean)

    logger.info("\n" + "=" * 55)
    logger.info("✅ Full pipeline complete!")
    logger.info("Launch dashboard with:")
    logger.info("  streamlit run dashboard/app.py")
    logger.info("=" * 55)
