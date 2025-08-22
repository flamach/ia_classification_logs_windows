import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_evaluate(features_path='eventlog_preprocessed.csv', labels_path='labels.npy'):
    logger.info("Chargement des données prétraitées...")
    X = pd.read_csv(features_path)
    y = np.load(labels_path)
    label_classes = np.load('label_classes.npy', allow_pickle=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=4,
        gamma=1.5,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    logger.info("Début de la validation croisée...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        scores['precision'].append(precision_score(y_fold_val, y_pred, average='weighted'))
        scores['recall'].append(recall_score(y_fold_val, y_pred, average='weighted'))
        scores['f1'].append(f1_score(y_fold_val, y_pred, average='weighted'))
        logger.info(f"Fold {fold}/5 terminé")

    mean_scores = {metric: np.mean(values) for metric, values in scores.items()}
    std_scores = {metric: np.std(values) for metric, values in scores.items()}

    metadata = {
        "model_type": "XGBoost with balanced sampling",
        "metrics": {
            metric: {
                "mean": float(mean_scores[metric]),
                "std": float(std_scores[metric])
            } for metric in scores
        },
        "features": list(X.columns),
        "date_trained": datetime.now().isoformat()
    }

    logger.info("Évaluation finale sur l'ensemble de test...")
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_scores = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, average='weighted'),
        'recall': recall_score(y_test, y_test_pred, average='weighted'),
        'f1': f1_score(y_test, y_test_pred, average='weighted')
    }
    metadata['test_scores'] = test_scores

    logger.info("Métriques globales sur le test :")
    for metric, score in test_scores.items():
        logger.info(f"{metric.capitalize():<10}: {score:.2%}")

    logger.info("Matrice de confusion :")
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(cm)

    logger.info("Rapport détaillé :")
    logger.info(classification_report(y_test, y_test_pred, target_names=label_classes))

    model.save_model("xgboost_model_balanced.model")
    with open("xgboost_model_balanced_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Modèle et métadonnées sauvegardés.")

    return model, metadata

if __name__ == "__main__":
    train_and_evaluate()
