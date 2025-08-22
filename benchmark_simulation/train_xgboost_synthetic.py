import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(csv_path='c:/Users/macha/OneDrive/Bureau/IA_logs/synthetic_eventlog.csv'):
    logger.info("Chargement et nettoyage des données...")
    df = pd.read_csv(csv_path)
    # Utiliser tout le dataset sans sous-échantillonnage ni équilibrage
    np.random.seed(42)
    df['message_length'] = df['Message'].str.len()
    df['word_count'] = df['Message'].str.split().str.len()
    df['has_error'] = df['Message'].str.lower().str.contains('error|fail').astype(int)
    df['has_error'] = df['has_error'] + np.random.normal(0, 0.2, size=len(df))
    sources = pd.Categorical(df['Source']).codes
    df['source'] = sources + np.random.normal(0, 0.3, size=len(sources))
    df['message_length'] = df['message_length'] * (1 + np.random.normal(0, 0.2, size=len(df)))
    df['word_count'] = df['word_count'] * (1 + np.random.normal(0, 0.2, size=len(df)))
    features = ['message_length', 'word_count', 'source', 'has_error']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['EntryType'])
    class_dist = df['EntryType'].value_counts()
    logger.info(f"\n{'='*60}")
    logger.info("DISTRIBUTION DES CLASSES APRÈS ÉQUILIBRAGE")
    logger.info(f"{'='*60}")
    for class_name, count in class_dist.items():
        logger.info(f"{class_name:<12}: {count:>5} exemples ({count/len(df):>6.1%})")
    return df[features], y, features, label_encoder

def evaluate_model():
    X, y, features, label_encoder = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    logger.info(f"\n{'='*60}")
    logger.info("DISTRIBUTION DES CLASSES - ENSEMBLE D'ENTRAÎNEMENT")
    logger.info(f"{'='*60}")
    for label in np.unique(y_train):
        class_name = label_encoder.inverse_transform([label])[0]
        count = np.sum(y_train == label)
        logger.info(f"{class_name:<12}: {count:>5} exemples ({count/len(y_train):>6.1%})")
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
    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    logger.info("Début de la validation croisée sur l'ensemble d'entraînement...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        logger.info(f"\nFold {fold}")
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
        "max_samples_per_class": 1000,
        "metrics": {
            metric: {
                "mean": float(mean_scores[metric]),
                "std": float(std_scores[metric])
            } for metric in scores
        },
        "features": features,
        "date_trained": datetime.now().isoformat()
    }
    logger.info("\nÉvaluation finale sur l'ensemble de test...")
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_scores = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, average='weighted'),
        'recall': recall_score(y_test, y_test_pred, average='weighted'),
        'f1': f1_score(y_test, y_test_pred, average='weighted')
    }
    logger.info(f"\n{'#'*80}")
    logger.info(f"{'#'*25} RÉSULTATS FINAUX SUR LE TEST {'#'*25}")
    logger.info(f"{'#'*80}")
    logger.info(f"\n{'='*60}")
    logger.info("MÉTRIQUES GLOBALES")
    logger.info(f"{'='*60}")
    for metric, score in test_scores.items():
        logger.info(f"{metric.capitalize():<10}: {score:.2%}")
    logger.info(f"\n{'='*60}")
    logger.info("MATRICE DE CONFUSION FINALE")
    logger.info(f"{'='*60}")
    cm = confusion_matrix(y_test, y_test_pred)
    class_names = label_encoder.classes_
    logger.info("Classes      " + "".join(f"{name:<12}" for name in class_names))
    for i, row in enumerate(cm):
        logger.info(f"{class_names[i]:<12}" + "".join(f"{val:<12}" for val in row))
    logger.info(f"\n{'='*60}")
    logger.info("RAPPORT DÉTAILLÉ PAR CLASSE")
    logger.info(f"{'='*60}")
    report = classification_report(y_test, y_test_pred, target_names=label_encoder.classes_)
    logger.info(report)
    metadata['test_scores'] = test_scores
    model.save_model("xgboost_model_synthetic.model")
    with open("xgboost_model_synthetic_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Modèle et métadonnées sauvegardés")
    return model, metadata

if __name__ == "__main__":
    import time
    start_time = time.time()
    evaluate_model()
    end_time = time.time()
    print(f"Temps total d'exécution du benchmark : {end_time - start_time:.2f} secondes")
