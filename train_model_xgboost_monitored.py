"""
Script d'entraînement XGBoost avec monitoring Prometheus intégré
Version instrumentée pour dashboard Grafana
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Import du monitoring Prometheus
from prometheus_monitoring import XGBoostPrometheusMonitor, extract_metrics_from_classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
FEATURES_PATH = Path("eventlog_preprocessed.csv")
LABELS_PATH = Path("labels.npy")
LABEL_CLASSES_PATH = Path("label_classes.npy")
MODEL_PATH = Path("xgboost_model_balanced.model")
METADATA_PATH = Path("xgboost_model_balanced_metadata.json")

# Paramètres modèle optimisés
OPTIMIZED_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'min_child_weight': 2,
    'gamma': 0.3,
    'random_state': 42,
    'n_jobs': -1,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss'
}


def load_data():
    """Charge les données prétraitées"""
    if not FEATURES_PATH.exists() or not LABELS_PATH.exists():
        logger.error("Fichiers prétraités manquants. Exécute preprocess_eventlog.py d'abord.")
        raise SystemExit(1)
    
    logger.info("Chargement des données prétraitées...")
    X = pd.read_csv(FEATURES_PATH)
    y = np.load(LABELS_PATH)
    
    logger.info(f"Données chargées: {X.shape[0]} échantillons, {X.shape[1]} features")
    logger.info(f"Distribution des classes: {np.bincount(y)}")
    
    return X, y


def load_label_names():
    """Charge les noms des classes"""
    if LABEL_CLASSES_PATH.exists():
        label_names = np.load(LABEL_CLASSES_PATH, allow_pickle=True)
        logger.info(f"Classes détectées: {label_names}")
        return label_names
    return None


def train_with_monitoring(monitor: XGBoostPrometheusMonitor, X, y, label_names):
    """
    Entraîne le modèle avec monitoring Prometheus
    """
    logger.info("=== DÉBUT ENTRAÎNEMENT AVEC MONITORING ===")
    
    # Enregistrer début d'entraînement
    monitor.record_training_start(total_rows=len(X))
    
    try:
        # Split train/test stratifié
        logger.info("Division train/test (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Validation croisée stratifiée
        logger.info("Validation croisée (StratifiedKFold, k=5)...")
        cv_start_time = time.time()
        
        model_cv = XGBClassifier(**OPTIMIZED_PARAMS)
        cv_scores = cross_val_score(model_cv, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1_weighted')
        
        cv_duration = time.time() - cv_start_time
        logger.info(f"Validation croisée terminée en {cv_duration:.2f}s")
        logger.info(f"Scores CV F1: {cv_scores}")
        logger.info(f"F1 moyen CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Entraînement final sur tout le train
        logger.info("Entraînement final...")
        final_model = XGBClassifier(**OPTIMIZED_PARAMS)
        final_model.fit(X_train, y_train)
        
        # Prédictions et évaluation
        logger.info("Évaluation sur ensemble test...")
        y_pred = final_model.predict(X_test)
        
        # Extraction des métriques
        metrics = extract_metrics_from_classification_report(y_test, y_pred, label_names)
        
        # Enregistrement des métriques dans Prometheus
        monitor.record_training_metrics(metrics, label_names)
        monitor.record_confusion_matrix(y_test, y_pred, label_names)
        
        # Affichage résultats
        logger.info("=== RÉSULTATS FINAUX ===")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 pondéré: {metrics['f1_weighted']:.4f}")
        logger.info(f"Précision pondérée: {metrics['precision_weighted']:.4f}")
        logger.info(f"Rappel pondéré: {metrics['recall_weighted']:.4f}")
        
        print(classification_report(y_test, y_pred, target_names=label_names))
        
        # Sauvegarde du modèle
        logger.info(f"Sauvegarde du modèle -> {MODEL_PATH}")
        final_model.save_model(str(MODEL_PATH))
        
        # Sauvegarde métadonnées
        metadata = {
            'model_type': 'XGBClassifier',
            'training_date': datetime.now().isoformat(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': X.shape[1],
            'classes': label_names.tolist() if label_names is not None else [],
            'hyperparameters': OPTIMIZED_PARAMS,
            'cv_scores': {
                'f1_scores': cv_scores.tolist(),
                'mean_f1': float(cv_scores.mean()),
                'std_f1': float(cv_scores.std())
            },
            'test_metrics': metrics,
            'confusion_matrix_shape': [len(label_names), len(label_names)] if label_names is not None else [0, 0]
        }
        
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Métadonnées sauvegardées -> {METADATA_PATH}")
        
        # Marquer entraînement comme réussi
        early_stopped = metrics['f1_weighted'] >= 0.85  # Simulation early stopping si objectif atteint
        monitor.record_training_end(success=True, early_stopped=early_stopped)
        
        return final_model, metrics
        
    except Exception as e:
        logger.error(f"Erreur pendant l'entraînement: {e}")
        monitor.record_training_end(success=False)
        raise


def main():
    """Fonction principale avec monitoring intégré"""
    # Initialisation du monitoring Prometheus
    monitor = XGBoostPrometheusMonitor(
        pushgateway_url="localhost:9091",
        job_name="xgboost_training"
    )
    
    try:
        # Chargement des données
        X, y = load_data()
        label_names = load_label_names()
        
        # Entraînement avec monitoring
        model, metrics = train_with_monitoring(monitor, X, y, label_names)
        
        # Push final des métriques
        logger.info("Push des métriques vers Prometheus...")
        monitor.push_metrics()
        
        # Résumé des métriques
        summary = monitor.get_metrics_summary()
        logger.info("=== RÉSUMÉ MONITORING ===")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        
        logger.info("=== ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ===")
        logger.info("Consultez le dashboard Grafana: http://localhost:3000")
        logger.info("Métriques Prometheus: http://localhost:9090")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        monitor.record_training_end(success=False)
        monitor.push_metrics()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
