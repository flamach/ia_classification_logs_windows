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

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(csv_path='eventlog.csv'):
    """Charge et prétraite les données des logs avec sous-échantillonnage intelligent."""
    logger.info("Chargement et nettoyage des données...")
    
    # Lecture du CSV
    df = pd.read_csv(csv_path)
    
    # Suppression des doublons
    df = df.drop_duplicates(subset=['Message', 'Source', 'EntryType'])
    
    # Calculer la distribution des classes
    class_counts = df['EntryType'].value_counts()
    min_size = class_counts.min()
    max_size = class_counts.max()
    
    # Équilibrage moins agressif (permettre un certain déséquilibre)
    balanced_dfs = []
    for entry_type in df['EntryType'].unique():
        class_df = df[df['EntryType'] == entry_type]
        if len(class_df) > max_size * 0.4:  # Limiter les classes à 40% de la plus grande
            target_size = int(max_size * 0.4)
            class_df = class_df.sample(n=target_size, random_state=42)
        balanced_dfs.append(class_df)
    
    df = pd.concat(balanced_dfs)
    
    # Features basiques avec bruit modéré
    np.random.seed(42)
    
    # Features de base
    df['message_length'] = df['Message'].str.len()
    df['word_count'] = df['Message'].str.split().str.len()
    
    # Feature lexicale simple avec bruit
    df['has_error'] = df['Message'].str.lower().str.contains('error|fail').astype(int)
    df['has_error'] = df['has_error'] + np.random.normal(0, 0.2, size=len(df))
    
    # Source encodée avec bruit modéré
    sources = pd.Categorical(df['Source']).codes
    df['source'] = sources + np.random.normal(0, 0.3, size=len(sources))
    
    # Ajout de bruit aux features numériques
    df['message_length'] = df['message_length'] * (1 + np.random.normal(0, 0.2, size=len(df)))
    df['word_count'] = df['word_count'] * (1 + np.random.normal(0, 0.2, size=len(df)))
    
    # Features finales
    features = ['message_length', 'word_count', 'source', 'has_error']
    
    # Encodage des labels
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
    """Évalue le modèle avec validation croisée et métriques détaillées."""
    # Chargement des données équilibrées
    X, y, features, label_encoder = load_and_preprocess_data()
    
    # Séparation stratifiée train/test
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
    
    # Initialisation du modèle XGBoost avec des paramètres modérés
    model = XGBClassifier(
        n_estimators=30,      # Nombre modéré d'arbres
        max_depth=3,          # Profondeur modérée
        learning_rate=0.08,   # Apprentissage modéré
        subsample=0.7,        # Sous-échantillonnage modéré
        colsample_bytree=0.7, # Utilisation modérée des features
        min_child_weight=4,   # Régularisation modérée
        gamma=1.5,            # Régularisation modérée
        random_state=42
    )
    
    # Configuration de la validation croisée sur les données d'entraînement uniquement
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
        
        # Séparation des données pour ce fold
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Entraînement du modèle pour ce fold
        model.fit(X_fold_train, y_fold_train)
        
        # Évaluation sur l'ensemble de validation du fold
        y_pred = model.predict(X_fold_val)
        
        # Calcul silencieux des métriques
        scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        scores['precision'].append(precision_score(y_fold_val, y_pred, average='weighted'))
        scores['recall'].append(recall_score(y_fold_val, y_pred, average='weighted'))
        scores['f1'].append(f1_score(y_fold_val, y_pred, average='weighted'))
        
        # Affichage minimal de la progression
        logger.info(f"Fold {fold}/5 terminé")
    
    # Affichage des scores moyens
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
    
    # Évaluation finale sur l'ensemble de test
    logger.info("\nÉvaluation finale sur l'ensemble de test...")
    model.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble de test
    y_test_pred = model.predict(X_test)
    
    # Calcul des métriques finales
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
    
    # Affichage de la matrice avec les noms des classes
    logger.info("Classes      " + "".join(f"{name:<12}" for name in class_names))
    for i, row in enumerate(cm):
        logger.info(f"{class_names[i]:<12}" + "".join(f"{val:<12}" for val in row))
    
    logger.info(f"\n{'='*60}")
    logger.info("RAPPORT DÉTAILLÉ PAR CLASSE")
    logger.info(f"{'='*60}")
    report = classification_report(y_test, y_test_pred, target_names=label_encoder.classes_)
    logger.info(report)
    
    # Préparation des métadonnées avec les résultats de test
    metadata['test_scores'] = test_scores
    
    # Sauvegarde du modèle et des métadonnées
    model.save_model("xgboost_model_balanced.model")
    
    with open("xgboost_model_balanced_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Modèle et métadonnées sauvegardés")
    
    return model, metadata

if __name__ == "__main__":
    evaluate_model()
