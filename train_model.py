import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Charge et prétraite les données des logs."""
    logger.info("Chargement des données...")
    try:
        # Lecture de la première ligne pour voir les colonnes
        first_chunk = pd.read_csv(file_path, nrows=1)
        logger.info(f"Colonnes disponibles dans le fichier : {first_chunk.columns.tolist()}")
        
        # Lecture complète du fichier par chunks
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=10000):
            chunks.append(chunk)
        df = pd.concat(chunks)
        
        # Affichage des informations sur le dataset
        logger.info(f"\nNombre total d'échantillons : {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        raise

def preprocess_data(df):
    """Prétraite les données pour l'entraînement."""
    logger.info("Prétraitement des données...")
    try:
        # Vérification des colonnes requises
        required_columns = ['Message', 'Source', 'TimeGenerated', 'EntryType']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes : {missing_columns}")
        
        # Sélection des colonnes pertinentes
        df_selected = df[required_columns].copy()
        
        # Affichage de la distribution des classes
        logger.info("\nDistribution des types d'entrées :")
        class_dist = df_selected['EntryType'].value_counts()
        logger.info(class_dist)
        logger.info("\nPourcentages :")
        logger.info(class_dist / len(df_selected) * 100)
        
        # Conversion des dates en caractéristiques temporelles
        df_selected['TimeGenerated'] = pd.to_datetime(df_selected['TimeGenerated'])
        df_selected['Hour'] = df_selected['TimeGenerated'].dt.hour
        df_selected['DayOfWeek'] = df_selected['TimeGenerated'].dt.dayofweek
        
        # Encodage des variables catégorielles
        le_source = LabelEncoder()
        df_selected['Source_encoded'] = le_source.fit_transform(df_selected['Source'])
        
        # Affichage des sources uniques
        logger.info("\nSources uniques :")
        logger.info(df_selected['Source'].nunique())
        
        # Création de la matrice de caractéristiques
        X_text = df_selected['Message']  # Messages pour TF-IDF
        X_numeric = df_selected[['Hour', 'DayOfWeek', 'Source_encoded']]
        
        # Target variable
        y = df_selected['EntryType']
        
        # Vérification des valeurs manquantes
        logger.info("\nVérification des valeurs manquantes :")
        logger.info(df_selected.isnull().sum())
        
        return X_text, X_numeric, y, le_source
        
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement: {str(e)}")
        raise

def train_model(X_text, X_numeric, y):
    """Entraîne le modèle de classification."""
    logger.info("Entraînement du modèle...")
    
    # Séparation des données avec stratification
    X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
        X_text, X_numeric, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vectorisation du texte avec moins de features
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_text_train_vectorized = vectorizer.fit_transform(X_text_train)
    X_text_test_vectorized = vectorizer.transform(X_text_test)
    
    # Combinaison des caractéristiques textuelles et numériques
    X_train = np.hstack((X_text_train_vectorized.toarray(), X_numeric_train))
    X_test = np.hstack((X_text_test_vectorized.toarray(), X_numeric_test))
    
    # Création d'un modèle plus simple
    model = RandomForestClassifier(
        n_estimators=50,  # Moins d'arbres
        max_depth=10,     # Profondeur limitée
        min_samples_split=5,
        random_state=42
    )
    
    # Validation croisée
    logger.info("\nValidation croisée (5-fold) :")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    logger.info(f"Scores de validation croisée : {cv_scores}")
    logger.info(f"Score moyen : {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Entraînement final
    model.fit(X_train, y_train)
    
    # Évaluation du modèle 
    y_pred = model.predict(X_test)
    logger.info("\nRapport de classification sur l'ensemble de test:\n")
    logger.info(classification_report(y_test, y_pred))
    
    # Affichage des features les plus importantes
    if hasattr(model, 'feature_importances_'):
        n_text_features = X_text_train_vectorized.shape[1]
        feature_importance = pd.DataFrame({
            'feature': ['text_features'] * n_text_features + ['hour', 'day_of_week', 'source'],
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.groupby('feature')['importance'].sum()
        logger.info("\nImportance des features :")
        logger.info(feature_importance)
    
    return model, vectorizer

def save_model(model, vectorizer, le_source, model_path='model.pkl', vectorizer_path='vectorizer.pkl', encoder_path='label_encoder.pkl'):
    """Sauvegarde le modèle et ses composants."""
    logger.info("Sauvegarde du modèle et des composants...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(le_source, f)

def main():
    try:
        # Chargement des données
        df = load_data('eventlog.csv')
        
        # Prétraitement
        X_text, X_numeric, y, le_source = preprocess_data(df)
        
        # Entraînement
        model, vectorizer = train_model(X_text, X_numeric, y)
        
        # Sauvegarde
        save_model(model, vectorizer, le_source)
        
        logger.info("Entraînement terminé avec succès!")
        
    except Exception as e:
        logger.error(f"Une erreur est survenue: {str(e)}")

if __name__ == "__main__":
    main() 