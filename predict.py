import pickle
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path='model.pkl', vectorizer_path='vectorizer.pkl', encoder_path='label_encoder.pkl'):
    """Charge le modèle et ses composants."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            le_source = pickle.load(f)
        return model, vectorizer, le_source
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise

def prepare_features(log_data, le_source):
    """Prépare les caractéristiques pour la prédiction."""
    try:
        # Conversion de la date
        log_data['TimeGenerated'] = pd.to_datetime(log_data['TimeGenerated'])
        log_data['Hour'] = log_data['TimeGenerated'].dt.hour
        log_data['DayOfWeek'] = log_data['TimeGenerated'].dt.dayofweek
        
        # Encodage de la source
        log_data['Source_encoded'] = le_source.transform(log_data['Source'])
        
        # Extraction des caractéristiques
        X_text = log_data['Message']
        X_numeric = log_data[['Hour', 'DayOfWeek', 'Source_encoded']]
        
        return X_text, X_numeric
        
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des données: {str(e)}")
        raise

def predict_log(log_data, model, vectorizer, le_source):
    """Prédit le type d'entrée pour un log."""
    try:
        # Préparation des caractéristiques
        X_text, X_numeric = prepare_features(log_data, le_source)
        
        # Vectorisation du texte
        X_text_vectorized = vectorizer.transform(X_text)
        
        # Combinaison des caractéristiques
        X = np.hstack((X_text_vectorized.toarray(), X_numeric))
        
        # Prédiction
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        
        return prediction, probability
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise

def main():
    try:
        # Chargement du modèle
        model, vectorizer, le_source = load_model()
        
        # Chargement des nouveaux logs
        logger.info("Chargement des nouveaux logs...")
        df = pd.read_csv('new_logs.csv')
        
        # Prédiction pour chaque log
        predictions, probabilities = predict_log(df, model, vectorizer, le_source)
        
        # Affichage des résultats
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            logger.info(f"Log {idx + 1}:")
            logger.info(f"Type prédit: {pred}")
            logger.info(f"Probabilités: {dict(zip(model.classes_, prob))}\n")
            
    except Exception as e:
        logger.error(f"Une erreur est survenue: {str(e)}")

if __name__ == "__main__":
    main() 