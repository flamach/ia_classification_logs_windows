# Système de Détection de Logs Dangereux

Ce projet utilise l'apprentissage automatique pour classifier les logs Windows comme étant dangereux ou normaux.

## Prérequis

- Python 3.11.9
- Un environnement virtuel (venv)
- Télécharger le dataset suivant : https://www.kaggle.com/datasets/mehulkatara/windows-event-log
- Extraire le csv et le placer à la racine du projet

## Installation

1. Créez et activez l'environnement virtuel :
```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Installez les dépendances depuis requirements.txt :
```powershell
pip install -r requirements.txt
```

## Structure des fichiers

- `train_model.py` : Script pour entraîner le modèle
- `predict.py` : Script pour prédire si de nouveaux logs sont dangereux
- `eventlog.csv` : Jeu de données d'entraînement 
- `model.pkl` : Modèle entraîné (généré après l'entraînement)
- `vectorizer.pkl` : Vectoriseur de texte (généré après l'entraînement)
- `requirements.txt` : Liste des dépendances Python requises

## Utilisation

1. Entraînement du modèle :
```powershell
python train_model.py
```

2. Prédiction sur de nouveaux logs :
```powershell
python predict.py
```

## Format des données

Les logs doivent être au format CSV. Pour la prédiction, placez vos nouveaux logs dans un fichier `new_logs.csv` avec la même structure que le fichier d'entraînement.

## Notes

- Le modèle utilise Random Forest pour la classification
- Les données textuelles sont vectorisées en utilisant TF-IDF
- Les prédictions incluent une probabilité de danger pour chaque log 