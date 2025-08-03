# Classification des logs Windows

Ce projet utilise XGBoost pour classifier les logs Windows en différentes catégories (Information, Warning, Error, etc.).

## Performance

Le modèle atteint une précision d''environ 85% sur les données de validation, avec une bonne balance entre précision et rappel pour toutes les classes.

## Features

Le modèle utilise des features simples mais efficaces :
- Longueur du message
- Nombre de mots
- Source du log

## Installation

1. Créer un environnement virtuel Python :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Pour entraîner et évaluer le modèle :
```bash
python test_model_xgboost_balanced.py
```

Le script :
1. Charge et prétraite les données
2. Effectue une validation croisée 5-fold
3. Sauvegarde le modèle final et les métriques de performance

## Structure du projet

- `test_model_xgboost_balanced.py` : Script principal contenant le code d''entraînement et d''évaluation
- `eventlog.csv` : Données des logs (non inclus dans le repo)
- `xgboost_model_balanced.model` : Modèle entraîné
- `xgboost_model_balanced_metadata.json` : Métadonnées et métriques de performance du modèle
