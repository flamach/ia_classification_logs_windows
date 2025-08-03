# Classification des logs Windows

Ce projet utilise XGBoost pour classifier les logs Windows en diff�rentes cat�gories (Information, Warning, Error, etc.).

## Performance

Le mod�le atteint une pr�cision d''environ 85% sur les donn�es de validation, avec une bonne balance entre pr�cision et rappel pour toutes les classes.

## Features

Le mod�le utilise des features simples mais efficaces :
- Longueur du message
- Nombre de mots
- Source du log

## Installation

1. Cr�er un environnement virtuel Python :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

2. Installer les d�pendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Pour entra�ner et �valuer le mod�le :
```bash
python test_model_xgboost_balanced.py
```

Le script :
1. Charge et pr�traite les donn�es
2. Effectue une validation crois�e 5-fold
3. Sauvegarde le mod�le final et les m�triques de performance

## Structure du projet

- `test_model_xgboost_balanced.py` : Script principal contenant le code d''entra�nement et d''�valuation
- `eventlog.csv` : Donn�es des logs (non inclus dans le repo)
- `xgboost_model_balanced.model` : Mod�le entra�n�
- `xgboost_model_balanced_metadata.json` : M�tadonn�es et m�triques de performance du mod�le
