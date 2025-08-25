# Guide de préparation des données

Prérequis
- Python 3.8+ (recommandé 3.11)
- Environnement virtuel avec les dépendances listées dans `requirements.txt`

Étapes rapides
1. Récupérer le fichier source `eventlog.csv` dans le dossier racine.
2. Lancer le script de prétraitement:
   python preprocess_eventlog.py
   - Produit `eventlog_preprocessed.csv`, `labels.npy`, `label_classes.npy`, `vectorizer.pkl` si applicable.
3. Vérifier les sorties et distributions (counts par classe) avant entraînement.

Bonnes pratiques
- Nettoyer les timestamps et normaliser en UTC.
- Gérer les valeurs manquantes: imputer ou marquer explicitement.
- Stocker les jeux volumineux en Parquet et partitionner par date.
- Versionner les transformations (hash du script + date) pour reproductibilité.

Vérifications post‑prétraitement
- Les labels sont bien répartis; exporter un histogramme des classes.
- Les features numériques ont scale raisonnable (standardisation si nécessaire).
- Sauvegarder un échantillon (1%) pour debug rapide.
