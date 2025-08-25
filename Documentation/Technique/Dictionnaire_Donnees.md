# Dictionnaire des données

Fichier source principal: `eventlog.csv` (voir `eventlog_preprocessed.csv` pour la version transformée)

Colonnes usuelles
- case_id: identifiant unique du cas/process (string)
- activity: nom de l'activité exécutée (string)
- timestamp: horodatage de l'événement (ISO 8601)
- resource: ressource ou utilisateur ayant exécuté l'étape (string)
- attributes: JSON encodé ou colonnes supplémentaires décrivant le contexte (optionnel)
- label: étiquette cible (si disponible) utilisée pour l'entraînement

Fichiers complémentaires
- `eventlog_preprocessed.csv`: output du script `preprocess_eventlog.py`. Contient features numériques prêtes pour le modèle.
- `labels.npy`, `label_classes.npy`, `label_encoder.pkl`: artefacts de prétraitement pour ré-associer les labels et encodages.
- `vectorizer.pkl`: si features textuelles ont été vectorisées.

Remarques
- Les horodatages doivent être en timezone UTC de préférence.
- Préférer Parquet pour gros volumes afin de réduire I/O et stockage.
