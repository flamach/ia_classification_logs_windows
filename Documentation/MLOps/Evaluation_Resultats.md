# Synthèse d'évaluation des résultats

Résumé des métriques (modèle validé)
- Jeu: `eventlog_preprocessed.csv` (11,654 échantillons)
- Accuracy: 0.9125
- F1-weighted: 0.9064

Confusion matrix
- Fichier: `confusion_matrix.csv` (contenu détaillé des vrais/preds)
- Images: `confusion_matrix_raw.png`, `confusion_matrix_normalized.png`

Validations réalisées
- CV stratifié 5 folds (moyenne F1-weighted ≈ 0.9122 ± 0.0087)
- Test final sur split indépendant
- Rapport signé: `validation_report_signed.json`

Recommandations
- Continuer la surveillance de drift via Prometheus metrics exposées.
- Automatiser les ré-entraînements si le F1 pondéré descend de >5%.
- Envisager l'utilisation de Parquet et Dask/Spark pour gros volumes (>10M).

Fichiers pertinents
- `xgboost_model_validated.model` — modèle prêt pour déploiement
- `vectorizer.pkl`, `label_encoder.pkl` — artefacts de prétraitement
- `train_model_xgboost_monitored.py` — script d'entraînement avec metrics push
