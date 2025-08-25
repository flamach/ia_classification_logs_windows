# Plan MLOps pour déploiement on‑premise

Objectif
- Industrialiser l'entraînement, la validation et le déploiement du modèle XGBoost en environnement on‑premise avec traçabilité et monitoring.

Composants clés
- Stockage: Partage de fichiers réseau (NFS) ou objet S3-compatible (MinIO) pour artefacts modèle et jeux de données.
- Orchestration: Cron / Airflow / Prefect pour planifier ré-entraînements et pipelines.
- Monitoring: Prometheus + Pushgateway + Grafana (dashboards fournis).
- Contrôle de modèle: stockage versionné des modèles, métadonnées JSON et rapport signé.
- CI/CD: pipeline GitHub Actions / GitLab CI pour tests et packaging.

Processus simplifié
1. ETL / Prétraitement: `preprocess_eventlog.py` produit dataset nettoyé et features.
2. Entraînement & Validation: `train_model_xgboost.py` / `validation_framework.py` pour CV et test final.
3. Publication: sauvegarder modèle validé + metadata JSON + signature. Copier artefacts vers MinIO/NFS.
4. Déploiement: endpoint batch (cron) ou microservice inference (Flask/FastAPI) utilisant `xgboost.Booster.load_model`.
5. Monitoring & Alerting: exporter métriques via Prometheus pushgateway et configurer alertes Grafana.

Sécurité et conformité
- Chiffrement au repos pour les artefacts sensibles.
- Gestion des accès (RBAC) pour écriture/lecture des artefacts.
- Journalisation des accès et exécutions pour audit.

Rollout phases
- Pilote: exécution manuelle + monitoring pour 2 semaines.
- MVP: automatisation via Cron/Task + alertes.
- Production: orchestration via Airflow/Prefect + CI/CD.

Indicateurs de succès
- Disponibilité du pipeline > 99% (sauf maintenance)
- Régression F1 ≤ 2% par rapport au modèle validé
- Temps moyen de ré-entraînement < 2 heures (sur set incrémental)

Pièces livrables
- Playbook d'incident, Runbook de déploiement, Dashboards Grafana, Scripts d'orchestration.
