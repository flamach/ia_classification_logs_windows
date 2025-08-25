# Vision et objectifs

Ce projet vise à détecter automatiquement des anomalies ou des classes d'intérêt dans les journaux d'événements (event logs) pour améliorer la supervision opérationnelle et la qualité des processus.

Objectifs principaux
- Fournir un modèle de classification capable d'identifier les états/événements à risque avec un F1-weighted ≥ 0.85 en production.
- Intégrer métriques et alerting via Prometheus/Grafana pour surveiller la performance en continu.
- Permettre ré-entraînement reproductible et traçabilité (modèles versionnés, rapports signés).

Bénéfices attendus
- Réduction du temps moyen de détection d'incidents.
- Amélioration de la qualité des décisions opérationnelles.
- Capacité d'industrialisation on‑premise sans dépendances cloud obligatoires.

Principaux livrables
- Modèle XGBoost validé (`xgboost_model_validated.model`).
- Rapport de validation signé (`validation_report_signed.json`).
- Dashboard Grafana et stack Prometheus/Pushgateway.

Parties prenantes
- Data Science: conception & validation des modèles
- Opérations / Exploitation: déploiement, monitoring
- Sécurité & Conformité: revue des données et accès

Roadmap (sprint 0 → 3 mois)
1. Préparation des données et définition des KPI
2. Entraînement initial, validation et génération du rapport signé
3. Déploiement pilote on‑premise + monitoring
4. Automatisation des réentraînements et intégration CI/CD
