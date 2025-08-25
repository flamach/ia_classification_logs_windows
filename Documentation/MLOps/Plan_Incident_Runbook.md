# Runbook d'incident (extrait)

Symptômes courants
- Baisse de F1-weighted > 5% par rapport à la référence
- Temps d'inférence anormalement élevé (> 1s par prédiction en batch)

Actions immédiates
1. Vérifier l'intégrité du dataset et les distributions de classes (script: `preprocess_eventlog.py --stats`).
2. Recalculer les métriques locales via `validation_framework.py`.
3. Si dataset OK, redéployer le dernier modèle validé et ouvrir investigation.

Escalation
- Si incident persiste >1 heure, contacter l'équipe Data Ops pour rollback des artefacts sur MinIO/NFS.
