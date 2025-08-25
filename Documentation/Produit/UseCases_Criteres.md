# Cas d'usage et critères d'acceptation

Cas d'usage prioritaires

1. Détection d'anomalies de processus
- Description: Identifier les exécutions de cas (case_id) dont le flux d'activités s'écarte du comportement attendu.
- Critères d'acceptation: précision ≥ 0.75 et rappel ≥ 0.7 sur le jeu de test validé.

2. Prédiction d'états à risque pour intervention humaine
- Description: Classifier les événements nécessitant une intervention humaine.
- Critères d'acceptation: F1-weighted ≥ 0.80; latence d'inférence ≤ 200ms en batch local.

3. Monitoring de dérive (data/model drift)
- Description: Détecter les baisses de performance en production.
- Critères d'acceptation: alertes déclenchées si F1-weighted descend de >5% vs référence.

Niveaux d'assurance et seuils
- Accepté: F1-weighted ≥ 0.85 sur validation indépendante.
- Revue requise: 0.80 ≤ F1-weighted < 0.85.
- Rejeter: F1-weighted < 0.80.

Livrables par cas d'usage
- Jeu d'essai annoté, notebook reproductible, rapport de validation signé, playbook d'intervention.
