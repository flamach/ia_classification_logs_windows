# Plan de Test et Validation - Modèle XGBoost Classification Logs

## 1. Objectifs et Critères d'Acceptation

### Objectifs de Validation
- Valider la performance du modèle XGBoost sur données réelles
- Garantir la généralisation et éviter le surapprentissage
- S'assurer de la stabilité des prédictions (reproductibilité)
- Respecter les seuils métier définis

### Critères d'Acceptation
- **F1 Score pondéré ≥ 0.85** (seuil métier principal)
- **Accuracy ≥ 0.80** (performance générale)
- **Précision pondérée ≥ 0.80** (limite fausses alarmes)
- **Rappel pondéré ≥ 0.75** (détection suffisante)
- **Rappel Warning ≥ 0.45** (classe critique minoritaire)
- **Écart-type CV ≤ 0.10** (stabilité entre folds)
- **Gap surapprentissage ≤ 0.15** (différence train-validation)

## 2. Protocole de Validation Croisée (K-Fold Stratifié)

### Configuration
- **Type**: StratifiedKFold 5-fold
- **Stratification**: par classe EntryType
- **Shuffle**: True avec seed=42
- **Parallélisation**: n_jobs=-1

### Métriques Évaluées par Fold
- Accuracy, F1 weighted/macro, Precision/Recall weighted
- Log-loss (probabiliste)
- Métriques par classe (si support suffisant)
- Temps d'exécution et stabilité

### Analyse des Résultats CV
- Moyenne ± écart-type pour chaque métrique
- Détection outliers (folds anormaux)
- Gap train/validation (surapprentissage)
- Distribution des classes par fold

## 3. Phases de Test

### Phase 1: Validation des Données
- **Intégrité**: vérification NaN, types, cohérence
- **Hash MD5**: signature des données pour reproductibilité
- **Distribution**: analyse déséquilibre classes
- **Split stratifié**: 80% train / 20% test final

### Phase 2: Validation Croisée Complète
- Exécution 5-fold avec métriques détaillées
- Analyse fold par fold (performance, distribution)
- Calcul gaps surapprentissage
- Temps d'exécution et stabilité

### Phase 3: Courbe d'Apprentissage
- Test sur 10 tailles d'entraînement (10% à 100%)
- Analyse convergence train vs validation
- Détection plateau de performance
- Recommandations taille optimale dataset

### Phase 4: Test Final Hold-Out
- Réentraînement sur 80% complet
- Évaluation unique sur 20% test (jamais vu)
- Métriques globales + analyse par classe
- Matrice de confusion (brute + normalisée)

### Phase 5: Validation Critères d'Acceptation
- Vérification automatique de tous les seuils
- Génération statut PASS/FAIL par critère
- Décision finale ACCEPTÉ/REJETÉ
- Recommandations si échec

## 4. Artefacts Générés et Signature

### Rapport de Validation Signé
```json
{
  "validation_framework": {
    "version": "1.0",
    "execution_timestamp": "ISO_DATETIME",
    "framework_signature": "ModelValidationFramework_v1.0"
  },
  "data_integrity": {
    "total_samples": N,
    "features_count": M,
    "class_distribution": {...},
    "data_hash": "MD5_HASH"
  },
  "cross_validation_results": {
    "config": {"n_splits": 5, "shuffle": true},
    "summary": {"f1_weighted": {"cv_mean": X, "cv_std": Y}},
    "fold_details": [...]
  },
  "test_evaluation": {
    "global_metrics": {...},
    "per_class_analysis": {...},
    "confusion_matrix": [[...]]
  },
  "acceptance_validation": {
    "all_criteria_passed": true/false,
    "individual_checks": {...}
  },
  "digital_signature": {
    "report_hash": "SHA256_HASH",
    "signature_timestamp": "ISO_DATETIME",
    "validation_authority": "ModelValidationFramework"
  }
}
```

### Fichiers de Sortie
- `validation_report_signed.json` - Rapport complet signé
- `xgboost_model_validated.model` - Modèle final validé
- `confusion_matrix_validation.png` - Visualisations
- `learning_curve_analysis.png` - Courbes d'apprentissage

## 5. Processus de Mise en Production

### Critères de Déploiement
1. **Validation PASS**: tous critères respectés
2. **Hash intégrité**: vérification signature rapport
3. **Tests d'intégration**: API/pipeline production
4. **Approbation métier**: validation experte domaine

### Pipeline de Déploiement
```
Validation Framework → Tests Unitaires → Staging → Validation Métier → Production
```

### Monitoring Post-Déploiement
- Drift detection (distribution features)
- Performance monitoring (métriques dégradation)
- Alertes automatiques si F1 < seuil
- Plan réentraînement automatique

## 6. Commandes d'Exécution

### Validation Complète
```bash
python validation_framework.py
```

### Vérification Signature Rapport
```python
import json, hashlib
report = json.load(open('validation_report_signed.json'))
hash_check = hashlib.sha256(json.dumps(report[:-1]).encode()).hexdigest()
print("Intégrité:", hash_check == report['digital_signature']['report_hash'])
```

### Critères d'Échec et Actions Correctives
- **F1 < 0.85**: augmenter données, rebalancer classes, tuner hyperparamètres
- **Surapprentissage**: régularisation, early stopping, plus de données
- **Instabilité CV**: vérifier données, augmenter folds, cross-validation répétée
- **Classe Warning faible**: oversampling ciblé, features spécialisées

## 7. Validation Reproductibilité
- **Seeds fixes**: 42 partout (random_state)
- **Versions packages**: requirements.txt figé
- **Hash données**: vérification intégrité avant chaque run
- **Environnement**: virtualenv isolé et versionné
