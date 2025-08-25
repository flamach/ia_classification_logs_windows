"""
Plan de Test et Validation du Modèle XGBoost - Classification Logs Windows
Phase de validation rigoureuse avec k-fold stratifié et résultats signés
"""

import logging
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, cross_validate, 
    validation_curve, learning_curve
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, log_loss
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelValidationFramework:
    """Framework de validation et test complet pour modèle XGBoost"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.validation_results = {}
        self.test_results = {}
        self.signature_metadata = {}
        
        # Seuils d'acceptation
        self.acceptance_criteria = {
            'min_f1_weighted': 0.85,
            'min_accuracy': 0.80,
            'min_precision_weighted': 0.80,
            'min_recall_weighted': 0.75,
            'max_std_cv': 0.10,  # Écart-type max entre folds
            'min_warning_recall': 0.45,  # Rappel minimum pour classe Warning
            'max_overfitting_gap': 0.15  # Gap max entre train et validation
        }
        
        # Configuration validation croisée
        self.cv_config = {
            'n_splits': 5,
            'shuffle': True,
            'random_state': random_state
        }
        
        logger.info(f"Framework de validation initialisé - Seed: {random_state}")
    
    def load_data(self, features_path: str, labels_path: str, label_classes_path: str):
        """Charge et valide les données"""
        logger.info("=== CHARGEMENT ET VALIDATION DES DONNÉES ===")
        
        # Chargement
        X = pd.read_csv(features_path)
        y = np.load(labels_path)
        label_names = np.load(label_classes_path, allow_pickle=True)
        
        # Validation intégrité
        assert len(X) == len(y), "Mismatch entre features et labels"
        assert not X.isnull().any().any(), "Features contiennent des NaN"
        assert not np.isnan(y).any(), "Labels contiennent des NaN"
        
        # Statistiques descriptives
        stats = {
            'total_samples': len(X),
            'features_count': X.shape[1],
            'classes_count': len(np.unique(y)),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'data_hash': hashlib.md5(pd.concat([X, pd.Series(y)], axis=1).to_string().encode()).hexdigest()[:16]
        }
        
        logger.info(f"Données chargées: {stats['total_samples']} échantillons, {stats['features_count']} features")
        logger.info(f"Distribution classes: {stats['class_distribution']}")
        logger.info(f"Hash données: {stats['data_hash']}")
        
        self.data_stats = stats
        return X, y, label_names
    
    def prepare_splits(self, X, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prépare les splits train/test avec stratification"""
        logger.info("=== PRÉPARATION DES SPLITS ===")
        
        # Split stratifié 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Vérification stratification
        train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
        test_dist = dict(zip(*np.unique(y_test, return_counts=True)))
        
        logger.info(f"Train: {len(X_train)} échantillons - {train_dist}")
        logger.info(f"Test: {len(X_test)} échantillons - {test_dist}")
        
        # Vérification proportions similaires
        for class_id in train_dist:
            train_prop = train_dist[class_id] / len(y_train)
            test_prop = test_dist[class_id] / len(y_test)
            prop_diff = abs(train_prop - test_prop)
            logger.info(f"Classe {class_id}: Train {train_prop:.3f}, Test {test_prop:.3f}, Diff {prop_diff:.3f}")
            assert prop_diff < 0.05, f"Stratification échouée pour classe {class_id}"
        
        self.split_info = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_distribution': train_dist,
            'test_distribution': test_dist,
            'split_hash': hashlib.md5(str(X_train.index.tolist() + X_test.index.tolist()).encode()).hexdigest()[:16]
        }
        
        return X_train, X_test, y_train, y_test
    
    def cross_validation_comprehensive(self, model, X_train, y_train, label_names) -> Dict:
        """Validation croisée complète avec métriques détaillées"""
        logger.info("=== VALIDATION CROISÉE STRATIFIÉE (K-FOLD) ===")
        
        # Configuration CV
        cv = StratifiedKFold(**self.cv_config)
        
        # Métriques à évaluer
        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'f1_macro': 'f1_macro',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'neg_log_loss': 'neg_log_loss'
        }
        
        # Cross-validation complète
        logger.info(f"Exécution CV {self.cv_config['n_splits']}-fold...")
        cv_start = time.time()
        
        cv_results = cross_validate(
            model, X_train, y_train, 
            cv=cv, scoring=scoring, 
            return_train_score=True,
            n_jobs=-1
        )
        
        cv_duration = time.time() - cv_start
        logger.info(f"CV terminée en {cv_duration:.2f}s")
        
        # Analyse des résultats
        results_summary = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results_summary[metric] = {
                'cv_mean': float(test_scores.mean()),
                'cv_std': float(test_scores.std()),
                'cv_scores': test_scores.tolist(),
                'train_mean': float(train_scores.mean()),
                'train_std': float(train_scores.std()),
                'overfitting_gap': float(train_scores.mean() - test_scores.mean())
            }
            
            logger.info(f"{metric}: CV {test_scores.mean():.4f} ± {test_scores.std():.4f}, "
                       f"Train {train_scores.mean():.4f}, Gap {results_summary[metric]['overfitting_gap']:.4f}")
        
        # Validation croisée par classe (approximative via stratification)
        logger.info("Analyse détaillée par fold...")
        fold_details = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Entraînement sur ce fold
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_fold_train, y_fold_train)
            y_fold_pred = fold_model.predict(X_fold_val)
            
            # Métriques par classe
            fold_report = classification_report(y_fold_val, y_fold_pred, 
                                              target_names=label_names, 
                                              output_dict=True, zero_division=0)
            
            fold_details.append({
                'fold': fold_idx,
                'val_size': len(y_fold_val),
                'val_distribution': dict(zip(*np.unique(y_fold_val, return_counts=True))),
                'classification_report': fold_report
            })
        
        # Compilation finale
        cv_comprehensive = {
            'duration_seconds': cv_duration,
            'config': self.cv_config,
            'summary': results_summary,
            'fold_details': fold_details,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        self.validation_results = cv_comprehensive
        return cv_comprehensive
    
    def learning_curve_analysis(self, model, X_train, y_train) -> Dict:
        """Analyse de la courbe d'apprentissage"""
        logger.info("=== ANALYSE COURBE D'APPRENTISSAGE ===")
        
        # Tailles d'entraînement à tester
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        learning_analysis = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'train_scores_std': train_scores.std(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist(),
            'val_scores_std': val_scores.std(axis=1).tolist(),
            'final_gap': float(train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1])
        }
        
        logger.info(f"Courbe d'apprentissage: gap final {learning_analysis['final_gap']:.4f}")
        
        return learning_analysis
    
    def final_test_evaluation(self, model, X_train, y_train, X_test, y_test, label_names) -> Dict:
        """Évaluation finale sur ensemble test"""
        logger.info("=== ÉVALUATION FINALE SUR ENSEMBLE TEST ===")
        
        # Réentraînement sur tout le train
        logger.info("Réentraînement sur ensemble train complet...")
        model.fit(X_train, y_train)
        
        # Prédictions test
        logger.info("Prédictions sur ensemble test...")
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        
        # Métriques globales
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'f1_weighted': precision_recall_fscore_support(y_test, y_test_pred, average='weighted')[2],
            'precision_weighted': precision_recall_fscore_support(y_test, y_test_pred, average='weighted')[0],
            'recall_weighted': precision_recall_fscore_support(y_test, y_test_pred, average='weighted')[1],
            'log_loss': log_loss(y_test, y_test_proba)
        }
        
        # Rapport de classification détaillé
        classification_rep = classification_report(y_test, y_test_pred, 
                                                 target_names=label_names, 
                                                 output_dict=True, zero_division=0)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_test_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Analyse par classe
        per_class_analysis = {}
        for i, class_name in enumerate(label_names):
            if str(class_name) in classification_rep:
                per_class_analysis[str(class_name)] = {
                    'precision': classification_rep[str(class_name)]['precision'],
                    'recall': classification_rep[str(class_name)]['recall'],
                    'f1_score': classification_rep[str(class_name)]['f1-score'],
                    'support': classification_rep[str(class_name)]['support']
                }
        
        test_evaluation = {
            'test_size': len(X_test),
            'global_metrics': test_metrics,
            'classification_report': classification_rep,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'per_class_analysis': per_class_analysis,
            'test_timestamp': datetime.now().isoformat()
        }
        
        logger.info("=== RÉSULTATS TEST FINAUX ===")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info("\nPerformance par classe:")
        for class_name, metrics in per_class_analysis.items():
            logger.info(f"{class_name}: F1={metrics['f1_score']:.3f}, "
                       f"Precision={metrics['precision']:.3f}, "
                       f"Recall={metrics['recall']:.3f}")
        
        self.test_results = test_evaluation
        return test_evaluation
    
    def acceptance_validation(self) -> Dict:
        """Validation des critères d'acceptation"""
        logger.info("=== VALIDATION CRITÈRES D'ACCEPTATION ===")
        
        validation_status = {}
        
        # Critères CV
        cv_f1 = self.validation_results['summary']['f1_weighted']['cv_mean']
        cv_f1_std = self.validation_results['summary']['f1_weighted']['cv_std']
        overfitting_gap = self.validation_results['summary']['f1_weighted']['overfitting_gap']
        
        # Critères test
        test_f1 = self.test_results['global_metrics']['f1_weighted']
        test_accuracy = self.test_results['global_metrics']['accuracy']
        test_precision = self.test_results['global_metrics']['precision_weighted']
        test_recall = self.test_results['global_metrics']['recall_weighted']
        
        # Rappel Warning si disponible
        warning_recall = 0
        if 'Warning' in self.test_results['per_class_analysis']:
            warning_recall = self.test_results['per_class_analysis']['Warning']['recall']
        
        # Vérifications
        checks = {
            'f1_weighted_threshold': (test_f1 >= self.acceptance_criteria['min_f1_weighted'], 
                                    f"F1 test {test_f1:.3f} >= {self.acceptance_criteria['min_f1_weighted']}"),
            'accuracy_threshold': (test_accuracy >= self.acceptance_criteria['min_accuracy'],
                                 f"Accuracy {test_accuracy:.3f} >= {self.acceptance_criteria['min_accuracy']}"),
            'precision_threshold': (test_precision >= self.acceptance_criteria['min_precision_weighted'],
                                  f"Precision {test_precision:.3f} >= {self.acceptance_criteria['min_precision_weighted']}"),
            'recall_threshold': (test_recall >= self.acceptance_criteria['min_recall_weighted'],
                               f"Recall {test_recall:.3f} >= {self.acceptance_criteria['min_recall_weighted']}"),
            'cv_stability': (cv_f1_std <= self.acceptance_criteria['max_std_cv'],
                           f"CV std {cv_f1_std:.3f} <= {self.acceptance_criteria['max_std_cv']}"),
            'overfitting_control': (overfitting_gap <= self.acceptance_criteria['max_overfitting_gap'],
                                  f"Overfitting gap {overfitting_gap:.3f} <= {self.acceptance_criteria['max_overfitting_gap']}"),
            'warning_recall': (warning_recall >= self.acceptance_criteria['min_warning_recall'],
                             f"Warning recall {warning_recall:.3f} >= {self.acceptance_criteria['min_warning_recall']}")
        }
        
        all_passed = all(check[0] for check in checks.values())
        
        validation_status = {
            'all_criteria_passed': all_passed,
            'individual_checks': checks,
            'acceptance_criteria': self.acceptance_criteria,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Résultats validation:")
        for check_name, (passed, message) in checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{status}: {message}")
        
        logger.info(f"\n{'✓ MODÈLE ACCEPTÉ' if all_passed else '✗ MODÈLE REJETÉ'}")
        
        return validation_status
    
    def generate_signed_report(self, model_path: str, output_path: str) -> Dict:
        """Génère un rapport signé avec hash d'intégrité"""
        logger.info("=== GÉNÉRATION RAPPORT SIGNÉ ===")
        
        # Collecte de toutes les informations
        full_report = {
            'validation_framework': {
                'version': '1.0',
                'random_state': self.random_state,
                'execution_timestamp': datetime.now().isoformat(),
                'framework_signature': 'ModelValidationFramework_v1.0'
            },
            'data_integrity': self.data_stats,
            'split_configuration': self.split_info,
            'cross_validation_results': self.validation_results,
            'test_evaluation': self.test_results,
            'acceptance_validation': self.acceptance_validation(),
            'model_metadata': {
                'model_path': model_path,
                'model_type': 'XGBClassifier'
            }
        }
        
        # Conversion des types NumPy pour JSON
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {str(convert_numpy_types(k)): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Hash d'intégrité global
        full_report_clean = convert_numpy_types(full_report)
        report_string = json.dumps(full_report_clean, sort_keys=True, ensure_ascii=False)
        integrity_hash = hashlib.sha256(report_string.encode()).hexdigest()
        
        # Signature finale
        full_report_clean['digital_signature'] = {
            'report_hash': integrity_hash,
            'signature_timestamp': datetime.now().isoformat(),
            'signature_method': 'SHA256',
            'validation_authority': 'ModelValidationFramework',
            'report_version': '1.0'
        }
        
        # Sauvegarde
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_report_clean, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Rapport signé généré: {output_path}")
        logger.info(f"Hash d'intégrité: {integrity_hash}")
        
        return full_report_clean


def main():
    """Exécution complète du plan de validation"""
    logger.info("=== DÉMARRAGE PLAN DE VALIDATION COMPLET ===")
    
    # Initialisation framework
    validator = ModelValidationFramework(random_state=42)
    
    # Chargement données
    X, y, label_names = validator.load_data(
        "eventlog_preprocessed.csv",
        "labels.npy", 
        "label_classes.npy"
    )
    
    # Préparation splits
    X_train, X_test, y_train, y_test = validator.prepare_splits(X, y)
    
    # Modèle à valider (paramètres optimisés)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.3,
        random_state=42,
        n_jobs=-1,
        objective='multi:softprob'
    )
    
    # Phase 1: Validation croisée
    cv_results = validator.cross_validation_comprehensive(model, X_train, y_train, label_names)
    
    # Phase 2: Courbe d'apprentissage
    learning_analysis = validator.learning_curve_analysis(model, X_train, y_train)
    
    # Phase 3: Test final
    test_results = validator.final_test_evaluation(model, X_train, y_train, X_test, y_test, label_names)
    
    # Phase 4: Génération rapport signé
    model.fit(X_train, y_train)  # Entraînement final
    model.save_model("xgboost_model_validated.model")
    
    signed_report = validator.generate_signed_report(
        "xgboost_model_validated.model",
        "validation_report_signed.json"
    )
    
    # Résumé final
    logger.info("=== RÉSUMÉ VALIDATION COMPLÈTE ===")
    acceptance = signed_report['acceptance_validation']
    logger.info(f"Statut final: {'ACCEPTÉ' if acceptance['all_criteria_passed'] else 'REJETÉ'}")
    logger.info(f"F1 pondéré test: {test_results['global_metrics']['f1_weighted']:.4f}")
    logger.info(f"Accuracy test: {test_results['global_metrics']['accuracy']:.4f}")
    logger.info(f"Hash rapport: {signed_report['digital_signature']['report_hash'][:16]}...")
    
    return acceptance['all_criteria_passed']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
