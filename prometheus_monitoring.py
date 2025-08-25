"""
Script d'instrumentation Prometheus pour le monitoring d'entraînement XGBoost
"""

import time
import psutil
import logging
from typing import Dict, Any
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
from prometheus_client.exposition import basic_auth_handler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

class XGBoostPrometheusMonitor:
    def __init__(self, pushgateway_url: str = "localhost:9091", job_name: str = "xgboost_training"):
        """
        Initialise le monitoring Prometheus
        
        Args:
            pushgateway_url: URL du Pushgateway Prometheus
            job_name: Nom du job pour identification
        """
        self.pushgateway_url = pushgateway_url
        self.job_name = job_name
        self.registry = CollectorRegistry()
        
        # Métriques de performance du modèle
        self.f1_weighted = Gauge('xgboost_validation_f1_weighted', 
                                'F1 score pondéré sur validation', registry=self.registry)
        self.accuracy = Gauge('xgboost_validation_accuracy', 
                             'Accuracy sur validation', registry=self.registry)
        self.precision_weighted = Gauge('xgboost_validation_precision_weighted', 
                                       'Précision pondérée sur validation', registry=self.registry)
        self.recall_weighted = Gauge('xgboost_validation_recall_weighted', 
                                    'Rappel pondéré sur validation', registry=self.registry)
        
        # F1 par classe
        self.f1_per_class = Gauge('xgboost_f1_score', 
                                 'F1 score par classe', ['class'], registry=self.registry)
        
        # Métriques système
        self.train_duration = Gauge('xgboost_train_duration_seconds', 
                                   'Durée d\'entraînement en secondes', registry=self.registry)
        self.rows_processed = Gauge('xgboost_rows_processed', 
                                   'Nombre de lignes traitées', registry=self.registry)
        self.cpu_usage = Gauge('xgboost_cpu_usage_percent', 
                              'Utilisation CPU en %', registry=self.registry)
        self.memory_usage = Gauge('xgboost_memory_usage_bytes', 
                                 'Utilisation mémoire en bytes', registry=self.registry)
        
        # Compteurs d'expériences
        self.experiments_total = Counter('xgboost_experiments_total', 
                                        'Nombre total d\'expériences', registry=self.registry)
        self.successful_runs = Counter('xgboost_successful_runs', 
                                      'Nombre de runs réussis', registry=self.registry)
        self.failed_runs = Counter('xgboost_failed_runs', 
                                  'Nombre de runs échoués', registry=self.registry)
        
        # Hyperparameter tuning
        self.hyperparameter_trials = Counter('xgboost_hyperparameter_trials', 
                                            'Nombre d\'essais hyperparamètres', registry=self.registry)
        self.early_stopping = Gauge('xgboost_early_stopping_triggered', 
                                   'Early stopping déclenché (1/0)', registry=self.registry)
        
        # Matrice de confusion
        self.confusion_matrix_metric = Gauge('xgboost_confusion_matrix', 
                                            'Valeurs matrice de confusion', 
                                            ['true_class', 'predicted_class'], registry=self.registry)
        
        # Historique de métriques
        self.validation_loss_history = Histogram('xgboost_validation_loss', 
                                                'Historique des losses de validation', 
                                                registry=self.registry)
        
        self.start_time = time.time()
        logger.info(f"Monitoring Prometheus initialisé - Job: {job_name}, Pushgateway: {pushgateway_url}")
    
    def update_system_metrics(self):
        """Met à jour les métriques système (CPU, RAM)"""
        try:
            self.cpu_usage.set(psutil.cpu_percent())
            self.memory_usage.set(psutil.virtual_memory().used)
        except Exception as e:
            logger.warning(f"Erreur collecte métriques système: {e}")
    
    def record_training_start(self, total_rows: int):
        """Enregistre le début d'entraînement"""
        self.experiments_total.inc()
        self.rows_processed.set(total_rows)
        self.start_time = time.time()
        logger.info(f"Début entraînement - {total_rows} lignes")
    
    def record_training_metrics(self, metrics: Dict[str, float], class_names: list = None):
        """
        Enregistre les métriques d'entraînement
        
        Args:
            metrics: Dict avec accuracy, f1_weighted, precision_weighted, recall_weighted
            class_names: Liste des noms de classes pour F1 par classe
        """
        try:
            self.accuracy.set(metrics.get('accuracy', 0))
            self.f1_weighted.set(metrics.get('f1_weighted', 0))
            self.precision_weighted.set(metrics.get('precision_weighted', 0))
            self.recall_weighted.set(metrics.get('recall_weighted', 0))
            
            # F1 par classe si disponible
            if 'f1_per_class' in metrics and class_names:
                f1_scores = metrics['f1_per_class']
                for class_name, f1_score in zip(class_names, f1_scores):
                    self.f1_per_class.labels(class=str(class_name)).set(f1_score)
            
            logger.info(f"Métriques enregistrées - F1: {metrics.get('f1_weighted', 0):.3f}")
            
        except Exception as e:
            logger.error(f"Erreur enregistrement métriques: {e}")
    
    def record_confusion_matrix(self, y_true, y_pred, class_names: list):
        """
        Enregistre la matrice de confusion
        
        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions
            class_names: Noms des classes
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            for i, true_class in enumerate(class_names):
                for j, pred_class in enumerate(class_names):
                    if i < cm.shape[0] and j < cm.shape[1]:
                        self.confusion_matrix_metric.labels(
                            true_class=str(true_class), 
                            predicted_class=str(pred_class)
                        ).set(cm[i, j])
            
            logger.info("Matrice de confusion enregistrée")
            
        except Exception as e:
            logger.error(f"Erreur enregistrement matrice confusion: {e}")
    
    def record_training_end(self, success: bool = True, early_stopped: bool = False):
        """Enregistre la fin d'entraînement"""
        duration = time.time() - self.start_time
        self.train_duration.set(duration)
        self.early_stopping.set(1 if early_stopped else 0)
        
        if success:
            self.successful_runs.inc()
        else:
            self.failed_runs.inc()
        
        logger.info(f"Fin entraînement - Durée: {duration:.2f}s, Succès: {success}")
    
    def record_hyperparameter_trial(self):
        """Enregistre un essai d'hyperparamètres"""
        self.hyperparameter_trials.inc()
    
    def record_validation_loss(self, loss: float):
        """Enregistre une loss de validation"""
        self.validation_loss_history.observe(loss)
    
    def push_metrics(self):
        """Pousse les métriques vers Pushgateway"""
        try:
            # Mise à jour métriques système avant push
            self.update_system_metrics()
            
            # Push vers Pushgateway
            push_to_gateway(
                self.pushgateway_url, 
                job=self.job_name, 
                registry=self.registry
            )
            logger.info("Métriques poussées vers Pushgateway")
            
        except Exception as e:
            logger.error(f"Erreur push métriques: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques actuelles"""
        return {
            'f1_weighted': self.f1_weighted._value._value if hasattr(self.f1_weighted._value, '_value') else 0,
            'accuracy': self.accuracy._value._value if hasattr(self.accuracy._value, '_value') else 0,
            'train_duration': self.train_duration._value._value if hasattr(self.train_duration._value, '_value') else 0,
            'rows_processed': self.rows_processed._value._value if hasattr(self.rows_processed._value, '_value') else 0,
        }


# Fonction utilitaire pour calculer les métriques depuis classification_report
def extract_metrics_from_classification_report(y_true, y_pred, class_names: list) -> Dict[str, float]:
    """
    Extrait les métriques depuis sklearn classification_report
    
    Returns:
        Dict avec accuracy, f1_weighted, precision_weighted, recall_weighted, f1_per_class
    """
    try:
        from sklearn.metrics import classification_report, accuracy_score
        
        # Rapport de classification
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Métriques pondérées
        weighted_avg = report.get('weighted avg', {})
        
        # F1 par classe
        f1_per_class = []
        for class_name in class_names:
            if str(class_name) in report:
                f1_per_class.append(report[str(class_name)].get('f1-score', 0))
            else:
                f1_per_class.append(0)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': weighted_avg.get('f1-score', 0),
            'precision_weighted': weighted_avg.get('precision', 0),
            'recall_weighted': weighted_avg.get('recall', 0),
            'f1_per_class': f1_per_class
        }
        
    except Exception as e:
        logger.error(f"Erreur extraction métriques: {e}")
        return {
            'accuracy': 0, 'f1_weighted': 0, 'precision_weighted': 0, 
            'recall_weighted': 0, 'f1_per_class': []
        }


# Exemple d'utilisation dans le script d'entraînement
if __name__ == "__main__":
    # Initialisation du monitoring
    monitor = XGBoostPrometheusMonitor()
    
    # Simulation d'un entraînement
    monitor.record_training_start(total_rows=10000)
    
    # Simulation métriques
    metrics = {
        'accuracy': 0.91,
        'f1_weighted': 0.89,
        'precision_weighted': 0.90,
        'recall_weighted': 0.88,
        'f1_per_class': [0.62, 0.95, 0.95, 0.60]
    }
    class_names = ['0', 'Error', 'Information', 'Warning']
    
    monitor.record_training_metrics(metrics, class_names)
    monitor.record_training_end(success=True)
    
    # Push final
    monitor.push_metrics()
    
    print("Monitoring test terminé - vérifiez Grafana")
