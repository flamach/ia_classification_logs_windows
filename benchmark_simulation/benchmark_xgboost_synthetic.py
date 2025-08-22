import pandas as pd
import numpy as np
import time
import psutil
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import logging
import os

def monitor_resources():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # MB
    cpu = process.cpu_percent(interval=1)
    return mem, cpu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

logger.info("Chargement du fichier synthetic_eventlog.csv...")
start_load = time.time()
df = pd.read_csv("benchmark_simulation/synthetic_eventlog.csv")
end_load = time.time()
mem_load, cpu_load = monitor_resources()
logger.info(f"Chargement terminé en {end_load - start_load:.2f}s | RAM: {mem_load:.1f}MB | CPU: {cpu_load:.1f}%")

logger.info("Prétraitement des données...")
start_prep = time.time()
df['message_length'] = df['Message'].str.len()
df['word_count'] = df['Message'].str.split().str.len()
df['has_error'] = df['Message'].str.lower().str.contains('error|fail').astype(int)
sources = pd.Categorical(df['Source']).codes
df['source'] = sources
features = ['message_length', 'word_count', 'source', 'has_error']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['EntryType'])
X = df[features]
end_prep = time.time()
mem_prep, cpu_prep = monitor_resources()
logger.info(f"Prétraitement terminé en {end_prep - start_prep:.2f}s | RAM: {mem_prep:.1f}MB | CPU: {cpu_prep:.1f}%")

logger.info("Entraînement du modèle XGBoost...")
start_train = time.time()
model = XGBClassifier(n_estimators=10, max_depth=3, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7, random_state=42)
model.fit(X, y)
end_train = time.time()
mem_train, cpu_train = monitor_resources()
logger.info(f"Entraînement terminé en {end_train - start_train:.2f}s | RAM: {mem_train:.1f}MB | CPU: {cpu_train:.1f}%")

logger.info("Prédiction sur l'ensemble complet...")
start_pred = time.time()
y_pred = model.predict(X)
end_pred = time.time()
mem_pred, cpu_pred = monitor_resources()
logger.info(f"Prédiction terminée en {end_pred - start_pred:.2f}s | RAM: {mem_pred:.1f}MB | CPU: {cpu_pred:.1f}%")

logger.info("Benchmark terminé. Stabilité et performance OK si pas d'erreur ou crash.")
