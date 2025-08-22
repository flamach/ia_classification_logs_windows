import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(csv_path='eventlog.csv', out_features='eventlog_preprocessed.csv', out_labels='labels.npy'):
    logger.info("Chargement et nettoyage des données...")
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=['Message', 'Source', 'EntryType'])
    class_counts = df['EntryType'].value_counts()
    max_size = class_counts.max()
    balanced_dfs = []
    for entry_type in df['EntryType'].unique():
        class_df = df[df['EntryType'] == entry_type]
        if len(class_df) > max_size * 0.4:
            target_size = int(max_size * 0.4)
            class_df = class_df.sample(n=target_size, random_state=42)
        balanced_dfs.append(class_df)
    df = pd.concat(balanced_dfs)
    np.random.seed(42)
    df['message_length'] = df['Message'].str.len() * (1 + np.random.normal(0, 0.2, size=len(df)))
    df['word_count'] = df['Message'].str.split().str.len() * (1 + np.random.normal(0, 0.2, size=len(df)))
    df['has_error'] = df['Message'].str.lower().str.contains('error|fail').astype(int)
    df['has_error'] = df['has_error'] + np.random.normal(0, 0.2, size=len(df))
    sources = pd.Categorical(df['Source']).codes
    df['source'] = sources + np.random.normal(0, 0.3, size=len(sources))
    features = ['message_length', 'word_count', 'source', 'has_error']
    X = df[features]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['EntryType'])
    logger.info("Sauvegarde des features et labels...")
    X.to_csv(out_features, index=False)
    np.save(out_labels, y)
    np.save('label_classes.npy', label_encoder.classes_)
    logger.info("Prétraitement terminé.")

if __name__ == "__main__":
    preprocess_data()
