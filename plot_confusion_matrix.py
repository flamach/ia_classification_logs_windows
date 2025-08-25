import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES_PATH = Path("eventlog_preprocessed.csv")
LABELS_PATH = Path("labels.npy")
LABEL_CLASSES_PATH = Path("label_classes.npy")
MODEL_PATH = Path("xgboost_model_balanced.model")
OUTPUT_IMAGE_RAW = Path("confusion_matrix_raw.png")
OUTPUT_IMAGE_NORM = Path("confusion_matrix_normalized.png")
OUTPUT_CSV = Path("confusion_matrix.csv")


def load_data():
    if not FEATURES_PATH.exists() or not LABELS_PATH.exists():
        logger.error("Fichiers prétraités manquants. Exécute preprocess_eventlog.py d'abord.")
        raise SystemExit(1)
    X = pd.read_csv(FEATURES_PATH)
    y = np.load(LABELS_PATH)
    return X, y


def load_model():
    if not MODEL_PATH.exists():
        logger.error(f"Modèle introuvable ({MODEL_PATH}). Entraîne le modèle d'abord.")
        raise SystemExit(1)
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def load_label_names():
    if LABEL_CLASSES_PATH.exists():
        return np.load(LABEL_CLASSES_PATH, allow_pickle=True)
    return None


def save_confusion_csv(cm, label_names, path):
    df = pd.DataFrame(cm, index=label_names, columns=label_names)
    df.to_csv(path)


def plot_and_save(cm, labels, title, outpath, fmt="d", cmap="Blues"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.ylabel("Vrai label")
    plt.xlabel("Prédit")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    logger.info("Chargement des données prétraitées...")
    X, y = load_data()

    logger.info("Split train/test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info("Chargement du modèle...")
    model = load_model()

    logger.info("Prédictions sur l'ensemble de test...")
    y_pred = model.predict(X_test)

    logger.info("Calcul de la matrice de confusion et du rapport de classification...")
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    label_names = load_label_names()
    if label_names is None:
        label_names = [str(i) for i in range(cm.shape[0])]

    logger.info("Classification report :\n" + classification_report(y_test, y_pred, target_names=label_names))

    logger.info(f"Sauvegarde CSV de la matrice brute -> {OUTPUT_CSV}")
    save_confusion_csv(cm, label_names, OUTPUT_CSV)

    logger.info(f"Sauvegarde image matrice brute -> {OUTPUT_IMAGE_RAW}")
    plot_and_save(cm, label_names, "Matrice de confusion (brute)", OUTPUT_IMAGE_RAW, fmt="d")

    # Formatage pourcentage pour la matrice normalisée
    cm_norm_pct = np.round(cm_norm * 100, 2)
    logger.info(f"Sauvegarde image matrice normalisée -> {OUTPUT_IMAGE_NORM}")
    plot_and_save(cm_norm_pct, label_names, "Matrice de confusion (normalisée, %)", OUTPUT_IMAGE_NORM, fmt=".2f", cmap="OrRd")

    logger.info("Terminé. Fichiers générés : %s, %s, %s", OUTPUT_CSV, OUTPUT_IMAGE_RAW, OUTPUT_IMAGE_NORM)


if __name__ == "__main__":
    main()
