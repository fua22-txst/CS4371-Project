import os
import random
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

from data_loader import load_and_preprocess_data
from model import create_cnn_model, train_model

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Match the class weight used in main.py
CLASS_WEIGHT = {0: 1.0, 1: 5.0}


def evaluate(name, y_true, y_pred):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')

    X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, label_encoder = \
        load_and_preprocess_data(data_dir, class_config=2)

    # Sklearn models need flat (n_samples, n_features) input
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat   = X_val.reshape(X_val.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    # Integer labels for sklearn
    y_train_int = y_train_cat.argmax(axis=1)
    y_val_int   = y_val_cat.argmax(axis=1)
    y_test_int  = y_test_cat.argmax(axis=1)

    # Decoded string labels for reporting
    y_test_decoded = label_encoder.inverse_transform(y_test_int)

    # Combine train + val for sklearn — they don't use a validation set during training
    X_sk = np.concatenate([X_train_flat, X_val_flat])
    y_sk = np.concatenate([y_train_int, y_val_int])
    sample_weights = np.array([CLASS_WEIGHT[y] for y in y_sk])

    results = {}

    # ── CNN ──────────────────────────────────────────────────────────────────
    print("\nTraining CNN...")
    cnn = create_cnn_model(input_shape=(X_train.shape[1], 1), num_classes=y_train_cat.shape[1])
    cnn = train_model(cnn, X_train, y_train_cat, X_val, y_val_cat,
                      epochs=15, batch_size=32, class_weight=CLASS_WEIGHT)
    results['CNN'] = label_encoder.inverse_transform(
        cnn.predict(X_test).argmax(axis=1)
    )

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight=CLASS_WEIGHT,
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_sk, y_sk)
    results['Random Forest'] = label_encoder.inverse_transform(rf.predict(X_test_flat))

    # ── Gradient Boosting ─────────────────────────────────────────────────────
    # GradientBoostingClassifier doesn't accept class_weight; use sample_weight instead
    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    gb.fit(X_sk, y_sk, sample_weight=sample_weights)
    results['Gradient Boosting'] = label_encoder.inverse_transform(gb.predict(X_test_flat))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    for name, y_pred in results.items():
        acc  = accuracy_score(y_test_decoded, y_pred)
        prec = precision_score(y_test_decoded, y_pred, average='weighted')
        rec  = recall_score(y_test_decoded, y_pred, average='weighted')
        f1   = f1_score(y_test_decoded, y_pred, average='weighted')
        print(f"{name:<22} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

    # ── Detailed per-model reports ────────────────────────────────────────────
    for name, y_pred in results.items():
        evaluate(name, y_test_decoded, y_pred)
