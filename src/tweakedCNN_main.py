import glob
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# -------------------------
# 1. Load CSV files
# -------------------------
# Get the absolute path of the directory where this script is located 
script_dir = os.path.dirname(os.path.abspath(__file__)) 
# Construct the full path to your data directory
data_dir = os.path.join(script_dir, '..', 'data')

# Get a list of all the data files
train_files = [f"{data_dir}/train/{f}" for f in os.listdir(f"{data_dir}/train") if f.endswith('.csv')]
test_files = [f"{data_dir}/test/{f}" for f in os.listdir(f"{data_dir}/test") if f.endswith('.csv')]

# Join the list of files together
files = train_files + test_files

print("Files found:", len(files))

if len(files) == 0:
    raise Exception("No CSV files found. Check folder path.")

df_list = []

for file in files:
    print("Loading:", file)
    temp_df = pd.read_csv(file)

    filename = os.path.basename(file)
    label = filename.replace(".pcap.csv", "")
    temp_df["Label"] = label

    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# -------------------------
# 2. Clean data
# -------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Optional: smaller sample for testing
# df = df.sample(50000, random_state=42)

# -------------------------
# 3. Features + Labels
# -------------------------
X = df.drop(columns=["Label"])
X = X.select_dtypes(include=[np.number])

y = df["Label"]

# -------------------------
# 4. Encode labels
# -------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# -------------------------
# 5. Scale features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# 6. Create sequences
# -------------------------
def create_sequences(X, y, seq_len):
    X_seq = []
    y_seq = []

    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])

    return np.array(X_seq), np.array(y_seq)

SEQ_LEN = 10
X_seq, y_seq = create_sequences(X_scaled, y_encoded, SEQ_LEN)

print("Sequence shape:", X_seq.shape)

# -------------------------
# 7. Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq,
    y_seq,
    test_size=0.2,
    random_state=42
)

# -------------------------
# 8. Build 1D CNN model
# -------------------------
model = Sequential([
    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        input_shape=(X_train.shape[1], X_train.shape[2])
    ),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(len(set(y_encoded)), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# 9. Train model
# -------------------------
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# -------------------------
# 10. Evaluate
# -------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_pred_categorical = model.predict(X_test)
y_pred_encoded = y_pred_categorical.argmax(axis=1)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

y_test_decoded = label_encoder.inverse_transform(y_test.argmax(axis=1))
accuracy = accuracy_score(y_test_decoded, y_pred)
precision = precision_score(y_test_decoded, y_pred, average='weighted')
recall = recall_score(y_test_decoded, y_pred, average='weighted')
f1 = f1_score(y_test_decoded, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("\nClassification Report:\n", classification_report(y_test_decoded, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_decoded, y_pred))
