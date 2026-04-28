import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

def create_cnn_model(input_shape, num_classes):
    """Create and compile the tweaked CNN model."""
    model = Sequential([
        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            input_shape=input_shape
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
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, epochs=10, batch_size=32):
    """Train the tweaked CNN model."""
    model.fit(X_train, y_train_categorical, epochs=epochs, batch_size=batch_size, 
              validation_data=(X_val, y_val_categorical))
    return model