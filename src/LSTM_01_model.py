import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(input_shape, num_classes):
    """Create and compile the LSTM model."""
    model = Sequential()
    # Add a 64-neuron Long Short Term Memory layer to start,
    # replacing the original model's format 
    # original: Conv1D -> Maxpooling1D -> Conv1D -> Maxpooling1D -> Flatten -> Dense
    # LSTM: LSTM -> Dense
    model.add(LSTM(64, input_shape=input_shape))

    # The Dense layer reconfigures the neuron's classifications as 
    # something more human-readable for multiclassification tasks
    model.add(Dense(num_classes, activation='softmax'))

    # Use the adam optimizer, multiclassification, and use accuracy as the primary metric
    # This is the same as the original paper's model's final parameters.
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# This function is the same as original paper's model but with callbacks (checkpoints)
# so that training can be paused/resumed
def train_model(model, X_train, y_train_categorical, 
                X_val, y_val_categorical, 
                epochs=10, batch_size=32, checkpoint_callback=[], initial_epoch=0):
    """Train the LSTM model with checkpointing."""
    
    model.fit(
        X_train, y_train_categorical,
        epochs=epochs,
        initial_epoch=initial_epoch,
        batch_size=batch_size,
        validation_data=(X_val, y_val_categorical),
        callbacks=[checkpoint_callback]
    )
    
    return model
