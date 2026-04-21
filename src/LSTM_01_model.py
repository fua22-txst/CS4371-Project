import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(input_shape, num_classes):
    """Create and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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
