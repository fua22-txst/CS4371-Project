import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# This function builds and compiles the RNN model
def create_rnn_model(input_shape, num_classes):
    """Create and compile the RNN model."""
    #first RNN layer, return sequence is true so that the next layer gets full output.
    model = Sequential()

    #second layer
    # Recurrent neural network feature extractor
    model.add(SimpleRNN(128, return_sequences=True, input_shape=input_shape))

    # Fully connected (dense) layer for learning patterns after RNN processing
    model.add(SimpleRNN(64))

    #output layer
    # Classifier
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) 

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, epochs=10, batch_size=32):
    """Train the RNN model."""
    model.fit(X_train, y_train_categorical, epochs=epochs, batch_size=batch_size, 
              validation_data=(X_val, y_val_categorical))
    return model
