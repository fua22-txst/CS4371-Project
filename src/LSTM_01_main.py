import os
import argparse
from LSTM_01_data_loader import load_and_preprocess_data
from LSTM_01_model import create_lstm_model, train_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM for network intrusion detection.")
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=2,
                        help="Number of classes for classification (2, 6, or 19)")
    args = parser.parse_args()

    # Get the absolute path of the directory where this script is located 
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    # Construct the full path to your data directory
    data_dir = os.path.join(script_dir, '..', 'data') 

    # Pass data_dir to the function:
    X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, label_encoder = load_and_preprocess_data(
        data_dir, args.class_config)  # Pass data_dir here 

    input_shape = (X_train.shape[1], 1) 
    model = create_lstm_model(input_shape, y_train_categorical.shape[1])

    # Check if GPU is enabled
    import tensorflow as tf 
    if tf.test.gpu_device_name():
        print('GPU is available!')
    else:
        print('GPU is not available. Using CPU.')



    # Uncomment this to resume from full-model HDF5 checkpoint (in this case, epoch 3/10)
    # model = tf.keras.models.load_model('model_epoch_03.h5')



    # Define unique filepath with epoch placeholder
    checkpoint_path = "LSTM_model_epoch_{epoch:02d}.h5"

    # Create callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,  # Set to True to save only weights
        save_best_only=False,     # Save every epoch
        verbose=1
    )
    print("Callbacks set up. Starting training...")


    model = train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, epochs=10, checkpoint_callback=checkpoint_callback)

    # Evaluate the loss and accuracy
    loss, accuracy = model.evaluate(X_test, y_test_categorical)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Calculate and print the other, more detailed metrics
    y_pred_categorical = model.predict(X_test)
    y_pred_encoded = y_pred_categorical.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    y_test_decoded = label_encoder.inverse_transform(y_test_categorical.argmax(axis=1))

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
