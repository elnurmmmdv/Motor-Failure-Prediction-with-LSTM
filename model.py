import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class Model:
    """
    Model - A class for building and training LSTM models.

    Attributes:
        early_stopping (EarlyStopping): An early stopping callback to prevent overfitting.

    Methods:
        __init__(self, model_type):
            Initializes the Model object with a specified model type.

        build_lstm_model(self):
            Builds an LSTM model based on the specified model type.

        _compile_model(self):
            Compiles the built model with an Adam optimizer and Mean Squared Error loss.

        create_sequence(self, data, sequence_length, sequence_type):
            Creates input sequences and target sequences based on the specified sequence type.

        _determine_sequence_length(self, data, sequence_type):
            Determines the appropriate sequence length based on the sequence type.

        prepare_data(self, train_data, test_data, sequence_type):
            Prepares training and testing data based on the specified sequence type.

        train_model(self, epochs=50, batch_size=128):
            Builds, compiles, and trains the LSTM model with specified epochs and batch size.

    Example:
        model = Model(model_type='m_to_m')
        model.prepare_data(train_data, test_data, sequence_type='sliding_window')
        model.train_model(epochs=50, batch_size=128)
    """

    early_stopping = EarlyStopping(
            monitor='val_mae',
            min_delta=0.1,
            patience=5,
            mode="auto",
            restore_best_weights=True
        )

    def __init__(self, model_type):
        """
        Initializes the Model object with a specified model type.

        Parameters:
            model_type (str): The type of model, either 'm_to_m' or 'm_to_1'.
        """
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.history = None  # To store training history

    def build_lstm_model(self):
        """
        Builds an LSTM model based on the specified model type.
        """
        self.model = tf.keras.Sequential([
            LSTM(150, activation='tanh', input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dense(128, activation='relu'),
            Dense(self.y_test.shape[1] if self.model_type == 'm_to_m' else 1)
        ])

        self._compile_model()

    def _compile_model(self):
        """
        Compiles the built model with an Adam optimizer and Mean Squared Error loss.
        """
        # setting learning rate decay
        lr_schedule = ExponentialDecay(0.01, 1, 0.99)
        adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(optimizer=adam, loss='mse', metrics='mae')

    def create_sequence(self, data, sequence_length, sequence_type):
        """
        Creates input sequences and target sequences based on the specified sequence type.

        Parameters:
            data (pd.DataFrame): The input data.
            sequence_length (int): The length of the input sequences.
            sequence_type (str): The type of sequence, either 'sliding_window' or 'padding'.

        Returns:
            X (np.array): Input sequences.
            y (np.array): Target sequences.
        """

        X, y = [], []

        if sequence_type == 'sliding_window':
            for _, group in data.groupby('id'):
                # to have the same sequence length, we are shifting through a sequence of data points with a fixed-size window
                # since we split data into train and test beforehand, we are applying it separately to each set
                for i in range(len(group) - sequence_length + 1):
                    X.append(group.iloc[i:i + sequence_length, :-1])

                    if self.model_type == 'm_to_m':
                        y.append(group.iloc[i:i + sequence_length, -1])
                    elif self.model_type == 'm_to_1':
                        y.append(group.iloc[i + sequence_length - 1, -1])

        elif sequence_type == 'padding' and self.model_type == 'm_to_m':
            # we will treat each id as sequence
            for _, group in data.groupby('id'):
                X.append(group.iloc[:, 0:-1])
                y.append(group.iloc[:, -1])
            # Use TensorFlow to pad the sequences
            X = pad_sequences(X, maxlen=sequence_length, padding='post')
            y = pad_sequences(y, maxlen=sequence_length, padding='post')

        else:
            raise ValueError('Please use possible combinations')

        return np.array(X), np.array(y)

    @staticmethod
    def _determine_sequence_length(data, sequence_type):
        """
        Determines the appropriate sequence length based on the sequence type.

        Parameters:
            data (pd.DataFrame): The input data.
            sequence_type (str): The type of sequence, either 'sliding_window' or 'padding'.

        Returns:
            int: The determined sequence length.
        """
        if sequence_type == 'sliding_window':
            return data.groupby('id').count().cycle.min()
        elif sequence_type == 'padding':
            return data.cycle.max()

    def prepare_data(self, train_data, test_data, sequence_type):
        """
        Prepares training and testing data based on the specified sequence type.

        Parameters:
            train_data (pd.DataFrame): The training data.
            test_data (pd.DataFrame): The testing data.
            sequence_type (str): The type of sequence, either 'sliding_window' or 'padding'.
        """

        sequence_length = self._determine_sequence_length(train_data, sequence_type)

        self.X_train, self.y_train = self.create_sequence(train_data, sequence_length, sequence_type)
        self.X_test, self.y_test = self.create_sequence(test_data, sequence_length, sequence_type)

    def train_model(self, epochs=50, batch_size=128):
        """
        Builds, compiles, and trains the LSTM model with specified epochs and batch size.

        Parameters:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
        """
        self.build_lstm_model()

        self.history = self.model.fit(
            self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(self.X_test, self.y_test), callbacks=[type(self).early_stopping]
        )

    def plot_training_history(self):
        """
        Plots the training and validation performance over epochs.
        """
        if self.history is not None:
            plt.figure(figsize=(12, 6))

            # Plot training & validation loss values
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper right')

            # Plot training & validation MAE values
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['mae'])
            plt.plot(self.history.history['val_mae'])
            plt.title('Mean Absolute Error (MAE)')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper right')

            plt.tight_layout()
            plt.show()
