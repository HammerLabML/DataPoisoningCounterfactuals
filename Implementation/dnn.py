import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers


class DeepNeuralNetwork():
    def __init__(self, hidden_layer_sizes=(128, 32), activation="relu", solver='adam'):
        self.model = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(ls, activation=activation, kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-2)) for ls in hidden_layer_sizes] +
            [tf.keras.layers.Dense(2)]) # ATTENTION: Assuming a binary classification problem
        self.solver = solver

    def fit(self, X_train, y_train, n_epochs=200, verbose=False):
        # Fit model by minimizing loss function
        self.model.compile(optimizer=self.solver, loss="binary_crossentropy", metrics=[])
        self.model.fit(X_train, y_train, epochs=n_epochs, verbose=verbose)

    def __call__(self, X):
        return self.predict(X)

    def predict_proba(self, X):
        X_ = X
        if not isinstance(X, np.ndarray):
            X_ = np.array(X)

        return tf.nn.softmax(self.model(X_)).numpy()
    
    def predict(self, X):
        X_ = X
        if not isinstance(X, np.ndarray):
            X_ = np.array(X)

        y_pred = self.model(X_)
        return tf.argmax(tf.nn.softmax(y_pred), axis=1).numpy()
