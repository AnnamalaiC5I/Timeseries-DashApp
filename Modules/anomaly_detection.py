import tensorflow as tf
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

class OutlierDetection:
    def __init__(self):
        self.iforest = None
        self.autoencoder = None

    def fit_isolation_forest(self, X, n_estimators=100, contamination=0.1):
        self.iforest = IsolationForest(n_estimators=n_estimators, contamination=contamination)
        self.iforest.fit(X.values.reshape(-1, 1))

    def predict_isolation_forest(self, X):
        if self.iforest is None:
            raise RuntimeError("Isolation Forest model not fitted. Call fit_isolation_forest() first.")
        return self.iforest.predict(X.values.reshape(-1, 1))

    def fit_autoencoder(self, X, encoding_dim=32, epochs=50, batch_size=64):
        if isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)
        else:
            X = X.values
        input_dim = X.shape[1]

        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoded = tf.keras.layers.Dense(input_dim, activation='linear')(encoded)
        self.autoencoder = tf.keras.models.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True)

    def predict_autoencoder(self, X):
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder model not fitted. Call fit_autoencoder() first.")
        X = X.values.reshape(-1, 1)
        predicted = self.autoencoder.predict(X)
        mse = np.mean(np.power(X - predicted, 2), axis=1)
        return mse

    def detect_outliers_autoencoder(self, X, threshold):
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder model not fitted. Call fit_autoencoder() first.")
        mse = self.predict_autoencoder(X)
        anomalies = X[mse > threshold]
        return anomalies

    def detect_outliers_isolation_forest(self, X):
        if self.iforest is None:
            raise RuntimeError("Isolation Forest model not fitted. Call fit_isolation_forest() first.")
        predictions = self.predict_isolation_forest(X)
        anomalies = X[predictions == -1]
        return anomalies

    def handle_outliers(self, X, column):
        X[column] = np.where(X[column] == -1, np.median(X[column]), X[column])
        return X
