import os
import joblib
import pandas as pd
import numpy as np
import hashlib

from keras.src.callbacks import EarlyStopping
from keras.src.layers import BatchNormalization
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM


class ModelTrainer:
    def __init__(self, file_path):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.file_path = os.path.join(os.path.dirname(__file__), file_path)
        self.data = pd.read_csv(self.file_path)

        self.columns_to_drop = ['label', 'filename']
        self.features = self.data.drop(columns=self.columns_to_drop, errors='ignore')
        self.labels = self.data['label']

        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()
        self.data_fingerprint = self._generate_data_fingerprint()

    def _split_data(self):
        return train_test_split(self.features, self.labels, test_size=0.2, random_state=42)

    def _generate_data_fingerprint(self):
        data_str = pd.util.hash_pandas_object(self.features).sum()
        return hashlib.md5(str(data_str).encode()).hexdigest()

    def _is_data_new(self, model_path, fingerprint_path):
        if os.path.exists(model_path) and os.path.exists(fingerprint_path):
            with open(fingerprint_path, 'r') as f:
                saved_fingerprint = f.read()
            if saved_fingerprint == self.data_fingerprint:
                return False  
        return True 

    def _save_data_fingerprint(self, fingerprint_path):
        with open(fingerprint_path, 'w') as f:
            f.write(self.data_fingerprint)

    @staticmethod
    def _calculate_checksum(model_path):
        with open(model_path, 'rb') as f:
            model_data = f.read()
        return hashlib.md5(model_data).hexdigest()

    @staticmethod
    def save_model_checksums(rf_checksum, xgb_checksum, lstm_checksum):
        checksums_path = '../../models/stored/model_checksums.txt'
        with open(checksums_path, 'w') as f:
            f.write(f"Random Forest Checksum: {rf_checksum}\n")
            f.write(f"XGBoost Checksum: {xgb_checksum}\n")
            f.write(f"LSTM Checksum: {lstm_checksum}\n")

    def train_random_forest(self):
        model_path = '../../models/stored/rf_model.pkl'
        fingerprint_path = '../../models/stored/rf_model_fingerprint.txt'

        if not self._is_data_new(model_path, fingerprint_path):
            print("Random Forest: No new data to add! Loading existing model results...")
            rf_model = self.load_random_forest()
            rf_predictions = rf_model.predict(self.X_test)
            rf_accuracy = accuracy_score(self.y_test, rf_predictions)
            rf_classification_report = classification_report(self.y_test, rf_predictions, zero_division=1)
            rf_probabilities = rf_model.predict_proba(self.X_test)[:, 1] 

            rf_checksum = self._calculate_checksum(model_path)
            return rf_accuracy, rf_classification_report, rf_probabilities.tolist(), rf_checksum

        print("Training Random Forest with new data...")
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(self.X_train, self.y_train)

        rf_predictions = rf_model.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_predictions)
        rf_classification_report = classification_report(self.y_test, rf_predictions, zero_division=1)
        rf_probabilities = rf_model.predict_proba(self.X_test)[:, 1]  

        joblib.dump(rf_model, model_path)
        self._save_data_fingerprint(fingerprint_path)
        rf_checksum = self._calculate_checksum(model_path)
        print("Random Forest training completed and model updated with new data.")

        return rf_accuracy, rf_classification_report, rf_probabilities.tolist(), rf_checksum

    def train_xgboost(self):
        model_path = '../../models/stored/xgb_model.pkl'
        fingerprint_path = '../../models/stored/xgb_model_fingerprint.txt'

        if not self._is_data_new(model_path, fingerprint_path):
            print("XGBoost: No new data to add! Loading existing model results...")
            xgb_model = self.load_xgboost()
            xgb_predictions = xgb_model.predict(self.X_test)
            xgb_accuracy = accuracy_score(self.y_test, xgb_predictions)
            xgb_classification_report = classification_report(self.y_test, xgb_predictions, zero_division=1)
            xgb_probabilities = xgb_model.predict_proba(self.X_test)[:, 1]
            xgb_checksum = self._calculate_checksum(model_path)
            return xgb_accuracy, xgb_classification_report, xgb_probabilities.tolist(), xgb_checksum

        print("Training XGBoost with new data...")
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(self.X_train, self.y_train)

        xgb_predictions = xgb_model.predict(self.X_test)
        xgb_accuracy = accuracy_score(self.y_test, xgb_predictions)
        xgb_classification_report = classification_report(self.y_test, xgb_predictions, zero_division=1)
        xgb_probabilities = xgb_model.predict_proba(self.X_test)[:, 1]

        joblib.dump(xgb_model, model_path)
        self._save_data_fingerprint(fingerprint_path)
        xgb_checksum = self._calculate_checksum(model_path)
        print("XGBoost training completed and model updated with new data.")

        return xgb_accuracy, xgb_classification_report, xgb_probabilities.tolist(), xgb_checksum

    def train_lstm(self):
        model_path = '../../models/stored/lstm_model.keras'
        fingerprint_path = '../../models/stored/lstm_model_fingerprint.txt'

        if not self._is_data_new(model_path, fingerprint_path):
            print("LSTM: No new data to add! Loading existing model results...")
            lstm_model = self.load_lstm()
            X_test_seq = np.expand_dims(pad_sequences(self.X_test.values), axis=-1)
            lstm_probabilities = lstm_model.predict(X_test_seq).flatten()
            lstm_predictions = (lstm_probabilities > 0.5).astype("int32")
            lstm_accuracy = accuracy_score(self.y_test, lstm_predictions)
            lstm_classification_report = classification_report(self.y_test, lstm_predictions, zero_division=1)
            lstm_checksum = self._calculate_checksum(model_path)
            return lstm_accuracy, lstm_classification_report, lstm_probabilities.tolist(), lstm_checksum

        print("Training LSTM with new data...")
        X_train_seq = np.expand_dims(pad_sequences(self.X_train.values), axis=-1)
        X_test_seq = np.expand_dims(pad_sequences(self.X_test.values), axis=-1)

        X_train_seq = X_train_seq / np.max(X_train_seq)
        X_test_seq = X_test_seq / np.max(X_test_seq)

        model = Sequential([
            Input(shape=(X_train_seq.shape[1], 1)),
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.5),
            LSTM(32),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(X_train_seq, self.y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping],
                  verbose=1)

        lstm_probabilities = model.predict(X_test_seq).flatten()
        lstm_predictions = (lstm_probabilities > 0.5).astype("int32")
        lstm_accuracy = accuracy_score(self.y_test, lstm_predictions)
        lstm_classification_report = classification_report(self.y_test, lstm_predictions, zero_division=1)

        model.save(model_path)
        self._save_data_fingerprint(fingerprint_path)
        lstm_checksum = self._calculate_checksum(model_path)
        print("LSTM training completed and model updated with new data.")

        return lstm_accuracy, lstm_classification_report, lstm_probabilities.tolist(), lstm_checksum

    @staticmethod
    def load_random_forest():
        return joblib.load('../../models/stored/rf_model.pkl')

    @staticmethod
    def load_xgboost():
        return joblib.load('../../models/stored/xgb_model.pkl')

    @staticmethod
    def load_lstm():
        return load_model('../../models/stored/lstm_model.keras')

    @staticmethod
    def save_results(rf_results, xgb_results, lstm_results):
        results_path = '../../models/evaluate'
        os.makedirs(results_path, exist_ok=True)
        results_file = os.path.join(results_path, 'model_results.txt')

        with open(results_file, "w") as f:
            f.write(f"Random Forest Results:\n")
            f.write(f"Accuracy: {rf_results[0]}\n")
            f.write(f"Classification Report:\n{rf_results[1]}\n")
            f.write(f"Probabilities: {rf_results[2]}\n")
            f.write(f"Checksum: {rf_results[3]}\n\n")

            f.write(f"XGBoost Results:\n")
            f.write(f"Accuracy: {xgb_results[0]}\n")
            f.write(f"Classification Report:\n{xgb_results[1]}\n")
            f.write(f"Probabilities: {xgb_results[2]}\n")
            f.write(f"Checksum: {xgb_results[3]}\n\n")

            f.write(f"LSTM Results:\n")
            f.write(f"Accuracy: {lstm_results[0]}\n")
            f.write(f"Classification Report:\n{lstm_results[1]}\n")
            f.write(f"Probabilities: {lstm_results[2]}\n")
            f.write(f"Checksum: {lstm_results[3]}\n")

