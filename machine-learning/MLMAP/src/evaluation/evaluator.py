import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from src.processing.data_extraction import DataExtractor
from src.processing.data_training import DataPreprocessor


class ReportAnalyzer:
    def __init__(self, rf_model_path, xgb_model_path, lstm_model_path, reports_dir, results_dir):
        try:
            self.rf_model = joblib.load(rf_model_path)
            self.xgb_model = joblib.load(xgb_model_path)
            self.lstm_model = load_model(lstm_model_path)
        except Exception as e:
            raise ValueError(f"Error loading models: {e}")

        self.reports_dir = reports_dir
        self.results_dir = results_dir
        self.data_extractor = DataExtractor(reports_dir, results_dir)

        if hasattr(self.rf_model, 'feature_names_in_'):
            self.feature_names = self.rf_model.feature_names_in_
        else:
            raise AttributeError("Random Forest model does not have feature names.")

        self.results = []
        self.probabilities = []
        self.labels = [] 

    def analyze_reports(self):
        if os.path.isdir(self.reports_dir):
            report_files = [f for f in os.listdir(self.reports_dir) if f.endswith(".json")]
            print(f"Found {len(report_files)} report files to analyze.")
            for filename in report_files:
                report_file = os.path.join(self.reports_dir, filename)
                self.analyze_report(report_file)
        elif os.path.isfile(self.reports_dir):
            print(f"Analyzing single report file: {self.reports_dir}")
            self.analyze_report(self.reports_dir)
        else:
            print(f"The specified path {self.reports_dir} is neither a directory nor a valid file.")

        if not self.probabilities:
            print("No probabilities available for plotting.")
        else:
            self.plot_model(self.reports_dir)

    def analyze_report(self, report_file):
        report_data = self.load_report(report_file)
        features = self.extract_features(report_data)

        features_df = pd.DataFrame([features])

        preprocessor = DataPreprocessor(features_df)
        transformed_data = preprocessor.preprocess()

        all_feature_names = self.rf_model.feature_names_in_
        required_features = list(all_feature_names)

        missing_features = pd.DataFrame(0, index=transformed_data.index,
                                        columns=[f for f in required_features if f not in transformed_data.columns])
        transformed_data = pd.concat([transformed_data, missing_features], axis=1)

        transformed_data = transformed_data[all_feature_names]

        if transformed_data.empty:
            print(f"No features extracted from {report_file}. Skipping analysis.")
            return

        predictions = self.predict(transformed_data)

        self.probabilities.append([predictions["rf"][0], predictions["xgb"][0], predictions["lstm"][0]])

        score = self.calculate_score(predictions)
        label = 'Malicious' if score > 0.5 else 'Clean'
        self.labels.append(report_data.get("malstatus", "Suspicious"))

        self.results.append({
            "report_name": os.path.basename(report_file),
            "label": label,
            "score": score,
            "malware_name": self.classify_malware(report_data),
            "predictions": predictions
        })

        self.export_report_results(report_file, label, score, predictions)

    @staticmethod
    def load_report(report_file):
        with open(report_file, 'r') as file:
            data = json.load(file)

        if isinstance(data, list) and data:
            data = data[0]

        if not isinstance(data, dict):
            raise ValueError("Loaded data is not a valid report dictionary.")

        return data

    @staticmethod
    def extract_features(report_data):
        features = {}

        # Basic file information
        target_info = report_data.get("target", {}).get("file", {})
        features["filename"] = target_info.get("name", "Unknown")
        features["size"] = target_info.get("size", 0) if target_info.get("size") is not None else 0
        features["md5"] = target_info.get("md5", "")
        features["sha1"] = target_info.get("sha1", "")
        features["sha256"] = target_info.get("sha256", "")
        features["ssdeep"] = target_info.get("ssdeep", "No-ssdeep")
        features["type"] = target_info.get("type", "")

        # String information (yara, cape_yara)
        yara_info = target_info.get("yara", [])
        features["yara_names"] = " ".join([y["name"] for y in yara_info])
        features["yara_descriptions"] = ", ".join(
            [y.get("meta", {}).get("description", "unknown") for y in yara_info])
        features["yara_strings"] = ", ".join(
            [s for y in yara_info for s in y.get("strings", [])])

        cape_yara_info = target_info.get("cape_yara", [])
        features["cape_yara_names"] = " ".join([y["name"] for y in cape_yara_info])
        features["cape_yara_descriptions"] = ", ".join(
            [cape.get("meta", {}).get("description", "unknown") for cape in cape_yara_info])
        features["cape_yara_strings"] = ", ".join(
            [s for cape in cape_yara_info for s in cape.get("strings", [])])

        # PE information
        pe_info = target_info.get("pe", {})
        features["pe_digital_signers"] = ", ".join(pe_info.get("digital_signers", []))
        features["pe_imagebase"] = pe_info.get("imagebase", "0")
        features["pe_entrypoint"] = pe_info.get("entrypoint", "0")
        features["pe_ep_bytes"] = pe_info.get("ep_bytes", "N/A")
        features["pe_reported_checksum"] = pe_info.get("reported_checksum", "0")
        features["pe_actual_checksum"] = pe_info.get("actual_checksum", "0")
        features["pe_exported_dll_name"] = pe_info.get("exported_dll_name", "N/A")
        features["pe_imported_dll_count"] = pe_info.get("imported_dll_count", 0)
        features["sections"] = len(pe_info.get("sections", []))
        features["pe_overlay_offset"] = pe_info.get("overlay", {}).get("offset", "N/A")
        features["pe_overlay_size"] = pe_info.get("overlay", {}).get("size", "N/A")
        features["pe_entropy"] = ", ".join(
            [str(section.get("entropy", "N/A")) for section in pe_info.get("sections", [])])
        features["pe_imports"] = ", ".join([f"{imp['name']} at {imp['address']}" for imp in
                                            pe_info.get("imports", {}).get("mscoree", {}).get("imports", [])])
        features["pe_exports"] = ", ".join([exp["name"] for exp in pe_info.get("exports", [])])
        features["custom_strings"] = ", ".join(target_info.get("strings", []))

        # Rating information
        features["malstatus"] = report_data.get("malstatus", "Suspicious")
        features["malscore"] = report_data.get("malscore", 0)

        # Behavior and process information
        behav_info = report_data.get("behavior", {})
        features["behavior_process_count"] = len(behav_info.get("processes", []))
        features["call_api"] = ", ".join(call.get("api", "") for process in
                                         behav_info.get("processes", [])
                                         for call in process.get("calls", []))
        features["call_repeat"] = [call.get("repeated", 0) for process in
                                   behav_info.get("processes", [])
                                   for call in process.get("calls", [])]

        # Signature information
        signatures = report_data.get("signatures", [])
        features["sig_name"] = " ".join([sig.get("name", "") for sig in signatures])
        features["sig_description"] = " ".join([sig.get("description", "") for sig in signatures])

        return features

    def predict(self, transformed_data):
        rf_prediction = self.rf_model.predict_proba(transformed_data)[:, 1]
        xgb_prediction = self.xgb_model.predict_proba(transformed_data)[:, 1]
        lstm_prediction = self.lstm_model.predict(transformed_data).flatten()

        return {
            "rf": rf_prediction,
            "xgb": xgb_prediction,
            "lstm": lstm_prediction
        }

    @staticmethod
    def calculate_score(predictions):
        rf_score = np.mean(predictions["rf"])
        xgb_score = np.mean(predictions["xgb"])
        lstm_score = np.mean(predictions["lstm"])

        score = round((rf_score + xgb_score + lstm_score) / 3 * 10, 1)  # Làm tròn đến 1 chữ số thập phân
        return score

    @staticmethod
    def classify_malware(report_data):
        return report_data.get("target", {}).get("file", {}).get("name", "Unknown")

    def plot_model(self, report_file):
        scores = np.array(self.probabilities)
        scores = np.nan_to_num(scores, nan=1)

        labels = ['Random Forest', 'XGBoost', 'LSTM']
        colors = ['green', 'yellow', 'red']

        avg_scores = np.mean(scores, axis=0)

        true_labels = [1 if label == "Malicious" else 0 for label in self.labels]
        pred_labels = [1 if result["label"] == "Malicious" else 0 for result in self.results]
        cm = confusion_matrix(true_labels, pred_labels, labels=[1, 0])

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax[0],
                    xticklabels=['Malicious', 'Clean'], yticklabels=['Malicious', 'Clean'])
        ax[0].set_title('Confusion Matrix')
        ax[0].set_xlabel('Predicted Labels')
        ax[0].set_ylabel('True Labels')

        ax[1].bar(labels, avg_scores, color=colors)
        ax[1].set_ylabel('Average Probability')
        ax[1].set_title('Model Performance Comparison')
        ax[1].set_ylim([0, 1])

        for i, score in enumerate(avg_scores):
            ax[1].text(i, score + 0.02, f"{score:.2f}", ha='center', color='black', fontweight='bold')

        plt.tight_layout()
        report_name = os.path.basename(report_file).replace('.json', '')
        plot_file_name = os.path.join(self.results_dir, f"{report_name}_plot_model.png")
        plt.savefig(plot_file_name)
        plt.show()

    def export_report_results(self, report_file, label, score, predictions):
        predictions_serializable = {model: pred.tolist() if isinstance(pred, np.ndarray) else pred for model, pred in
                                    predictions.items()}

        rf_prob = predictions_serializable['rf'][0] if isinstance(predictions_serializable['rf'], list) else \
        predictions_serializable['rf']
        xgb_prob = predictions_serializable['xgb'][0] if isinstance(predictions_serializable['xgb'], list) else \
        predictions_serializable['xgb']
        lstm_prob = predictions_serializable['lstm'][0] if isinstance(predictions_serializable['lstm'], list) else \
        predictions_serializable['lstm']
        hybrid = round((rf_prob + xgb_prob + lstm_prob) / 3, 4)

        report_name = os.path.basename(report_file).replace('.json', '') + '_results'
        file_name = os.path.basename(report_file)
        results_file = os.path.join(self.results_dir, f"{report_name}.txt")

        os.makedirs(self.results_dir, exist_ok=True)

        report_data = self.load_report(report_file)

        with open(results_file, 'w') as file:
            file.write(f"Report Name: {file_name}\n")
            file.write(f"File: {self.classify_malware(report_data)}\n")
            file.write(f"Classification: {label}\n")
            file.write(f"Score (0-10): {score}\n")
            file.write(f"RF Prediction Probability: {rf_prob:.4f}\n")
            file.write(f"XGB Prediction Probability: {xgb_prob:.4f}\n")
            file.write(f"LSTM Prediction Probability: {lstm_prob:.4f}\n")
            file.write(f"Hybrid Prediction: {hybrid}\n")
