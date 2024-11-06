import os
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExtractor:
    def __init__(self, dataset_dir: str = "../..", output_dir: str = "../.."):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

    def load_cuckoo_dataset(self):
        cuckoo_data = []
        path = os.path.join(self.dataset_dir, "cuckoo")
        if os.path.exists(path):
            for filename in os.listdir(path):
                if filename.endswith(".json"):
                    with open(os.path.join(path, filename), "r") as f:
                        data = json.load(f)
                        cuckoo_data.extend(data)
            if not cuckoo_data:
                logger.warning("No Cuckoo data found in the 'cuckoo' directory.")
        else:
            logger.warning(f"Directory '{path}' does not exist.")
        return cuckoo_data

    def extract_cuckoo_data(self, cuckoo_data, output_filename):
        records = []
        for report in cuckoo_data:
            if isinstance(report, dict):
                content = report.get("content", {})
                target = content.get("target", {}).get("file", {})
                filename = report.get("filename")

                target_pe = report.get("target")

                overlay_offset = "N/A"
                overlay_size = "N/A"
                if target_pe and "pe" in target_pe:
                    overlay_offset = target_pe["pe"].get("overlay", {}).get("offset", "N/A")
                    overlay_size = target_pe["pe"].get("overlay", {}).get("size", "N/A")

                cape_yara_names = ", ".join(cape['name'] for cape in target.get('cape_yara', []))
                cape_yara_descriptions = ", ".join(
                    cape.get('meta', {}).get('description', 'unknown') for cape in target.get('cape_yara', []))
                cape_yara_strings = ", ".join(
                    string for cape in target.get('cape_yara', []) for string in cape.get('strings', []))

                static_features = {
                    "filename": target.get("name", "Unknown"),
                    "size": target.get("size", 0),
                    "md5": target.get("md5", "No-hashed"),
                    "sha1": target.get("sha1", "No-hashed"),
                    "sha256": target.get("sha256", "No-hashed"),
                    "ssdeep": target.get("ssdeep", "No-ssdeep"),
                    "type": target.get("type", "unknown"),
                    "yara_names": ", ".join([yara['name'] for yara in target.get('yara', [])]),
                    "yara_descriptions": ", ".join(
                        [yara.get('meta', {}).get('description', 'unknown') for yara in target.get('yara', [])]),
                    "yara_strings": ", ".join(
                        [yara_string for yara in target.get('yara', []) for yara_string in yara.get('strings', [])]),
                    "cape_yara_names": cape_yara_names, 
                    "cape_yara_descriptions": cape_yara_descriptions, 
                    "cape_yara_strings": cape_yara_strings, 
                    "pe_digital_signers": ", ".join(target.get("pe", {}).get("digital_signers", [])),
                    "pe_imagebase": target.get("pe", {}).get("imagebase", "N/A"),
                    "pe_entrypoint": target.get("pe", {}).get("entrypoint", "N/A"),
                    "pe_ep_bytes": target.get("pe", {}).get("ep_bytes", "N/A"),
                    "pe_reported_checksum": target.get("pe", {}).get("reported_checksum", "N/A"),
                    "pe_actual_checksum": target.get("pe", {}).get("actual_checksum", "N/A"),
                    "pe_exported_dll_name": target.get("pe", {}).get("exported_dll_name", "N/A"),
                    "pe_imported_dll_count": target.get("pe", {}).get("imported_dll_count", "N/A"),
                    "pe_sections": ", ".join([section['name'] for section in target.get("pe", {}).get("sections", [])]),
                    "pe_overlay_offset": overlay_offset,
                    "pe_overlay_size": overlay_size,
                    "pe_entropy": ", ".join([section.get('entropy', 'N/A') for section in target.get("pe", {}).get("sections", [])]),
                    "pe_imports": ", ".join([f"{imp['name']} at {imp['address']}" for imp in
                                              target.get("pe", {}).get("imports", {}).get("mscoree", {}).get("imports", [])]),
                    "pe_exports": ", ".join([exp['name'] for exp in target.get("pe", {}).get("exports", [])]),
                    "custom_strings": ", ".join(target.get("strings", [])),
                }

                dynamic_features = {
                    "malstatus": content.get("malstatus", "unknown"),
                    "malscore": content.get("malscore", 0),
                    "behavior_process_count": len(content.get("behavior", {}).get("processes", [])),
                    "call_api": ", ".join(call.get('api', '') for process in content.get("behavior", {}).get("processes", []) for call in process.get("calls", [])),
                    "call_repeat": [call.get('repeated', 0) for process in
                                    content.get("behavior", {}).get("processes", [])
                                    for call in process.get("calls", [])],
                    "sig_name": ", ".join(signame['name'] for signame in content.get("signatures", [])),
                    "sig_description": ", ".join(sigdesc['description'] for sigdesc in content.get("signatures", [])),
                    "sig_severity": [sigsev['severity'] for sigsev in content.get("signatures", [])],
                    "sig_confidence": [sigconf['confidence'] for sigconf in content.get("signatures", [])],
                    "ttp_signatures": ", ".join(ttp['signature'] for ttp in content.get("ttps", []))
                }

                label = 1 if content.get("malstatus", "").lower() == "malicious" else 0
                record = {"filename": filename, "label": label}
                record.update(static_features)
                record.update(dynamic_features)
                records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"DataFrame columns after extraction: {df.columns.tolist()}")

        # Print extracted data
        # print(df)

        logger.info(f"Output directory: {self.output_dir}")

        output_file_path = os.path.join(self.output_dir, f"{output_filename}.csv")
        logger.info(f"Saving to: {output_file_path}")  # In ra đường dẫn sẽ lưu file

        df.to_csv(output_file_path, index=False, encoding='utf-8')
        logger.info(f"Dataset extracted and saved into file: {output_file_path}")

        return df

    @staticmethod
    def replace_none_with_zero(data):
        if isinstance(data, pd.DataFrame):
            return data.fillna(0)
        elif isinstance(data, dict):
            return {k: (v if v is not None else 0) for k, v in data.items()}
        return data
