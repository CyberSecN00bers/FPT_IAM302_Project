import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.label_encoders = {}
        self.word2vec_model = None
        self.tfidf_vectorizer = None

    def fill_missing_values(self):
        self.data = self.data.fillna({
            "size": 0,
            "malscore": 0,
            "behavior_process_count": 0,
            "yara_names": "",
            "yara_descriptions": "",
            "cape_yara": "",
            "process_calls": "",
            "signatures": "",
            "ttps": ""
        })

    def convert_hex_to_int(self, column_name):
        def hex_to_int(hex_value):
            try:
                return int(hex_value, 16)
            except ValueError:
                return 0  

        if column_name in self.data.columns:
            self.data[column_name] = self.data[column_name].apply(hex_to_int)

    def encode_categorical_columns(self):
        categorical_columns = ['type', 'malstatus']
        for col in categorical_columns:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le

    def process_text_features(self):
        text_columns = ['yara_names', 'yara_descriptions', 'cape_yara_names', 'cape_yara_descriptions',
                        'cape_yara_strings', 'yara_strings', 'custom_strings', 'call_api', 'sig_name',
                        'sig_description', 'ttp_signatures']

        existing_columns = [col for col in text_columns if col in self.data.columns]

        if not existing_columns:
            print("No valid text columns found in the DataFrame.")
            return

        self.data[existing_columns] = self.data[existing_columns].fillna("no data")

        sentences = self.data[existing_columns].apply(lambda row: ' '.join(row.values.astype(str)).split(), axis=1)

        if self.word2vec_model is None:
            self.word2vec_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

        for col in existing_columns:
            sentences = self.data[col].apply(lambda x: x.split() if isinstance(x, str) else [])

            def get_avg_word2vec(sentence):
                vectors = [self.word2vec_model.wv[word] for word in sentence if word in self.word2vec_model.wv]
                if not vectors:
                    return np.zeros(self.word2vec_model.vector_size)
                return np.mean(vectors, axis=0)

            word2vec_features = sentences.apply(get_avg_word2vec)
            word2vec_df = pd.DataFrame(
                word2vec_features.tolist(),
                columns=[f'{col}_vec{i}' for i in range(self.word2vec_model.vector_size)]
            )

            self.data = pd.concat([self.data.reset_index(drop=True), word2vec_df.reset_index(drop=True)], axis=1)

        print("Word2Vec features processed and added to the DataFrame.")

    def remove_hash_columns(self):
        hash_columns = ['md5', 'sha1', 'sha256', 'ssdeep']
        self.data.drop(columns=hash_columns, inplace=True, errors='ignore')

    def convert_columns_to_numeric(self):
        for column in self.data.columns:
            self.data[column] = pd.to_numeric(self.data[column], errors='coerce')
            self.data[column] = self.data[column].replace([np.inf, -np.inf], np.nan)  # Gán lại giá trị cho cột
            self.data[column] = self.data[column].fillna(0)  # Gán lại giá trị cho cột

    def normalize_numerical_columns(self):
        numerical_columns = ['size', 'malscore', 'behavior_process_count']
        scaler = StandardScaler()
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])

    def preprocess(self):
        self.fill_missing_values()

        for col in ['pe_imagebase', 'pe_entrypoint']:  
            self.convert_hex_to_int(col)

        self.encode_categorical_columns()
        self.process_text_features()
        self.remove_hash_columns()
        self.convert_columns_to_numeric()
        self.normalize_numerical_columns()

        self.data = self.data.drop(columns=['filename'], errors='ignore')  

        return self.data