#imports
from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Functions
def create_grouped_feature_arrays(data, quality_col):
    """
    Groups data by FileCode and creates a dictionary of feature arrays and quality arrays.

    """
    feature_cols = [col for col in data.columns if col not in ['FileCode', quality_col]]
    grouped_data = data.groupby('FileCode')

    max_len = 0 
    for _, group_df in grouped_data:
        max_len = max(max_len, len(group_df))

    
    grouped_data = data.groupby('FileCode')

    result = {}
    for file_code, group_df in grouped_data:
        feature_matrix = np.array([row[feature_cols].tolist() for _, row in group_df.iterrows()])
        IQ_scores = np.array(group_df[quality_col].tolist())
        feature_matrix_padded = pad_sequences(feature_matrix, maxlen=max_len, padding='post', dtype='float32')
        IQ_scores_padded = pad_sequences(IQ_scores.reshape(-1, 1), maxlen=max_len, padding='post', dtype=IQ_scores.dtype).flatten()
        result[file_code] = {'features': feature_matrix_padded, 'IQ': IQ_scores_padded}
    return result

if __name__ == "__main__":
    base_path = Path("/home/paulgering/Documents/PhD/multimodal_data/iq_lego/LEGOv2/corpus")
    data_file = base_path / "csv/clean_interactions_lstm.csv"

    data = pd.read_csv(data_file)

    quality_col = 'IQAverage'
    
    grouped_arrays_padded = create_grouped_feature_arrays(data, quality_col)
    
