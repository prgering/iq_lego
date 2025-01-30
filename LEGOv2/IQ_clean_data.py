""" 
Loading and Cleaning LEGO Database

"""

# Imports
import pandas as pd
from pathlib import Path
import re
import csv

# Functions
def get_headers(txt_file, columns_to_add):
    """
    Function to extract column_names from readme.txt file and combine with
    those column names not included in that section of the .txt file but
    specified by Schmitt et al. (2011) or obvious from examination of csv file. 
    """

    with open(txt_file, 'r', encoding="latin1") as f:
        lines = f.readlines()
    column_names = []
    for line_number in range(66, 75):  
        line = lines[line_number].strip().replace(",", "").replace("#", "Sum").replace("%", "Percent")
        words = line.split()
        for name in words:
            match = re.search(r'\((.*?)\)', name)
            if match:
                bracket_contents = match.group(1)
                modified_string = re.sub(r'\(.*?\)', '', name).strip()
                column_names.append(f"Context{bracket_contents}{modified_string}")
            else:
                column_names.append(name)
    column_names.insert(0, columns_to_add[0])
    for columns in columns_to_add[1:]:
        column_names.append(columns)
    return column_names

def drop_shifted_rows(data, columns):
    """
    Reading the csv resulted in a few rows recording data in the wrong cells, 
    due to the reliance on a delimiter. This function removes the rows with
    these problems, but it doesn't remove all the data from a single recording
    """

    for column in columns:
        numerical_rows = data[data[column].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
        data = data.drop(numerical_rows.index)
    return data

def read_clean_write_data(data, column_names, removal_columns, dummy_columns, new_name):
    """
    Function to read and clean the data from Schmitt et al. (2011)
    """
    with open(data, 'r', encoding='latin1') as file:
        try:
            dialect = csv.Sniffer().sniff(file.read(1024))
            delimiter = dialect.delimiter
            print(f"Detected delimiter: {delimiter}")
        except csv.Error:
            delimiter = ';' 
            print(f"Default delimiter used: {delimiter}")
    df = pd.read_csv(data, encoding='latin1', header=None, delimiter=delimiter)
    df.columns = column_names
    df = drop_shifted_rows(df, columns = ["LoopName", "ExMo"])
    null_values = ["", "nan","NA", "null", "None", "\\N"]
    df = df.replace(null_values, pd.NA, regex=False)
    df = df.dropna(subset=["IQAverage"])
    df_dropped = df.drop(removal_columns, axis=1)
    df_dummy = pd.get_dummies(data=df_dropped, columns=dummy_columns)
    df_dummy.to_csv(new_name, index = False, header = True) 
    return df_dummy

# Code
if __name__ == "__main__":
    base_path = Path("/home/paulgering/Documents/PhD/multimodal_data/iq_lego/LEGOv2/corpus")
    IQ_file = base_path / "csv/interactions.csv"
    New_file = base_path / "csv/clean_interactions.csv"
    Read_me = base_path.parent / "readme.txt"    
    additional_col = ["FileCode", "WavFile", "EmotionState", "IQ1", "IQ2", "IQ3", "IQAverage"]
    removal_columns = ['Prompt', 'Utterance', 'ASRRecognitionStatus', 'Modality', 'SemanticParse', 'Activity','SystemDialogueAct', 'UserDialogueAct', 'WavFile', 'EmotionState']
    dummy_columns = ["ExMo", "ActivityType", "RoleName", "LoopName"]

    column_names = get_headers(Read_me, additional_col)
    cleaned_df = read_clean_write_data(IQ_file, column_names, removal_columns, dummy_columns, New_file)

    # Check the dataframe for any missing data or errors
    for column in cleaned_df.columns:
        print(f"\nColumn: {column}")
        print(cleaned_df[column].unique()) 
    

    
