""" 
Loading and Cleaning LEGO Database

"""

# Imports
import pandas as pd
import re
import csv
import json
from pathlib import Path
from analyse_sem_pas import SemParse

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

def read_clean_write_data(data, column_names, new_name):
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
    
    replace_ASR = {'no input': 'timeout', 'no match': 'reject', 'complete': 'success'}
    replace_modality = {'voice': 'speech'}   

    df["ASRRecognitionStatus"] = df["ASRRecognitionStatus"].replace(replace_ASR)
    df["Modality"] = df["Modality"].replace(replace_modality)
     

    sem_parser = SemParse()
    entity_counts = [sem_parser.count_entities(entry) for entry in df["SemanticParse"].dropna().tolist()]
    with open('entity_counts.txt', 'w') as file:
        file.write(json.dumps(entity_counts, indent=3, sort_keys=False)) 


    first_keys = []
    for dictionary in entity_counts:
        if dictionary: # Check if the dictionary is not empty
            first_key = list(dictionary.keys())[0]
            first_keys.append(first_key)
        else: # if it is empty append an empty string or something else to keep the same length
            first_keys.append("") # or first_keys.append(None) or anything that you want
    unique_first_keys = list(set(first_keys))
    for key in first_keys:
        if key: #check if the key is not an empty string
            df[key + '_present'] = df['SemanticParse'].str.contains(r'\b' + re.escape(key) + r'\b', case=False, regex=True)
        else: 
            continue

    df.to_csv(new_name, index = False, header = True)

    return df

# Code
if __name__ == "__main__":
    base_path = Path("/home/paulgering/Documents/PhD/multimodal_data/iq_lego/LEGOv2/corpus")
    IQ_file = base_path / "csv/interactions.csv"
    New_file = base_path / "csv/interactions_with_headers.csv"
    Read_me = base_path.parent / "readme.txt"    
    additional_col = ["FileCode", "WavFile", "EmotionState", "IQ1", "IQ2", "IQ3", "IQAverage"]

    column_names = get_headers(Read_me, additional_col)
    cleaned_df = read_clean_write_data(IQ_file, column_names, New_file)
