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
from sentence_transformers import SentenceTransformer




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
        line = lines[line_number].strip().replace(",", "").replace("#", "Sum").replace("%", "Percent") # Clean column names
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

def read_data(data):
    """
    Function to read LEGO data from interaction.csv

    CSV file is read into Python using delimiter that is identified
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
    return df

def extract_keys_sem_parse(data):
    """
    Function to identify unique high-level keys in the semantic parse column of csv.
    The function uses the class SemParse
    """
    sem_parser = SemParse()
    entity_counts = [sem_parser.count_entities(entry) for entry in data["SemanticParse"].dropna().tolist()]
    with open('entity_counts.txt', 'w') as file:
        file.write(json.dumps(entity_counts, indent=3, sort_keys=False))
    
    first_keys = []
    for dictionary in entity_counts:
        if dictionary:
            first_key = list(dictionary.keys())[0]
            first_keys.append(first_key)
        else: 
            first_keys.append("")
    unique_first_keys = list(set(first_keys))

    return unique_first_keys 

def extract_word_embeddings(data, column):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    embeddings = data[column].apply(model.encode).to_list()

    embeddings_array = np.array(embeddings)

    embeddings_df = pd.DataFrame(embeddings_array, index=data.index)

    return embeddings_df


def clean_data(df, column_names, dummy_columns, drop_columns):
    """
    The following steps are taken to clean the data:
    1. column headers added
    2. categorical variable labels changed if inconsistent with documentation
    3. null values removed
    4. information from semantic parser column extracted in ML appropriate format
    5. columns not necessary for ML pipeline dropped
    6. categorical variables dummy coded
    """

    df.columns = column_names

    replace_ASR = {'no input': 'timeout', 'no match': 'reject', 'complete': 'success'}
    replace_modality = {'voice': 'speech'}   

    df["ASRRecognitionStatus"] = df["ASRRecognitionStatus"].replace(replace_ASR)
    df["Modality"] = df["Modality"].replace(replace_modality)
     
    null_values = ["", "nan","NA", "null", "None", "\\N"]
    df = df.replace(null_values, pd.NA, regex=False)
    df = df.dropna(subset=["IQAverage"])
    df = df.drop(df.loc[df['SystemDialogueAct'].isin(["SDA_GREETING", "SDA_OFFERHELP"])].index)
    
    sem_keys = extract_keys_sem_parse(df)

    df.loc[df['SemanticParse'] == "Semantic no match", 'SemanticParse'] = "semantic_no_match"

    for key in sem_keys:
        if key:
            df[key + '_present'] = df['SemanticParse'].str.contains(r'\b' + re.escape(key) + r'\b', case=False, regex=True).replace(pd.NA, False)
        else: 
            continue
    
    embed_prompt = extract_word_embeddings(df, "Prompt")
    embed_utterance = extract_word_embeddings(df, "Utterance")
    breakpoint()

    df_dropped = df.drop(drop_columns, axis=1)
    df = df.replace(null_values, pd.NA, regex=False)

    
    df_dummy = pd.get_dummies(data=df_dropped, columns=dummy_columns)

    return df_dummy



# Code
if __name__ == "__main__":
    base_path = Path("/home/paulgering/Documents/PhD/multimodal_data/iq_lego/LEGOv2/corpus")
    IQ_file = base_path / "csv/interactions.csv"
    new_file = base_path / "csv/clean_interactions_lstm.csv"
    read_me = base_path.parent / "readme.txt"    
    add_col = ["FileCode", "WavFile", "EmotionState", "IQ1", "IQ2", "IQ3", "IQAverage"]
    dummy_col = ["ASRRecognitionStatus", "ExMo", "Modality", "Activity", "ActivityType", "RoleName", "LoopName", "SystemDialogueAct", "UserDialogueAct", "EmotionState"]
    drop_col = ['Prompt', 'Utterance', 'SemanticParse', 'WavFile', "IQ1", "IQ2", "IQ3", "HelpRequest?", "SumHelpRequests", "ContextSumHelpRequest", "PercentHelpRequest"]

    column_names = get_headers(read_me, add_col)
    uncleaned_df = read_data(IQ_file)
    cleaned_df = clean_data(uncleaned_df, column_names, dummy_col, drop_col)
    cleaned_df.to_csv(new_file, index = False, header = True)

    for column in cleaned_df.columns:
        print(f"\nColumn: {column}")
        print(cleaned_df[column].unique()) 