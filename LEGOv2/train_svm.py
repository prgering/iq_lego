import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score


def train_opt_split(df):
    x = df.drop("IQAverage", axis=1)
    y = df["IQAverage"]
    return train_test_split(x, y, random_state=42, train_size = .60)

def feature_dict(df):
    feature_types = {}
    keys = ["ASR", "SLU", "DM", "DAcT", "EMO"]
    for key in keys:
        feature_types[key] = []
    
    feature_types["ASR"].append(df.filter(regex='ASR|TimeOut|Barge|UTD|ExMo|WPUT|Modality').columns)
    feature_types["DM"].append(df.filter(regex='WPST|DD|RoleIndex|^Prompt$|Exchange|Turns|SystemQuestions|Activity|RoleName|LoopName|RePrompt').columns)
    feature_types["SLU"].append(df.filter(regex='present').columns)
    feature_types["DAcT"].append(df.filter(regex='DialogueAct').columns)
    feature_types["EMO"].append(df.filter(regex='EmotionState').columns)


    return feature_types

def split_features(df, feature_types):
    dfs = {}
    for key, columns_list in feature_types.items():
        for columns in columns_list:
            column_names = columns.tolist()  
            dfs[key] = df[column_names]
    df_AUTO = pd.concat([dfs["ASR"], dfs["SLU"], dfs["DM"]], axis=1, join='outer')
    df_AUTOEMO = df.drop(dfs["EMO"].columns, axis=1)
    breakpoint()

    return dfs, df_AUTO, df_AUTOEMO

def train_svm(X, y):
    macro_recall_scores = []
    k = 10
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scaler = StandardScaler()

    for i, (train_index, val_index) in enumerate(kf.split(X, y)): # Note: y passed to split
        X_train, X_val = X.iloc[train_index], X.iloc[val_index] # Use .iloc for indexing
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        clf = SVC(kernel='linear')
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_val_scaled)

        try:
            macro_recall = recall_score(y_val, y_pred, average='macro')
        except ValueError:
            print(f"Fold {i}: ValueError in recall_score.")
            macro_recall = np.nan
        macro_recall_scores.append(macro_recall)
        # print(f"Fold {i} Macro-Recall: {macro_recall}")  # Print less frequently

    average_macro_recall = np.mean(macro_recall_scores)
    print(f"Average Macro-Recall across {k} folds: {average_macro_recall}")


    
if __name__ == "__main__":
    base_path = Path("/home/paulgering/Documents/PhD/multimodal_data/iq_lego/LEGOv2/corpus")
    data_file = base_path / "csv/clean_interactions.csv"

    data = pd.read_csv(data_file)

    X_train, X_opt, y_train, y_opt = train_opt_split(data)

    feature_type_dict = feature_dict(data)

    dict_feature_data, auto_data, autoemo_data = split_features(X_train, feature_type_dict)

    for key, dataframe in dict_feature_data.items():
        print(f"Training {key} feature data \n")
        train_svm(dataframe.to_numpy(), y_train.to_numpy())
    
    print("Training AUTO feature data \n")

    train_svm(auto_data.to_numpy(), y_train.to_numpy())

    print("Training AUTOEMO feature data \n")

    train_svm(autoemo_data.to_numpy(), y_train.to_numpy())

    print("Training ALL feature data \n")


