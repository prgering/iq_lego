import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from svm_smo.src.smo_optimizer import SVM
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
    feature_types["DM"].append(df.filter(regex='WPST|DD|RoleIndex|Prompt|Exchange|Turns|SystemQuestions|Activity|Rolename|LoopName').columns)
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
    kf = KFold(n_splits=k, shuffle=True,random_state=42)
    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        clf = SVM(kernel_type='linear')
        clf.fit(X_train, y_train)
        y_pred, _ = clf.predict(X_val)
        macro_recall = recall_score(y_val, y_pred, average='macro')
        macro_recall_scores.append(macro_recall)
        print(f"Fold {i} Macro-Recall: {macro_recall}")

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


