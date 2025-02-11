"""

Creating Train/Optimisation split and training an SVC

"""

# Imports

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# Functions

def train_opt_split(df):
    x = df.drop("IQAverage", axis=1)
    y = df["IQAverage"]
    return train_test_split(x, y, random_state=42, train_size = .60)


def feature_dict(df):
    """
    Creates dictionary of input features arranged under keys based on the feature type
    """

    feature_types = {}
    keys = ["ASR", "SLU", "DM", "DAcT", "EMO"]
    for key in keys:
        feature_types[key] = []
    
    feature_types["ASR"].append(df.filter(regex='ASR|TimeOut|Barge|UTD|ExMo|WPUT|Modality|grammar_emb|utterance_emb|triggered_grammar').columns)
    feature_types["DM"].append(df.filter(regex='WPST|DD|RoleIndex|^Prompt$|Exchange|Turns|SystemQuestions|Activity|RoleName|LoopName|RePrompt').columns)
    feature_types["SLU"].append(df.filter(regex='present').columns)
    feature_types["DAcT"].append(df.filter(regex='DialogueAct').columns)
    feature_types["EMO"].append(df.filter(regex='EmotionState').columns)


    return feature_types


def split_features(df, feature_types):
    """
    Splits features in a dataframe according to a dictionary of feature types
    """
    dfs = {}
    for key, columns_list in feature_types.items():
        for columns in columns_list:
            column_names = columns.tolist()  
            dfs[key] = df[column_names]
    df_AUTO = pd.concat([dfs["ASR"], dfs["SLU"], dfs["DM"]], axis=1, join='outer')
    df_AUTOEMO = df.drop(dfs["EMO"].columns, axis=1)

    return dfs, df_AUTO, df_AUTOEMO


def dim_reduce(df, n_components = 50, prefix=""):
    """
    Uses Principal Components Analysis to reduce the components of the word embedding features
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df) 
    pca = PCA(n_components=n_components)
    df_reduced = pca.fit_transform(df_scaled) 
    new_cols = [f"{prefix}_{i+1}" for i in range(n_components)] 
    df_with_heads = pd.DataFrame(df_reduced, index=df.index, columns=new_cols)
    return df_with_heads, pca, scaler


def train_svm(X, y):
    """
    Trains an SVC on X to predict y using Stratified 10-fold Cross-Validation
    """
    macro_recall_scores = []
    k = 10
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scaler = StandardScaler()

    for i, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
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
        print(f"Fold {i} Macro-Recall: {macro_recall}")

    average_macro_recall = np.mean(macro_recall_scores)
    print(f"Average Macro-Recall across {k} folds: {average_macro_recall}")
    
    return average_macro_recall


    
if __name__ == "__main__":
    base_path = Path("/home/acp23prg/fastdata/projects/iq_lego/LEGOv2/corpus")
    data_file = base_path / "csv/clean_interactions.csv"

    data = pd.read_csv(data_file)

    X_train, X_opt, y_train, y_opt = train_opt_split(data)

    embedding_prefixes = ["grammar_emb", "utterance_emb", "triggered_grammar"]
    best_recall = 0
    best_n = 0

    # Test different parameter sizes
    for n in [25,50,75,100]:

        # Reduce Dimensions for each embedding variable
        reduced_embeddings = []
        for prefix in embedding_prefixes:
            filtered_cols = [col for col in X_train if col.startswith(prefix)]
            reduced_embeds, pca, scaler = dim_reduce(X_train[filtered_cols], n_components=n, prefix=prefix)
            reduced_embeddings.append(reduced_embeds)

        all_reduced = pd.concat(reduced_embeddings, axis=1)
        X_train_reduced = pd.concat([X_train.drop([col for col in X_train if any(emb in col for emb in embedding_prefixes)], axis=1), all_reduced], axis=1)

        # Create Dictionary of feature types and split data based on feature types
        feature_type_dict = feature_dict(X_train_reduced)
        X_train_dict_reduced, X_train_auto_reduced, X_train_autoemo_reduced = split_features(X_train_reduced, feature_type_dict)

        # Train SVM on different feature sets
        print(f"Training ALL feature data with {n} embedding dimensions\n")
        recall = train_svm(X_train_reduced, y_train)

        if recall > best_recall:
            best_recall = recall
            best_n = n

        for key, dataframe in X_train_dict_reduced.items():
            print(f"Training {key} feature data with {n} embedding dimensions\n")
            recall = train_svm(dataframe, y_train)
            if recall > best_recall:
                best_recall = recall
                best_n = n

        print(f"Training AUTO feature data with {n} embedding dimensions\n")
        recall = train_svm(X_train_auto_reduced, y_train)
        if recall > best_recall:
            best_recall = recall
            best_n = n

        print(f"Training AUTOEMO feature data with {n} embedding dimensions\n")
        recall = train_svm(X_train_autoemo_reduced, y_train)
        if recall > best_recall:
            best_recall = recall
            best_n = n

    print(f"Best recall of {best_recall} was achieved by {best_n} components per embedding type")


