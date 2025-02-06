import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold
from svm_smo.src.smo_optimizer import SVM

def train_opt_split(df):
    x = df.drop("IQAverage", axis=1)
    y = df["IQAverage"]
    return train_test_split(x, y, random_state=42, train_size = .60)


def train_svm(X, y):
    macro_recall_scores = []
    kf = KFold(n_splits=10, shuffle=True,random_state=42)
    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        clf = SVM(kernel_type='linear').fit(X_train, y_train)
        y_pred = model.predict(X_val)
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

    train_svm(X_train.to_numpy(), y_train.to_numpy())

