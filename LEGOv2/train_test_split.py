import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score

def train_opt_split(df):
    x = df.drop("IQAverage", axis=1)
    y = df["IQAverage"]
    return train_test_split(x, y, random_state=0, train_size = .60)


def train_svm(X, y):
    clf = svm.SVC(kernel='linear', C = 1.0).fit(X_train, y_train)
    scores = cross_val_score(clf, X, y, cv=10, scoring='recall_weighted')
    print(scores)

    
if __name__ == "__main__":
    base_path = Path("/home/paulgering/Documents/PhD/multimodal_data/iq_lego/LEGOv2/corpus")
    data_file = base_path / "csv/clean_interactions.csv"

    data = pd.read_csv(data_file)

    X_train, X_opt, y_train, y_opt = train_opt_split(data)

    train_svm(X_train, y_train)

