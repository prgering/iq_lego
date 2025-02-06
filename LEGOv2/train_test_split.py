import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def train_opt_split(df, target):
    x = df.drop(target, axis=1)
    y = df[target]
    X_train, X_opt, y_train, y_opt = train_test_split(X, y, random_state=0, train_size = .60)


if __name__ == "__main__":
    base_path = Path("/home/paulgering/Documents/PhD/multimodal_data/iq_lego/LEGOv2/corpus")
    data_file = base_path / "csv/clean_interactions.csv"

    df = pd.read_csv(data_file)
    
    y_columns = ['IQ1', 'IQ2', 'IQ3', 'IQAverage']

    x, y = train_test_split(cleaned_df, y_columns)

