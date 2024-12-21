import pandas as pd
from sklearn.model_selection import train_test_split
import json


def read_config(path):
    """Read the config file.

    Parameters
    ----------
    path : str
        Path to the config file.

    Returns
    -------
    config: JASON file

    """

    with open(path, "r") as f:
        config = json.load(f)

    return config

def load_data(config, seed=0):
    """Load data from config file and split to training and test set.

    Parameters
    ----------
    config : JSON file
        A JSON file containing the path to the data and labels
    seed : int
        Random seed for data split (Default value = 0)

    Returns
    -------
    tuple: A tuple of five numpy.arrays (X_train, X_test, y_train, y_test, feature_names)

    """

    X = pd.read_csv(config["data_path"])
    y = pd.read_csv(config["labels_path"])
    feature_names = X.columns
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()


    X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.3, random_state=seed)

    return X_train, X_test, y_train, y_test, feature_names