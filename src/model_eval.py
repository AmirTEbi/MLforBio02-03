from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

def eval(X_test, y_test, model):
    """Evaluate the trained model on test set.

    Parameters
    ----------
    X_test : numpy.array
        Test data.
    y_test : numpy.array
        Test labels.
    model : object
        

    Returns
    -------
    metrics : dict
    """

    y_pred = model.predict(X_test)

    metrics = {}
    accuracy = accuracy_score(y_test, y_pred)
    metrics["accuracy"] = accuracy
    percision = precision_score(y_test, y_pred, average='weighted')
    metrics["percision"] = percision
    recall = recall_score(y_test, y_pred, average='weighted')
    metrics["recall"] = recall
    f1 = f1_score(y_test, y_pred, average='weighted')
    metrics["f1"] = f1

    return metrics