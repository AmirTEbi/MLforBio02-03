from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def get_func_name(model_name):
    """Get the name of the training function.

    Parameters
    ----------
    model_name : str
        The name of the model to train.

    Returns
    -------
    The name of the training function: str
    
    """

    if model_name == "logistic_regression":
        return "train_logreg"
    
    elif model_name == "knn":
        return "train_knn"
    
    elif model_name == "svm":
        return "train_svm"
    
    elif model_name == "RF":
        return "train_RF"
    
    elif model_name == "mlp":
        return "train_mlp"
    
    elif model_name == "mix":
        return "train_mix"
    

# Logistic Regression
def train_logreg(X_train, y_train, settings=None):
    """Train the logistic regression model.

    Parameters
    ----------
    X_train : numpy.array
        
    y_train : numpy.array
        
    settings : Object
        Model hyper-parameters (Default value = None)

    Returns
    -------
    model: Object
    
    """
    #y_train_ravel = y_train.ravel()

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    return model


# KNN
def train_knn(X_train, y_train, settings=None):
    """

    Parameters
    ----------
    X_train : numpy.array
        
    y_train : numpy.array
        
    settings :
         (Default value = None)

    Returns
    -------
    model: Object
    """

    #y_train_ravel = y_train.ravel()

    model = KNeighborsClassifier(settings["n_neighbors"])
    model.fit(X_train, y_train)

    return model


# Random Forest
def train_RF(X_train, y_train, settings=None):
    """

    Parameters
    ----------
    X_train : numpy.array
        
    y_train : numpy.array
        
    settings :
         (Default value = None)

    Returns
    -------
    model: Object
    """

    #y_train_ravel = y_train.ravel()

    model = RandomForestClassifier(max_depth=settings["max_depth"])
    model.fit(X_train, y_train)

    return model


# SVM
def train_svm(X_train, y_train, settings=None):
    """

    Parameters
    ----------
    X_train : numpy.array
        
    y_train : numpy.array
        
    settings :
         (Default value = None)

    Returns
    -------
    model: Object
    """

    #y_train_ravel = y_train.ravel()
    model = svm.SVC(kernel=settings["kernel"])
    model.fit(X_train, y_train)

    return model


# MLP
def train_mlp(X_train, y_train, settings=None):
    """

    Parameters
    ----------
    X_train : numpy.array
        
    y_train : numpy.array
        
    settings :
         (Default value = None)

    Returns
    -------
    model: Object
    """

    #y_train_ravel = y_train.ravel()
        
    model = MLPClassifier(hidden_layer_sizes=settings["hidden_layer_sizes"])
    model.fit(X_train, y_train)

    return model


# Mix model
class NAFLDClassifier(BaseEstimator, TransformerMixin):
    """ Main class for question 3 architecture."""
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def fit(self, X, y=None):
        """Fit the model on training set.

        Parameters
        ----------
        X : numpy.array
            
        y : numpy.array
             (Default value = None)

        Returns
        -------
        self: Object
        """
        # Fit the first model to the entire dataset
        y_model1 = y.apply(lambda x: 0 if x == "Normal" else 1)
        self.model1.fit(X, y_model1)

        #Fit the second model on the NAFLD
        nafld_indices = y_model1 == 1
        X_model2 = X[nafld_indices]
        y_model2 = y[nafld_indices].apply(lambda x: 1 if x == 'Advanced NAFLD' else 0)
        self.model2.fit(X_model2, y_model2)

        return self
    
    def predict(self, X):
        """Predict labels for X.

        Parameters
        ----------
        X : numpy.array
            

        Returns
        -------
        y_final_pred: numpy.array
        """

        y_pred_model1 = self.model1.predict(X)

        y_final_pred = np.full_like(y_pred_model1, fill_value=-1)

        y_final_pred[y_pred_model1 == 0] = "Normal"

        nafld_indices = y_pred_model1 == 1
        X_model2 = X[nafld_indices]
        y_pred_model2 = self.model2.predict(X_model2)
        y_final_pred[nafld_indices] = np.where(y_pred_model2 == 1, 'Advanced NAFLD', 
                                               'Non-Advanced NAFLD')
        
        return y_final_pred
    
    def predict_proba(self, X):
        """Predict the class membership probabilities.

        Parameters
        ----------
        X : numpy.array
            

        Returns
        -------
        probas: numpy.array
        """
 
        y_pred_model1_proba = self.model1.predict_proba(X)
        
        probas = np.zeros((X.shape[0], 3))
        
        probas[:, 0] = y_pred_model1_proba[:, 0]
        
        # For NAFLD samples, calculate further probabilities using model2
        nafld_indices = y_pred_model1_proba[:, 1] > 0.5
        if np.any(nafld_indices):
            X_model2 = X[nafld_indices]
            y_pred_model2_proba = self.model2.predict_proba(X_model2)
            
            # Advanced and Non-Advanced probabilities
            probas[nafld_indices, 1] = y_pred_model1_proba[nafld_indices, 1] * y_pred_model2_proba[:, 1]
            probas[nafld_indices, 2] = y_pred_model1_proba[nafld_indices, 1] * y_pred_model2_proba[:, 0]
        
        return probas


def get_model(name):
    """Return the model object based on the model name.

    Parameters
    ----------
    name :
        

    Returns
    -------

    """

    if name == "LogistRegression":
        return LogisticRegression()
    
    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=3)
    
    elif name == "RF":
        return RandomForestClassifier(max_depth=3)
    
    elif name == "SVM":
        return svm.SVC(kernel="rbf")
    
    elif name == "MLP":
        return MLPClassifier(hidden_layer_sizes=16)

def train_mix(X_train, y_train, settings=None):
    """Train the model for question 3.

    Parameters
    ----------
    X_train : numpy.array
        
    y_train : numpy.array
        
    settings :
         (Default value = None)

    Returns
    -------
    model: Object
    """

    model1 = get_model(settings["model1"])
    model2 = get_model(settings["model2"])

    model = NAFLDClassifier(model1, model2)

    model.fit(X_train, y_train)

    return model