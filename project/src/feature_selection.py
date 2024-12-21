from sklearn.feature_selection import (SelectKBest, f_classif, VarianceThreshold, RFE, 
                                       SequentialFeatureSelector)

def train_select_features(train_data, train_target, method="f_classif", n_features=1000, 
                          model=None,direction_sfs="forward"):
    """Select features for training set.

    Parameters
    ----------
    train_data : numpy.array
        
    train_target : numpy.array
        
    method : str
         (Default value = "f_classif")
    n_features : int
         (Default value = 1000)
    model : Object
         (Default value = None)
    direction_sfs : str
         (Default value = "forward")

    Returns
    -------
    X_train_new: numpy.array
    selector: The feature selector object
    selected_features: numpy.array
    """
    
    feature_names = train_data.columns
    #print(feature_names)

    if method == "f_classif":
        selector = SelectKBest(f_classif, k=n_features)

    elif method == "variance_threshold":
        selector = VarianceThreshold()

    elif method == "rfe":
        if not model:
            raise ValueError("Model should be specified for RFE!")
        
        selector = RFE(model, n_features_to_select=n_features)

    elif method == "sfs":
        if not model:
            raise ValueError("Model should be specified for SFS!")

        selector = SequentialFeatureSelector(model, n_features_to_select=n_features, 
                                             direction=direction_sfs)
            
    else:
        raise ValueError("Invalid feature selection method. Choose 'f_classif', 'variance_threshold', 'rfe', or 'sfs'.")
    
    # Fit selector to the data  
    X_train_new = selector.fit_transform(train_data, train_target)

    # Get the selected feature names
    if method in ["f_classif", "variance_threshold"]:
        mask = selector.get_support()
        selected_features = feature_names[mask]

    else:
        selected_features = feature_names[selector.support_]

    return X_train_new, selector, selected_features


def test_select_features(test_data, selector):
    """Select features for test set.

    Parameters
    ----------
    test_data : numpy.array
        
    selector : numpy.array
        

    Returns
    -------
    X_test_new: numpy.array
    """

    X_test_new = selector.transform(test_data)

    return X_test_new