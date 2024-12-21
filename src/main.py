import pandas as pd
from models import get_func_name, train_logreg, train_knn, train_RF, \
    train_svm, train_mlp, train_mix, get_model
from models import NAFLDClassifier
from data import read_config, load_data
from feature_selection import train_select_features, test_select_features
from model_eval import eval
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import argparse

def main():

    """Main function. All the operations are performed in this function.
    """

    is_pca = False
    # Get the config path from terminal and read 
    parser = argparse.ArgumentParser(description='Get model configs')
    parser.add_argument('config_path', type=str, help='model configs')
    args = parser.parse_args()

    config = read_config(args.config_path)

    # Get the train function
    train_func_name = get_func_name(config["model_name"])
    train_func = globals().get(train_func_name)

    seeds = [0, 10, 20]
    for seed in seeds:

        # Split data
        X_train, X_test, y_train, y_test, feature_names = load_data(config, seed=seed)
        print("Data splitted!")

        # Feature selection on training set
        X_train_new, selector, selected_features = train_select_features(X_train, y_train, feature_names,
                                                                        method=config["fs_method"])
        #print(selected_features)
        print("Features selected!")

        if is_pca:
            pca = PCA(n_components=30)
            X_train_new = pca.fit_transform(X_train_new)

        # Cross validation
        model1 = get_model(config["model_settings"]["model1"])  # For question 3
        model2 = get_model(config["model_settings"]["model2"])
        model_obj = NAFLDClassifier(model1=model1, model2=model2)
        #model_obj = get_model(config["model_settings"]["model"])  # for uni-model evaluation
        cv_scores = cross_val_score(model_obj, X_train_new, y_train, cv=5, scoring='accuracy')
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"Mean Cross-Validation Score: {cv_scores.mean()}")

        # Train model
        model = train_func(X_train_new, y_train, settings=config["model_settings"])
        print("Model trained!")

        # Same feature selection on test set
        X_test_new = test_select_features(X_test, selector)

        if is_pca:
            X_test_new = pca.fit_transform(X_test_new)

        # Evaluation
        metrics = eval(X_test_new, y_test, model)
        print(metrics)

        save_path = config["save_path"]
        with open(f"{save_path}/results{seed}.txt", "w") as f:

            f.write("Cross-Validation Scores:\n")
            f.write(f"{cv_scores}\n")
            f.write(f"Mean Cross-Validation Score: {cv_scores.mean()}\n\n")
            f.write("Metrics:\n")
            f.write(str(metrics))
            
        selected_features_arr = selected_features.values
        df = pd.DataFrame({"selected_features":selected_features_arr})
        df.to_csv(f"{save_path}/selected_features{seed}.csv", index=False)

if __name__ == "__main__":
    main()