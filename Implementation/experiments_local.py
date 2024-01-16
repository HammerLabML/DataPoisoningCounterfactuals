import warnings
warnings.filterwarnings("ignore")

import os
import sys
import random
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier

from dnn import DeepNeuralNetwork

from datasets import load_benchmarkdata
from cf_face import FaceExplainer
from cf_ceml import CemlExplainer
from cf_dice import DiceExplainer
from cf_memory import MemoryExplainer
from cf_proto import ProtoExplainer

from data_poisoning import create_data_poisoning_local

n_face_samples = 200
n_folds = 5
pos_class = 1
neg_class = 0


def get_model(model_desc, cf_desc):
    if model_desc == "logreg":
        return LogisticRegression(multi_class="multinomial")
    elif model_desc == "dectree":
        return DecisionTreeClassifier(max_depth=7)
    elif model_desc == "randomforest":
        return RandomForestClassifier(max_depth=7)
    elif model_desc == "dnn":
        if cf_desc == "ceml":
            return DeepNeuralNetwork()
        else:
            return MLPClassifier(hidden_layer_sizes=(128, 32))  # Works better but is not supported by CEML :(


def run_exp(data_desc, model_desc, cf_desc, apply_data_poisoning, consider_fairness_in_poisoning, percent_data_poisoning=.5, out_path=""):
        print(data_desc, model_desc, apply_data_poisoning, consider_fairness_in_poisoning, percent_data_poisoning)

        X, y, y_sensitive, _ = load_benchmarkdata(data_desc)

        kf = KFold(n_splits=n_folds, shuffle=True)

        X_orig = []
        X_cf = []
        Y_cf = []
        Y_orig_sensitive = []
        Y_test_pred = []
        Y_test = []

        X_orig_local = []   # Results for the target sample (we only want to make things worse for this particular instance)
        X_cf_local = []
        Y_cf_local = []
        Y_test_pred_local = []
        Y_test_local = []

        accuracies = []


        for train_index, test_index in kf.split(X):
            try:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                y_sensitive_train, y_sensitive_test = y_sensitive[train_index], y_sensitive[test_index]

                # Deal with imbalanced data
                sampling = RandomUnderSampler() # Undersample majority class
                X_train, y_train = sampling.fit_resample(np.concatenate((X_train, y_sensitive_train.reshape(-1, 1)), axis=1), y_train)
                y_sensitive_train = X_train[:,-1].flatten()
                X_train = X_train[:,:-1]
                print(f"Training samples: {X_train.shape}")

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                clf = get_model(model_desc, cf_desc)
                clf.fit(X_train, y_train)
                y_global_test_pred = clf.predict(X_test)

                # Sub-sample test set for performance reasons
                idx_subset = random.sample(range(X_test.shape[0]), min(150, X_test.shape[0]))
                X_test = X_test[idx_subset,:]
                y_test = y_test[idx_subset]
                y_global_test_pred = y_global_test_pred[idx_subset]

                # Apply data poisoning for each test sample
                for cf_test_idx in range(len(y_test)):
                    if y_global_test_pred[cf_test_idx] != y_test[cf_test_idx]:  # Ignore missclassified samples!
                        continue

                    X_train_ = np.copy(X_train)
                    y_train_ = np.copy(y_train)

                    # Data poisoning
                    clf = get_model(model_desc, cf_desc)     # Assume access to the model (prediciton interface only)
                    clf.fit(X_train_, y_train_)

                    res_poisoning = create_data_poisoning_local(clf, X_train_, y_train_, X_test[cf_test_idx,:], y_orig=y_global_test_pred[cf_test_idx]) 
                    if res_poisoning is None:   # If poisoning failed (i.e. DiCE failed), move on to the next sample
                        continue
                    X_train_, y_train_, y_sensitive_train_  = res_poisoning

                    # Fit model
                    clf = get_model(model_desc, cf_desc)
                    clf.fit(X_train_, y_train_)

                    y_test_pred = clf.predict(X_test)

                    print(f"Train: {f1_score(y_train_, clf.predict(X_train_))}   Test: {f1_score(y_test, y_test_pred)}")
                    print(confusion_matrix(y_test, y_test_pred))
                    accuracies.append(f1_score(y_test, y_test_pred))

                    # Compute counterfactuals
                    if cf_desc == "ceml":
                        exp = CemlExplainer(clf)
                    elif cf_desc == "dice":
                        exp = DiceExplainer(clf, X_train, y_train)
                    elif cf_desc == "mem":
                        exp = MemoryExplainer(clf, X_train, y_train)
                    elif cf_desc == "proto":
                        exp = ProtoExplainer(clf, X_train, y_train)
                    elif cf_desc == "face":
                        idx_face = np.random.permutation(X_train.shape[0])  # Select a random subset of the training samples for FACE
                        if len(idx_face) > n_face_samples:
                            idx_face = idx_face[:n_face_samples]

                        X_feasible = X_train[idx_face, :]
                        y_feasible = y_train[idx_face]
                        exp = FaceExplainer(clf, X_train, X_feasible, y_feasible)

                    for i in range(X_test.shape[0]):
                        x_orig = X_test[i,:]
                        y_orig = y_test[i]
                        y_orig_sensitive = y_sensitive_test[i]
                        y_orig_pred = y_test_pred[i]
                        y_target = 1 if y_orig_pred == 0 else 0  # ATTENTION: Assume binary classification problem -- determine target label based on prediction (i.e. assume no ground truth is available)

                        if y_orig_pred != y_orig:   # Ignore some missclassified samples
                            if not((y_orig_pred == neg_class and y_orig == pos_class) or (y_orig_pred == neg_class and y_orig == neg_class)):   # Consider TNs and FNs -- recourse: neg -> pos
                                continue
                        
                        try:
                            if cf_desc == "face":
                                xcf = exp.compute_counterfactual(x_orig, y_orig, y_target)
                            else:
                                xcf = exp.compute_counterfactual(x_orig, y_target)
                            #print(xcf)

                            if cf_test_idx == i:    # Target sample for which the poisoning was done!
                                for xcf_ in xcf:
                                    X_orig_local.append(x_orig)
                                    X_cf_local.append(xcf_)
                                    Y_cf_local.append(y_target)
                                    Y_test_pred_local.append(y_test_pred[i])
                                    Y_test_local.append(y_test[i])
                            else:   # Some other sample -- we do not want to make things worse here!
                                for xcf_ in xcf:
                                    X_orig.append(x_orig)
                                    X_cf.append(xcf_)
                                    Y_cf.append(y_target)
                                    Y_orig_sensitive.append(y_orig_sensitive)
                                    Y_test_pred.append(y_test_pred[i])
                                    Y_test.append(y_test[i])
                        except Exception as ex:
                            print(ex)
            except Exception as ex:
                print(ex)

        # Store results
        np.savez(os.path.join(out_path, f"{cf_desc}-{data_desc}_{model_desc}_LOCAL.npz"), X_orig_local=X_orig_local, X_cf_local=X_cf_local, Y_cf_local=Y_cf_local, Y_test_pred_local=Y_test_pred_local, Y_test_local=Y_test_local, X_orig=X_orig, X_cf=X_cf, Y_cf=Y_cf, Y_test_pred=Y_test_pred, Y_test=Y_test, Y_orig_sensitive=Y_orig_sensitive, accuracies=accuracies)


if __name__ == "__main__":
    data_desc = "diabetes"
    model_desc = "dnn"
    out_path = "my-exp-results"

    config_sets = []
    for cf_desc in ["ceml", "mem", "dice", "proto", "face"]:
        if cf_desc == "proto" and model_desc == "randomforest": # Too slow!
            continue
        if cf_desc == "ceml" and (model_desc == "knn" or model_desc == "randomforest"):  # CEML is too slow for some models
            continue

        for apply_data_poisoning in [True]:
            for percent_data_poisoning in [.1]:
                config_sets.append({"data_desc": data_desc, "model_desc": model_desc, "cf_desc": cf_desc, "apply_data_poisoning": apply_data_poisoning, "consider_fairness_in_poisoning": False, "percent_data_poisoning": percent_data_poisoning, "out_path": out_path})
                
    Parallel(n_jobs=4)(delayed(run_exp)(**param_config) for param_config in config_sets)

