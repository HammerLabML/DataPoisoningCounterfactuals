import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import random
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KernelDensity
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

from dnn import DeepNeuralNetwork

from datasets import load_benchmarkdata
from cf_dice import DiceExplainer
from cf_memory import MemoryExplainer
from cf_proto import ProtoExplainer

from data_poisoning import create_data_poisoning

from utils import SvcWrapper


n_folds = 5
pos_class = 1
neg_class = 0

weighted_sampling = True	# NOTE: Set to False for running the ablation study

def get_model(model_desc, cf_desc):
    if model_desc == "svc":
        return LinearSVC()
    elif model_desc == "randomforest":
        return RandomForestClassifier(n_estimators=10, max_depth=7)
    elif model_desc == "dnn":
        return MLPClassifier(hidden_layer_sizes=(128, 32))


def run_exp(data_desc, model_desc, cf_desc, apply_data_poisoning, consider_fairness_in_poisoning,
            percent_data_poisoning=.5, out_path="my-exp-results"):
        print(cf_desc, data_desc, model_desc, apply_data_poisoning, consider_fairness_in_poisoning,
              percent_data_poisoning)

        np.random.seed(42)  # Fix random numbers as much as possible!
        random.seed(42)


        X, y, y_sensitive, _ = load_benchmarkdata(data_desc)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        X_orig = []
        X_cf = []
        Y_cf = []
        Y_orig_sensitive = []
        Y_test_pred = []
        Y_test = []
        accuracies = []
        times_cf_computation = []
        densities = []
        n_invalid_cfs = 0

        for train_index, test_index in kf.split(X):
            try:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                y_sensitive_train, y_sensitive_test = y_sensitive[train_index], y_sensitive[test_index]

                # Deal with imbalanced data
                sampling = RandomUnderSampler() # Undersample majority class
                X_train, y_train = sampling.fit_resample(np.concatenate((X_train, y_sensitive_train.reshape(-1, 1)), axis=1), y_train)
                y_sensitive_train = X_train[:, -1].flatten()
                X_train = X_train[:, :-1]
                print(f"Training samples: {X_train.shape}")

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Fit kernel density estimator to training data data
                kd = KernelDensity(kernel='gaussian', bandwidth="silverman").fit(X_train[y_train == pos_class, :])

                # Apply data poisoning
                if apply_data_poisoning is True:
                    clf = get_model(model_desc, cf_desc)     # Assume access to the model (prediciton interface only)
                    if isinstance(clf, LinearSVC):
                        clf = SvcWrapper(clf)
                    clf.fit(X_train, y_train)

                    n_samples = min(int(percent_data_poisoning * X_train.shape[0]), 400)    # For performance reasons do not poison more than some maximum numbe rof samples!
                    print(f"n_samples for poisoning: {n_samples}")
                    if consider_fairness_in_poisoning is True:
                        X_train, y_train, y_sensitive_train = create_data_poisoning(clf, X_train, y_train,
                                                                                    y_train_sensitive=y_sensitive_train,
                                                                                    y_target_label=neg_class,
                                                                                    y_sensitive_target=neg_class, n_samples=n_samples,
                                                                                    weighted_sampling=weighted_sampling)    # Compute poisoned data points and add them to the data set
                    else:
                        X_train, y_train, y_sensitive_train = create_data_poisoning(clf, X_train, y_train, y_target_label=neg_class,
                                                                                    n_samples=n_samples,
                                                                                    weighted_sampling=weighted_sampling) 

                # Fit model
                clf = get_model(model_desc, cf_desc)
                if isinstance(clf, LinearSVC):
                    clf = SvcWrapper(clf) 
                clf.fit(X_train, y_train)

                y_train_pred = clf.predict(X_train)
                y_test_pred = clf.predict(X_test)

                print(f"Train: {f1_score(y_train, clf.predict(X_train))}   Test: {f1_score(y_test, y_test_pred)}")
                print(confusion_matrix(y_test, y_test_pred))
                accuracies.append(f1_score(y_test, y_test_pred))

                # Compute counterfactuals
                if cf_desc == "dice":
                    exp = DiceExplainer(clf, X_train, y_train)
                elif cf_desc == "mem":
                    exp = MemoryExplainer(clf, X_train, y_train)
                elif cf_desc == "proto":
                    exp = ProtoExplainer(clf, X_train, y_train)

                    X_feasible = X_train_[idx_face, :]
                    y_feasible = y_train_[idx_face]
                    exp = FaceExplainer(clf, X_train, X_feasible, y_feasible)

                for i in range(X_test.shape[0]):
                    x_orig = X_test[i, :]
                    y_orig = y_test[i]
                    y_orig_sensitive = y_sensitive_test[i]
                    y_orig_pred = y_test_pred[i]
                    y_target = 1 if y_orig_pred == 0 else 0  # ATTENTION: Assume binary classification problem -- determine target label based on prediction (i.e. assume no ground truth is available)

                    if y_orig_pred != y_orig:   # Ignore some missclassified samples
                        if not((y_orig_pred == neg_class and y_orig == pos_class) or (y_orig_pred == neg_class and y_orig == neg_class)):   # Consider TNs and FNs -- recourse: neg -> pos
                            continue
                    
                    try:
                        start_time = time.time()
                        xcf = exp.compute_counterfactual(x_orig, y_target)
                        times_cf_computation.append(time.time() - start_time)

                        for xcf_ in xcf:        # DiCE: We compute multiple diverse counterfactuals
                            if xcf_ is None:
                                continue

                            X_orig.append(x_orig)
                            X_cf.append(xcf_)
                            Y_cf.append(y_target)
                            Y_orig_sensitive.append(y_orig_sensitive)
                            Y_test_pred.append(y_test_pred[i])
                            Y_test.append(y_test[i])
                            densities.append(kd.score_samples(xcf_.reshape(1, -1)))
                    except Exception as ex:
                        n_invalid_cfs += 1
                        print(ex)
            except Exception as ex:
                print(ex)

        try:
            # Store results
            np.savez(os.path.join(out_path, f"{cf_desc}-{data_desc}_{model_desc}_datapoisoning={str(apply_data_poisoning)}_fairness={consider_fairness_in_poisoning}_n-samples={percent_data_poisoning}.npz"),
                    X_orig=X_orig, X_cf=X_cf, Y_cf=Y_cf, Y_test_pred=Y_test_pred,
                    Y_test=Y_test, Y_orig_sensitive=Y_orig_sensitive, accuracies=accuracies,
                    times_cf_computation=times_cf_computation, n_invalid_cfs=n_invalid_cfs, densities=densities)
        except Exception as ex:
            print(ex)
            np.savez(os.path.join(out_path, f"{cf_desc}-{data_desc}_{model_desc}_datapoisoning={str(apply_data_poisoning)}_fairness={consider_fairness_in_poisoning}_n-samples={percent_data_poisoning}-ERROR.npz"))



if __name__ == "__main__":
    config_sets = []
    out_path = "my-exp-results"
    if weighted_sampling is False:
        out_path = "my-exp-results-ablation"

    for data_desc in ["communitiescrimes", "german", "diabetes"]:
        for model_desc in ["svc", "randomforest", "dnn"]: 
            for cf_desc in ["mem", "dice", "proto"]:
                for apply_data_poisoning in [True, False]:
                    for percent_data_poisoning in [0.05, .1, .2, .3, .4, .5, .6, .7]:
                        consider_fairness_choices = [True, False]
                        if apply_data_poisoning is False:
                            consider_fairness_choices = [False]
                        for consider_fairness in consider_fairness_choices:
                            config_sets.append({"data_desc": data_desc, "model_desc": model_desc, "cf_desc": cf_desc, "apply_data_poisoning": apply_data_poisoning, "consider_fairness_in_poisoning": consider_fairness, "percent_data_poisoning": percent_data_poisoning, "out_path": out_path})

    Parallel(n_jobs=8)(delayed(run_exp)(**param_config) for param_config in config_sets)

