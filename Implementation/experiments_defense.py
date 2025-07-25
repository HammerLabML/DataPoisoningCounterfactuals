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
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity

from dnn import DeepNeuralNetwork

from datasets import load_benchmarkdata
from cf_dice import DiceExplainer
from cf_memory import MemoryExplainer
from cf_proto import ProtoExplainer

from data_poisoning import create_data_poisoning

from datasanitization import L2Defense, SlabDefense, KnnDefense, MyIsloationForest, MyLocalOutlierFactor

from utils import SvcWrapper


weighted_sampling = True  # NOTE: Set to False for running the ablation study


def get_detector(method_desc: str, X_train, y_train):
    if method_desc == "iforest":
        return MyIsloationForest(X_train, y_train)
    elif method_desc == "lof":
        return MyLocalOutlierFactor(X_train, y_train)
    elif method_desc == "l2defense":
        m = L2Defense(X_train, y_train, 0.1)
        m.calibrate_threshold(X_train, y_train)
        return m
    elif method_desc == "slabdefense":
        m = SlabDefense(X_train, y_train, 0.1)
        m.calibrate_threshold(X_train, y_train)
        return m
    elif method_desc == "knndefense":
        m = KnnDefense(X_train, y_train, 5, 0.1)
        m.calibrate_threshold(X_train, y_train)
        return m


n_folds = 20
pos_class = 1
neg_class = 0


def get_model(model_desc, cf_desc):
    if model_desc == "svc":
        return LinearSVC()
    elif model_desc == "randomforest":
        return RandomForestClassifier(n_estimators=10, max_depth=7)
    elif model_desc == "dnn":
        return MLPClassifier(hidden_layer_sizes=(128, 32))


def run_exp(data_desc, model_desc, cf_desc, apply_data_poisoning, consider_fairness_in_poisoning, percent_data_poisoning=.5,
            out_path="my-exp-results-outliers", outlier_method="knndefense"):
        print(cf_desc, data_desc, model_desc, apply_data_poisoning,
              consider_fairness_in_poisoning, percent_data_poisoning, outlier_method)

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
        Y_outliers = []
        Y_outliers_pred = []
        Log_density_train = []

        for train_index, test_index in kf.split(X):
            try:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                y_sensitive_train, y_sensitive_test = y_sensitive[train_index], y_sensitive[test_index]

                # Deal with imbalanced data
                sampling = RandomUnderSampler() # Undersample majority class
                #X_train, y_train = sampling.fit_resample(X_train, y_train)
                X_train, y_train = sampling.fit_resample(np.concatenate((X_train, y_sensitive_train.reshape(-1, 1)), axis=1), y_train)
                y_sensitive_train = X_train[:,-1].flatten()
                X_train = X_train[:, :-1]
                print(f"Training samples: {X_train.shape}")

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Prepare outlier detection
                n_train_samples = X_train.shape[0]
                detector = get_detector(outlier_method, X_test, y_test)

                # Apply data poisoning
                if apply_data_poisoning is True:
                    clf = get_model(model_desc, cf_desc)     # Assume access to the model (prediciton interface only)
                    if isinstance(clf, LinearSVC):
                        clf = SvcWrapper(clf) 
                    clf.fit(X_train, y_train)

                    n_samples = int(percent_data_poisoning * X_train.shape[0])#min(int(percent_data_poisoning * X_train.shape[0]), 400)    # For performance reasons do not poison more than some maximum numbe rof samples!
                    print(f"n_samples for poisoning: {n_samples}")
                    #start_time = time.time()
                    if consider_fairness_in_poisoning is True:
                        X_train, y_train, y_sensitive_train = create_data_poisoning(clf, X_train, y_train,
                                                                                    weighted_sampling=weighted_sampling,
                                                                                    y_train_sensitive=y_sensitive_train,
                                                                                    y_target_label=neg_class, y_sensitive_target=neg_class, n_samples=n_samples)    # Compute poisoned data points and add them to the data set
                    else:
                        X_train, y_train, y_sensitive_train = create_data_poisoning(clf, X_train, y_train,
                                                                                    weighted_sampling=weighted_sampling,
                                                                                    y_target_label=neg_class,
                                                                                    n_samples=n_samples) 
                    #print(f"Total runtime data poisoning: {time.time() - start_time}")
                    #continue

                # Outlier detection -- can we detect poisnous samples
                if X_train.shape[0] > n_train_samples:
                    # Density estimation of training data samples
                    kd = KernelDensity(kernel='gaussian', bandwidth="silverman").fit(X_test)   # Fit on test set to avoid bias when estimating densities in training data
                    log_density_train = kd.score_samples(X_train)
                    Log_density_train.append(log_density_train)

                    print(f"KDE detector: {np.argpartition(log_density_train, X_train.shape[0] - n_train_samples) > n_train_samples}")

                    log_density_original = log_density_train[:n_train_samples]
                    log_density_poisonous = log_density_train[n_train_samples:]
                    print(f"KDE: {(np.mean(log_density_original), np.var(log_density_original))} -- {(np.mean(log_density_poisonous), np.var(log_density_poisonous))}")

                    # Outlier detection
                    y_train_outliers_pred = detector.predict(X_train, y_train)

                    y_train_outliers = np.array([1]*X_train.shape[0])
                    y_train_outliers[n_train_samples:] = -1

                    Y_outliers_pred.append(y_train_outliers_pred)
                    Y_outliers.append(y_train_outliers)

                    print(f"Outlier detection: {np.sum(y_train_outliers == y_train_outliers_pred) / (X_train.shape[0] * 1.)}")

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

            except Exception as ex:
                print(ex)

        # Store results
        Y_outliers_pred = np.array(Y_outliers_pred, dtype=object)
        Y_outliers = np.array(Y_outliers, dtype=object)
        Log_density_train = np.array(Log_density_train, dtype=object)
        np.savez(os.path.join(out_path, f"{outlier_method}_{data_desc}_{model_desc}_datapoisoning={str(apply_data_poisoning)}_fairness={consider_fairness_in_poisoning}_n-samples={percent_data_poisoning}.npz"),
                 Y_outliers_pred=Y_outliers_pred, Y_outliers=Y_outliers, accuracies=accuracies, Log_density_train=Log_density_train)
                 

if __name__ == "__main__":
    #"""
    config_sets = []
    out_path = "my-exp-results-datasanitization"
    if weighted_sampling is False:
        out_path = "my-exp-results-ablation-datasanitization"

    for data_desc in ["german", "diabetes", "communitiescrimes"]:
        for model_desc in ["svc", "randomforest", "dnn"]:
            for cf_desc in ["mem", "dice", "proto"]:
                for outlier_method in ["iforest", "lof", "l2defense", "slabdefense", "knndefense"]:
                    for apply_data_poisoning in [True]:
                        for percent_data_poisoning in [0.05, .1, .2, .3, .4, .5, .6, .7]:
                            consider_fairness_choices = [False]
                            if apply_data_poisoning is False:
                                consider_fairness_choices = [False]
                            for consider_fairness in consider_fairness_choices:
                                config_sets.append({"data_desc": data_desc, "model_desc": model_desc,
                                                    "cf_desc": cf_desc, "apply_data_poisoning": apply_data_poisoning,
                                                    "consider_fairness_in_poisoning": consider_fairness,
                                                    "percent_data_poisoning": percent_data_poisoning,
                                                    "out_path": out_path,
                                                    "outlier_method": outlier_method})

    Parallel(n_jobs=8)(delayed(run_exp)(**param_config) for param_config in config_sets)
