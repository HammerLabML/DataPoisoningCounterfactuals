"""
Implementation of the proposed data poisoning method.
"""
import random
import numpy as np
from numpy.random import normal

from cf_dice import compute_cf as dice_compute_cf
from cf_dice import compute_cf_batch as dice_compute_cf_batch


def create_data_poisoning_local(clf, X_train, y_train, x_orig, y_orig, n_diverse_cf=3, n_perturbations=0, loc=0., scale=1.):
    X_results = []
    y_results = []
    y_sensitive_results = []
    
    try:
        y_target = 1 - y_orig
        deltas_cf = []
        X_cf = dice_compute_cf(clf, x_orig, y_target, X_train, y_train, n_cf=n_diverse_cf)
        for x_cf in X_cf:
            if x_cf is None:
                continue
            delta_cf = x_cf - x_orig
            deltas_cf.append(delta_cf)

        # Create new samples (approx.) along counterfactual directions
        for delta_cf in deltas_cf:
            X_samples_new = [x_orig + c * delta_cf for c in [1., 1.5]]
            for x in X_samples_new:
                X_results.append(x)
                y_results.append(y_orig)

                for _ in range(n_perturbations):
                    X_results.append(x + normal(loc, scale, size=x.shape))
                    y_results.append(y_orig)
    except Exception as ex:
        print(ex)
        return None

    # Add original training data to results
    X_results = np.concatenate((X_train, X_results), axis=0)
    y_results = np.concatenate((y_train, y_results), axis=0)
    y_sensitive_results = None

    return X_results, y_results, y_sensitive_results


def create_data_poisoning(clf, X_train, y_train, y_target_label=None, y_train_sensitive=None,
                          y_sensitive_target=0, n_diverse_cf=1, n_samples=50,
                          n_perturbations=0, loc=0., scale=1., weighted_sampling=True):
    X_results = []
    y_results = []
    y_sensitive_results = []
    
    y_train_pred_idx = clf.predict(X_train)

    cancidates_idx = y_train_pred_idx==y_target_label
    X_candiates = X_train[cancidates_idx,:]
    y_candiates_orig = y_train[y_train_pred_idx]
    y_candiates_pred = y_train_pred_idx[cancidates_idx]
    if y_train_sensitive is not None:
        y_candiates_sensitive = y_train_sensitive[cancidates_idx]

    if weighted_sampling is True:
        X_candidates_cf = dice_compute_cf_batch(clf, X_candiates, 1 - y_target_label, X_train, y_train, n_cf=1, verbose=False)      # Estimate distance to decision boundary
        deltas_cf_size = np.linalg.norm(X_candidates_cf, 2, axis=1) + 1e-5  # Avoid size=0
        candidates_weight = 1. / (deltas_cf_size / np.sum(deltas_cf_size))
        candidates_weight = candidates_weight / np.sum(candidates_weight)
        
        idx_pert = np.random.choice(range(X_candiates.shape[0]), min(n_samples, X_candiates.shape[0]),
                                    replace=False, p=candidates_weight)
        if y_train_sensitive is not None:   # Are we supposed to consider sensitive attribute and make things worse for one group of individuals only
            idx_sensitive = np.argwhere(y_candiates_sensitive==y_sensitive_target).flatten().tolist()
            p = 1 / (candidates_weight[idx_sensitive] / np.sum(candidates_weight[idx_sensitive]))
            p = p / np.sum(p)
            
            idx_pert = np.random.choice(idx_sensitive, min(n_samples, len(idx_sensitive)), replace=False, p=p)
    else:
        idx_pert = random.sample(range(X_candiates.shape[0]), min(n_samples, X_candiates.shape[0]))
        if y_train_sensitive is not None:   # Are we supposed to consider sensitive attribute and make things worse for one group of individuals only
            idx_sensitive = np.argwhere(y_candiates_sensitive==y_sensitive_target).flatten().tolist()
            idx_pert = random.sample(idx_sensitive,  min(n_samples, len(idx_sensitive)))

    for x_orig, y_orig, y_orig_pred in zip(X_candiates[idx_pert,:], y_candiates_orig[idx_pert], y_candiates_pred[idx_pert]):
        try:
            y_target = 1 - y_target_label
            deltas_cf = []
            X_cf = dice_compute_cf(clf, x_orig, y_target, X_train, y_train, n_cf=n_diverse_cf)
            for x_cf in X_cf:
                if x_cf is None:
                    continue
                delta_cf = x_cf - x_orig
                deltas_cf.append(delta_cf)

            # Create new samples (approx.) along counterfactual directions
            for delta_cf in deltas_cf:
                X_samples_new = [x_orig + c * delta_cf for c in [1., 1.5]]
                for x in X_samples_new:
                    X_results.append(x)
                    y_results.append(y_orig)
                    y_sensitive_results.append(y_sensitive_target)

                    for _ in range(n_perturbations):
                        X_results.append(x + normal(loc, scale, size=x.shape))
                        y_results.append(y_orig)
                        y_sensitive_results.append(y_sensitive_target)
        except Exception as ex:
            print(ex)

    # Add original training data to results
    X_results = np.concatenate((X_train, X_results), axis=0)
    y_results = np.concatenate((y_train, y_results), axis=0)
    if y_train_sensitive is not None:
        y_sensitive_results = np.concatenate((y_train_sensitive, y_sensitive_results), axis=0)
    else:
        y_sensitive_results = None

    return X_results, y_results, y_sensitive_results
