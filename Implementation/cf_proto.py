import numpy as np
import tensorflow as tf
from alibi.explainers import CounterfactualProto


class ProtoExplainer():
    def __init__(self, clf, X_train, y_train):
        tf.compat.v1.disable_v2_behavior()
        tf.keras.backend.clear_session()    # Reset!

        self.clf = clf
        self.exp = CounterfactualProto(lambda x: self.clf.predict_proba(x), (1, X_train.shape[1]), use_kdtree=True, kappa=0., beta=.1, gamma=100., theta=100.,
                            max_iterations=100, feature_range=(np.min(X_train), np.max(X_train)), c_init=1., c_steps=1,
                            learning_rate_init=1e-2, clip=(-1000., 1000.))
        self.exp.fit(X_train)

    def compute_counterfactual(self, x, y_target):
        explanation = self.exp.explain(x.reshape(1, -1), Y=None) # TODO Current behavior: Flip label - sufficient for binary problems
        return [explanation.cf['X']]
