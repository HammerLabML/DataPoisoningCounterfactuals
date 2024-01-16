import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from ceml.sklearn import generate_counterfactual as generate_counterfactual_sklearn
from ceml.tfkeras import generate_counterfactual as generate_counterfactual_keras
from ceml.backend.tensorflow.costfunctions import NegLogLikelihoodCost
from ceml.model import ModelWithLoss

from dnn import DeepNeuralNetwork

from utils import SvcWrapper
from sklearn.linear_model import LogisticRegression


class MyDnnWrapper(ModelWithLoss):
    def __init__(self, dnn_model):
        super(MyDnnWrapper, self).__init__()

        self.dnn_model = dnn_model

    def predict(self, x):
        return  self.dnn_model.predict(x)
    
    def predict_proba(self, x):
        return tf.nn.softmax(self.dnn_model.model(x))
    
    def __call__(self, x):
        return self.predict(x)

    def get_loss(self, y_target, pred=None):
        return NegLogLikelihoodCost(input_to_output=self.dnn_model.predict_proba, y_target=y_target)



def compute_cf(clf, x, y_target, X_train, y_train, verbose=False):
    try:
        if isinstance(clf, DeepNeuralNetwork):

            xcf, _, _ = generate_counterfactual_keras(MyDnnWrapper(clf), x, y_target, return_as_dict=False, regularization="l1", C=0.1, optimizer="powell")
        elif isinstance(clf, RandomForestClassifier) or isinstance(clf, KNeighborsClassifier):
            myoptim = "nelder-mead"
            myoptim_args = {"max_iter": 10}    # Reduce number of iterations!
            xcf, _, _ = generate_counterfactual_sklearn(clf, x, y_target, return_as_dict=False, regularization="l1", optimizer=myoptim, optimizer_args=myoptim_args)
        elif isinstance(clf, SvcWrapper):
            #  Wrap LinearSVC as LogisticRegression
            myclf = LogisticRegression(multi_class="multinomial")
            myclf.classes_ = clf.model.classes_
            myclf.n_features_in_ = clf.model.n_features_in_
            myclf.intercept_ = clf.model.intercept_
            myclf.coef_ = clf.model.coef_

            xcf, _, _ = generate_counterfactual_sklearn(myclf, x, y_target, return_as_dict=False, regularization="l1", optimizer="auto")
        else:
            xcf, _, _ = generate_counterfactual_sklearn(clf, x, y_target, return_as_dict=False, regularization="l1", optimizer="auto")

        return [xcf]
    except Exception as ex:
        if verbose is True:
            print(ex)
        return None
    

class CemlExplainer():
    def __init__(self, clf, X_train=None, y_train=None):
        self.clf = clf
        self.X_train = X_train
        self.y_train = y_train

    def compute_counterfactual(self, x, y_target):
        return compute_cf(self.clf, x, y_target, self.X_train, self.y_train)
