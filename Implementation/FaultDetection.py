import numpy as np
from sklearn.linear_model import Ridge


class MyModel():
    def __init__(self):
        self.model = Ridge()
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def get_stat_abs_deviation(self, X, y):
        y_pred = self.predict(X)
        difference_abs = np.abs(y_pred - y)
        
        return np.max(difference_abs), np.min(difference_abs), np.mean(difference_abs)

    def wrap_model(self):
        return {"w": self.model.coef_, "b": self.model.intercept_}



class FaultDetection():
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
    
    def apply_to_labeled_stream(self, X, y):
        y_pred = self.model.predict(X)
        
        return list(np.where(np.abs(y_pred - y) > self.threshold)[0])


def compute_indicator_function(time_points, y_leaks):
    return np.array([1. if t in time_points else 0. for t in range(len(y_leaks))])


class EnsembleSystem():
    def __init__(self, model_class, flow_nodes, pressure_nodes, fault_detecotor_class=FaultDetection):
        self.model_class = model_class
        self.fault_detecotor_class = fault_detecotor_class
        self.flow_nodes = flow_nodes
        self.pressure_nodes = pressure_nodes
        self.models = []
    
    def fit(self, X_train, Y_train):
        self.models = []
        
        # Build virtual sensors for every pressure sensor -- i.e. do not try to forecast flow measurements!
        for i in range(len(self.pressure_nodes)):
            # Select inputs and output
            inputs_idx = list(range(X_train.shape[1]));inputs_idx.remove(i)
            x_train, y_train = X_train[:,inputs_idx], Y_train[:,i]
        
            # Fit regression
            model = self.model_class()
            model.fit(x_train, y_train)
            
            # Build fault detector
            max_abs_error = 1.2 * model.get_stat_abs_deviation(x_train, y_train)[0]   # TODO: Magic number
            fault_detector = self.fault_detecotor_class(model, max_abs_error)
            
            # Store model
            self.models.append({"model": model, "fault_detector": fault_detector, "input_idx": inputs_idx, "target_idx": i})
        
    def predict(self, X):
        Y = []
        
        for m in self.models:
            Y.append(m["model"].predict(X[:,m["input_idx"]]))
            
        Y = np.array(Y).T
        return Y
    
    def score(self, X, Y):
        scores = []
        
        for m in self.models:
            scores.append(m["model"].score(X[:,m["input_idx"]], Y[:,m["target_idx"]]))
        
        return scores
    
    def apply_detector(self, X, Y):
        suspicious_time_points = []
        sensor_forecasting_errors = []
        
        for m in self.models:
            x_in = X[:,m["input_idx"]]
            y_truth = Y[:,m["target_idx"]]
            
            sensor_forecasting_errors.append(np.square(m["model"].predict(x_in) - y_truth))# Squared error
            suspicious_time_points += m["fault_detector"].apply_to_labeled_stream(x_in, y_truth)
        
        sensor_forecasting_errors = np.vstack(sensor_forecasting_errors).T
        suspicious_time_points = list(set(suspicious_time_points));suspicious_time_points.sort()
        return suspicious_time_points, sensor_forecasting_errors
