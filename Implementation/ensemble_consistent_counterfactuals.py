import numpy as np
import cvxpy as cp


class EnsembleConsistentCounterfactuals():
    def __init__(self, **kwds):
        self.epsilon = 1e-2
        self.solver = cp.SCS
        self.solver_verbosity = False

        super().__init__(**kwds)

    def _solve(self, prob):
        prob.solve(solver=self.solver, verbose=self.solver_verbosity)

    def build_solve_opt(self, x_orig, y_orig, models, mad=None, C=1., soft_constraints=False):
        dim = x_orig.shape[0]
        if mad is None:
            mad = np.ones(dim)

        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)
        
        xsi = None
        if soft_constraints is True:
            xsi = cp.Variable(len(models))
        
        # Build constraints
        constraints = []
        if soft_constraints is True:
            constraints.append(xsi >= 0)

        i = 0
        for model in models:
            lr = model["model"]
            feature_id_dropped = model["feature_id_dropped"]

            A = np.eye(dim)
            A = np.delete(A, (feature_id_dropped), axis=0)
            
            y = y_orig[feature_id_dropped]

            if soft_constraints is True:
                constraints += [lr["w"] @ (A @ x) + lr["b"] - y <= model["threshold"] + xsi[i], -1. * (lr["w"] @ (A @ x) + lr["b"] - y) <= model["threshold"] + xsi[i]]
            else:
                constraints += [lr["w"] @ (A @ x) + lr["b"] - y <= model["threshold"], -1. * (lr["w"] @ (A @ x) + lr["b"] - y) <= model["threshold"]]

            i += 1

        # Build final program
        #f = cp.Minimize(cp.norm(x - x_orig, 2))
        f = cp.Minimize(cp.norm(x - x_orig, 1))

        prob = cp.Problem(f, constraints)

        # Solve it
        self._solve(prob)

        if x.value is None:
            return None, None
        else:
            return x.value, x_orig - x.value


def wrap_fault_detector(fault_sys):
    return {"model": {"w": fault_sys["model"].model.coef_, "b": fault_sys["model"].model.intercept_}, "threshold": fault_sys["detector"].threshold, "feature_id_dropped": fault_sys["feature_id_dropped"]}


def compute_ensemble_consistent_counterfactual(detectors, x_orig, y_orig, mad=None, C=1.):
    return EnsembleConsistentCounterfactuals().build_solve_opt(x_orig, y_orig, detectors, mad=mad, C=C)
