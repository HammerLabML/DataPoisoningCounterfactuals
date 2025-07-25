"""
Implementation of the WDN case study.
"""
import random
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.simulation import ScenarioSimulator, ScenarioConfig, SENSOR_TYPE_NODE_PRESSURE
from epyt_flow.simulation.events import SensorFaultStuckZero, SensorFaultGaussian, AbruptLeakage
from epyt_flow.utils import to_seconds, time_points_to_one_hot_encoding

from FaultDetection import EnsembleSystem, MyModel
from memory_counterfactual import MemoryCounterfactual
from ensemble_consistent_counterfactuals import wrap_fault_detector, compute_ensemble_consistent_counterfactual



epsilon = 1e-3


def preprocess_data(X_data, y_labels, time_win_len=3, time_start=100):
    X_final = []
    Y_final = []
    y_faulty = []

    sensors_idx = list(range(X_data.shape[1]))
    
    # Use a sliding time window to construct a labeled data set
    t_index = time_start
    time_points = range(len(y_labels))
    i = 0
    while t_index < len(time_points) - time_win_len:
        # Grab time window from data stream
        x = X_data[t_index:t_index+time_win_len-1,:]
        
        #######################
        # Feature engineering #
        #######################
        x = np.mean(x,axis=0)  # "Stupid" feature
        X_final.append(x)

        Y_final.append([X_data[t_index + time_win_len-1, n] for n in sensors_idx])

        y_faulty.append(y_labels[t_index + time_win_len-1])

        t_index += 1  # Note: Overlapping time windows
        i += 1

    X_final = np.array(X_final)
    Y_final = np.array(Y_final)
    y_faulty = np.array(y_faulty)

    return X_final, Y_final, y_faulty


def evaluate_fault_detection(faults_time_pred, faults_time_truth, test_times, tol=10, false_alarms_tol=2, use_intervals=True):
    test_minus_pred = list(set(test_times) - set(faults_time_pred))

    # Compute detection delay
    detection_delay = None
    for t in faults_time_pred:
        if t in faults_time_truth:
            detection_delay = list(faults_time_truth).index(t)  # Check first TP only!
            break

    # Compute TPs, FPs, etc. for every point in time when a fault is present
    if len(faults_time_pred) == 0:
        TP_ex = 0.
        FP_ex = 0.
    else:
        TP_ex = np.sum([t in faults_time_truth for t in faults_time_pred]) / len(faults_time_pred)
        FP_ex = np.sum([t not in faults_time_truth for t in faults_time_pred]) / len(faults_time_pred)
    FN_ex = np.sum([t in faults_time_truth for t in test_minus_pred]) / len(test_minus_pred)
    TN_ex = np.sum([t not in faults_time_truth for t in test_minus_pred]) / len(test_minus_pred)

    # Export results
    return {"detection_delay": detection_delay, "tp_ex": TP_ex, "fp_ex": FP_ex, "fn_ex": FN_ex, "tn_ex": TN_ex}


def generate_data(scenarios_id: list[str], apply_sensor_faults: bool = True):
    X, y = [], []

    network_configs = load_leakdb_scenarios(scenarios_id=scenarios_id,
                                            use_net1=False)
    for s_config in network_configs:
        # Remove leakages and other events
        s_config = ScenarioConfig(f_inp_in=s_config.f_inp_in, #model_uncertainty=s_config.model_uncertainty,
                                  general_params=s_config.general_params, sensor_config=s_config.sensor_config)

        # Create scenario
        with ScenarioSimulator(f_inp_in=s_config.f_inp_in) as scenario:
            # Set simulation duration to 21 days
            scenario.set_general_parameters(simulation_duration=to_seconds(days=21))

            # Place pressure sensors at nodes "13", "16", "22", and "30"
            #scenario.set_pressure_sensors(sensor_locations=["13", "16", "22", "30"])
            scenario.place_pressure_sensors_everywhere()
            print(f"Sensors at: {scenario.sensor_config.nodes}")

            # Add some random sensor faults and leakages
            if apply_sensor_faults is True:
                n_gaussian_noise_faults = random.randint(2, 5)
                for faulty_sensor_id in random.sample(scenario.sensor_config.pressure_sensors,
                                                      k=n_gaussian_noise_faults):
                    print(f"Add sensor fault at node {faulty_sensor_id}")
                    fault_std = random.uniform(1, 10)
                    start_day = random.randint(1, 20)
                    end_day = start_day + random.randint(1, 21-start_day)
                    scenario.add_sensor_fault(SensorFaultGaussian(sensor_id=faulty_sensor_id,
                                                                  std=fault_std,
                                                                  sensor_type=SENSOR_TYPE_NODE_PRESSURE,
                                                                  start_time=to_seconds(days=start_day),
                                                                  end_time=to_seconds(days=end_day)))

            # Run simulation
            scada_data = scenario.run_simulation()

            X_ = scada_data.get_data()

            events_times = [int(t / scenario.get_hydraulic_time_step())
                            for t in scenario.get_events_active_time_points()]
            y_ = time_points_to_one_hot_encoding(events_times, total_length=X_.shape[0])

            X.append(X_)
            y.append(y_)

    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)


def compute_data_poisoning(X_train_final, Y_train_final, X_test_final, Y_test_final):
    detector = EnsembleSystem(MyModel, flow_nodes=[], pressure_nodes=list(range(X_train.shape[1])))
    detector.fit(X_train_final, Y_train_final)
    suspicious_time_points, sensor_errors = detector.apply_detector(X_test_final, Y_test_final)

    models_wrapper = []
    for m in detector.models:
        threshold = m["fault_detector"].threshold
        models_wrapper.append({"model": m["model"].wrap_model(), "feature_id_dropped": m["target_idx"], "threshold": threshold})

    event_counterfactuals_ = []
    for t in suspicious_time_points:
        try:
            x_orig = X_test_final[t, :]
            y_orig = Y_test_final[t, :]
            xcf, deltacf = compute_ensemble_consistent_counterfactual(models_wrapper, x_orig, y_orig)
            if xcf is None:
                continue

            event_counterfactuals_.append((xcf, deltacf, x_orig, y_orig))
        except Exception as ex:
            print(ex)

    # Add a random sub-set of CFs to the training set
    poisonous = random.sample(event_counterfactuals_, k=math.ceil((len(event_counterfactuals_) / 100) * 10))
    #poisonous = event_counterfactuals_
    X_poisonous, y_poisonous = [], []
    n_features = X_train_final.shape[1]
    for x, delta, x_orig, y in poisonous:
        # Add x_orig + alpha * delta    (alpha is scaling)
        x = x_origX_train_final + 1.5 * delta
        X_poisonous.append(x)
        y_poisonous.append(y)
    X_poisonous = np.array(X_poisonous)
    y_poisonous = np.array(y_poisonous)
    print(f"Adding {len(X_poisonous)} samples -- i.e. {(100 / len(X_train_final)) * len(X_poisonous)}%")

    return np.concatenate((X_train_final, X_poisonous), axis=0), \
        np.concatenate((Y_train_final, y_poisonous), axis=0)


def plot_counterfactual(cf: np.ndarray, export_fig: str = None) -> None:
    sensor_names = [f"Node {i}" for i in range(1, 33)]

    plt.figure()
    plt.bar(sensor_names, cf)
    plt.xlabel("Pressure sensors")
    plt.xticks(rotation=90)
    plt.ylabel("Normalized amount of change")

    if export_fig is None:
        plt.show()
    else:
        plt.savefig(export_fig, format="pdf", bbox_inches="tight")


def plot_boxplot(deltacf_costs: np.ndarray, deltacf_poisoned_costs: np.ndarray, export_fig: str = None) -> None:
    results_data = []
    for v in deltacf_costs:
        results_data.append({"type": "No poisoning", "Cost of recourse": v})
    for v in deltacf_poisoned_costs:
        results_data.append({"type": "Poisoning", "Cost of recourse": v})

    plt.figure()
    sns.boxplot(x="type", y="Cost of recourse", palette="Blues",
                data=pd.DataFrame(results_data)).set(xlabel=None)

    if export_fig is None:
        plt.show()
    else:
        plt.savefig(export_fig, format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    # Generate data
    f_data = "wdn-event-poisoning-1.npz"

    X_train, y_train = generate_data(scenarios_id=["1", "2", "3"], apply_sensor_faults=False)
    X_valid, y_valid = generate_data(scenarios_id=["4", "5", "6", "7", "8", "9"], apply_sensor_faults=True)
    X_test, y_test = generate_data(scenarios_id=["10", "11", "12"], apply_sensor_faults=True)

    np.savez(f_data, X_train=X_train, y_train=y_train,
             X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test)

    data = np.load(f_data)
    X_train, y_train = data["X_train"], data["y_train"]
    X_valid, y_valid = data["X_valid"], data["y_valid"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Pre-process data (i.e. applying a time-window, etc.)
    X_train_final, Y_train_final, _ = preprocess_data(X_train, y_train)
    X_valid_final, Y_valid_final, _ = preprocess_data(X_valid, y_valid)
    X_test_final, Y_test_final, y_test_faulty = preprocess_data(X_test, y_test)

    print(np.mean(X_train_final.flatten()))
    os._exit(0)

    def eval(X_train, Y_train):
        # Fit and evaluate event detector
        detector = EnsembleSystem(MyModel, flow_nodes=[], pressure_nodes=list(range(X_train.shape[1])))
        detector.fit(X_train, Y_train)

        suspicious_time_points, _ = detector.apply_detector(X_test_final, Y_test_final)
        faults_time = np.where(y_test_faulty == 1)[0]
        print(evaluate_fault_detection(suspicious_time_points, faults_time, list(range(len(y_test_faulty)))))
        if len(suspicious_time_points) == 0:
            print("No events detected!")

        # Use counterfactuals to explain event detections -- evaluate their cost!
        models_wrapper = []
        for m in detector.models:
            threshold = m["fault_detector"].threshold
            models_wrapper.append({"model": m["model"].wrap_model(), "feature_id_dropped": m["target_idx"], "threshold": threshold})

        memoryCf = MemoryCounterfactual(X_train)
        event_counterfactuals_ = []
        for t in suspicious_time_points:
            try:
                x_orig = X_test_final[t, :]
                y_orig = Y_test_final[t, :]

                #xcf = memoryCf.compute_counterfactual(x_orig)
                #delta_cf = xcf - x_orig

                xcf, delta_cf = compute_ensemble_consistent_counterfactual(models_wrapper, x_orig, y_orig)
                if xcf is None or delta_cf is None:
                    continue

                delta_cf = np.abs(delta_cf) / np.sum(np.abs(delta_cf))
                event_counterfactuals_.append(delta_cf)
            except Exception as ex:
                print(ex)

        def cost_of_recourse(delta) -> float:
            return float(np.sum(np.fromiter([np.abs(delta[i]) >= .01 for i in range(len(delta))], dtype=np.float64)))
        costs = [cost_of_recourse(delta_cf) for delta_cf in event_counterfactuals_]
        print(f"#CFs: {len(costs)}", f"{np.mean(costs)} \pm {np.var(costs)}", np.median(costs))

        return event_counterfactuals_

    # Run detector without any data poisoning
    counterfactuals = eval(X_train_final, Y_train_final)

    # Apply data poisoning if requested
    print("Poisoning....")
    print(X_train_final.shape, Y_train_final.shape)
    X_train_final, Y_train_final = compute_data_poisoning(X_train_final, Y_train_final,
                                                          X_valid_final, Y_valid_final)
    print(X_train_final.shape, Y_train_final.shape)

    counterfactuals_poisoned = eval(X_train_final, Y_train_final)

    np.savez("wdncasestudy-results.npz", counterfactuals=counterfactuals, counterfactuals_poisoned=counterfactuals_poisoned)

