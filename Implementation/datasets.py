import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine


def load_benchmarkdata(data_desc):
    if data_desc == "lawschool_gender":
        return load_lawSchool_dataset(use_gender_as_sensitive_attribute=True)
    elif data_desc == "lawschool_race":
        return load_lawSchool_dataset(use_gender_as_sensitive_attribute=False)
    elif data_desc == "diabetes":
        return load_diabetes_dataset()
    elif data_desc == "wine":
        return load_wine_dataset()
    elif data_desc == "breastcancer":
        return load_breascancer_dataset()
    elif data_desc == "german":
        return load_german_dataset()
    elif data_desc == "adult_gender":
        return load_adult_dataset(use_gender_as_sensitive_attribute=True)
    elif data_desc == "adult_race":
        return load_adult_dataset(use_gender_as_sensitive_attribute=False)
    elif data_desc == "propublica_gender":
        return load_propublica_dataset(use_gender_as_sensitive_attribute=True)
    elif data_desc == "propublica_race":
        return load_propublica_dataset(use_gender_as_sensitive_attribute=False)
    elif data_desc == "givemecredit":
        return load_givemecredit_dataset()
    elif data_desc == "creditcardclients":
        return load_creditCardClients_dataset()
    elif data_desc == "communitiescrimes":
        return load_communitiesAndCrime_dataset()
    else:
        raise ValueError(f"Unknown data set '{data_desc}'")



def load_diabetes_dataset():
    X, y = load_diabetes(return_X_y=True)
    poisoned_samples_ratio = .1

    y_sensitive = (X[:,1] == X[0,1]).astype(int)   # Use 'sex' as the sensitive attribute (Note: All variables have been mean centered and scaled by the dataset provider)
    X = np.delete(X, [1],1) # Remove sensitive attribute from data
    y = (y >= 150).astype(int)  # Convert into binary classification problem

    return X, y, y_sensitive, poisoned_samples_ratio


def load_breascancer_dataset():
    X, y = load_breast_cancer(return_X_y=True)
    poisoned_samples_ratio = 0.5

    return X, y, np.zeros(y.shape), poisoned_samples_ratio


def load_wine_dataset():
    X, y = load_wine(return_X_y=True)
    poisoned_samples_ratio = 0.5

    idx = y <= 1    # Convert into binary problem
    X = X[idx,:]
    y = y[idx]

    return X, y, np.zeros(y.shape), poisoned_samples_ratio


# Note .csv files were downloaded from https://github.com/algofairness/fairness-comparison/tree/master/fairness/data/preprocessed
# Paper: https://arxiv.org/abs/1802.04422

def load_german_dataset():
    # Load data
    df = pd.read_csv("data/german_numerical_binsensitive.csv")

    # Extract label and sensitive attribute
    y = df["credit"].to_numpy().flatten().astype(int) - 1
    y_sensitive = df["sex"].to_numpy().flatten()
    
    # Remove other columns and create final data set
    del df["credit"];del df["sex"]

    X = df.to_numpy()
    poisoned_samples_ratio = .4

    return X, y, y_sensitive, poisoned_samples_ratio


def load_adult_dataset(use_gender_as_sensitive_attribute=True):
    # Load data
    df = pd.read_csv("data/adult_numerical-binsensitive.csv")

    # Extract label and sensitive attribute
    y = df["income-per-year"].to_numpy().flatten().astype(int)
    y_sensitive = df["sex"].to_numpy().flatten()
    if use_gender_as_sensitive_attribute is False:
        y_sensitive = df["race"].to_numpy().flatten()
    
    # Remove other columns and create final data set
    del df["income-per-year"];del df["sex"];del df["race"];del df["race-sex"]

    X = df.to_numpy()
    poisoned_samples_ratio = .2

    return X, y, y_sensitive, poisoned_samples_ratio


def load_givemecredit_dataset():
    # Load data
    df = pd.read_csv("data/GiveMeSomeCredit-training.csv")

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Extract label and sensitive attribute
    y = df["SeriousDlqin2yrs"].to_numpy().flatten().astype(int)
    y_sensitive = df["age"]
    y_sensitive = (y_sensitive < 35).astype(int).to_numpy().flatten().astype(int)

    # Remove other columns and create final data set
    del df["SeriousDlqin2yrs"];del df["age"]

    X = df.to_numpy()
    poisoned_samples_ratio = .05

    return X, y, y_sensitive, poisoned_samples_ratio


def load_propublica_dataset(use_gender_as_sensitive_attribute=True):
    # Load data
    df = pd.read_csv("data/propublica-violent-recidivism_numerical-binsensitive.csv")

    # Extract label and sensitive attribute
    y = df["two_year_recid"].to_numpy().flatten().astype(int)
    y_sensitive = df["sex"].to_numpy().flatten().astype(int)
    if use_gender_as_sensitive_attribute is False:
        y_sensitive = df["race"].to_numpy().flatten().astype(int)

    # Remove other columns and create final data set
    del df["two_year_recid"];del df["sex"];del df["race"];del df["sex-race"]

    X = df.to_numpy()
    poisoned_samples_ratio = .1

    return X, y, y_sensitive, poisoned_samples_ratio




# Note: Many .csv files in the data directory were downloaded from https://github.com/tailequy/fairness_dataset/tree/main/experiments/data
# Paper: https://arxiv.org/pdf/2110.00530.pdf



def load_lawSchool_dataset(use_gender_as_sensitive_attribute=True):
    # Load data
    df = pd.read_csv("data/law_school_clean.csv")
    
    # Extract label and sensitive attribute
    y = df["pass_bar"].to_numpy().flatten().astype(int)
    if use_gender_as_sensitive_attribute is True:
        y_sensitive = df["male"].to_numpy().flatten().astype(int)
    else:
        y_sensitive = df["race"]
        y_sensitive = (y_sensitive == "White").astype(int).to_numpy().flatten().astype(int)

    del df["pass_bar"]

    # Remove other columns and create final data set
    del df["male"];del df["race"]

    X = df.to_numpy()
    poisoned_samples_ratio = .05

    return X, y, y_sensitive, poisoned_samples_ratio


# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
def load_creditCardClients_dataset():
    # Load data
    df = pd.read_csv("data/credit-card-clients.csv")

    # Extract label and sensitive attribute (AGE could also be used as a sensitive attribute)
    y_sensitive = df["SEX"].to_numpy().flatten().astype(int) - 1  # Transform it to {0, 1}
    y = df["default payment"].to_numpy().flatten().astype(int)

    del df["SEX"];del df["default payment"]

    # Remove other "meaningless" columns and create final data set
    # [MARRIAGE, AGE]
    del df["MARRIAGE"];del df["AGE"]

    X = df.to_numpy()
    poisoned_samples_ratio = .05

    return X, y, y_sensitive, poisoned_samples_ratio


# https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
def load_communitiesAndCrime_dataset():
    # Load data
    df = pd.read_csv("data/communities_crime.csv")

    # Extract label and sensitive attribute
    y_sensitive = df["Black"].to_numpy().flatten().astype(int)
    y = df["class"].to_numpy().flatten().astype(int)

    del df["Black"];del df["class"]

    # Remove other "meaningless" columns and create final data set
    # [state, communityname, fold]
    del df["state"];del df["communityname"];del df["fold"]

    X = df.to_numpy()
    poisoned_samples_ratio = .5

    # Return final dataset
    return X, y, y_sensitive, poisoned_samples_ratio