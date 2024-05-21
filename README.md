# The Effect of Data Poisoning on Counterfactual Explanations

This repository contains the implementation of the experiments as proposed in the paper [The Effect of Data Poisoning on Counterfactual Explanations](paper.pdf) by André Artelt, Shubham Sharma, Freddy Lecué, and Barbara Hammer.

## Abstract

Counterfactual explanations provide a popular method for analyzing the predictions of black-box systems, and they can offer the opportunity for computational recourse by suggesting actionable changes on how to change the input to obtain a different (i.e. more favorable) system output. However, recent work highlighted their vulnerability to different types of manipulations.

This work studies the vulnerability of counterfactual explanations to data poisoning. We formally introduce and investigate data poisoning in the context of counterfactual explanations for increasing the cost of recourse on three different levels: locally for a single instance, or a sub-group of instances, or globally for all instances.
In this context, we characterize and prove the correctness of several different data poisonings.
We also empirically demonstrate that state-of-the-art counterfactual generation methods and toolboxes are vulnerable to such data poisoning.

## Details

### Data

The data sets used in this work are stored in [Implementation/data/](Implementation/data/). Many of these .csv files in the data folder were downloaded from https://github.com/tailequy/fairness_dataset/tree/main/experiments/data.

### Experiments

Algorithm 1 for generating a poisoned training data set is implemented in [Implementation/data_poisoning.py](Implementation/data_poisoning.py) and all experiments are implemented in [Implementation/experiments.py](Implementation/experiments.py) and [Implementation/experiments_local.py](Implementation/experiments_local.py).

## Requirements

- Python 3.8
- Packages as listed in [Implementation/REQUIREMENTS.txt](Implementation/REQUIREMENTS.txt)

## License

MIT license - See [LICENSE](LICENSE).

## How to cite

You can cite the version on [arXiv](http://arxiv.org/abs/2402.08290).
