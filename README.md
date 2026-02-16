# The Effect of Data Poisoning on Counterfactual Explanations

This repository contains the implementation of the experiments as proposed in the paper [The Effect of Data Poisoning on Counterfactual Explanations](https://doi.org/10.1016/j.inffus.2026.104237) by André Artelt, Shubham Sharma, Freddy Lecué, and Barbara Hammer.

## Abstract

Counterfactual explanations are a widely used approach for examining the predictions of black-box systems. They can offer the opportunity for computational recourse by suggesting actionable changes on how to alter the input to obtain a different (i.e., more favorable) system output. However, recent studies have pointed out their susceptibility to various forms of manipulation.

This work studies the vulnerability of counterfactual explanations to data poisoning. We formally introduce and investigate data poisoning in the context of counterfactual explanations for increasing the cost of recourse on three different levels: locally for a single instance, a sub-group of instances, or globally for all instances.
In this context, we formally introduce and characterize data poisonings, from which we derive and investigate a general data poisoning mechanism.
We demonstrate the impact of such data poisoning in the critical real-world application of explaining event detections in water distribution networks. Additionally, we conduct an extensive empirical evaluation, demonstrating that state-of-the-art counterfactual generation methods and toolboxes are vulnerable to such data poisoning. Furthermore, we find that existing defense methods fail to detect those poisonous samples.

## Details

### Data

The data sets used in this work are stored in [Implementation/data/](Implementation/data/). Many of these .csv files in the data folder were downloaded from https://github.com/tailequy/fairness_dataset/tree/main/experiments/data.

The data sets for the case study on water distribution systems generated in [Implementation/wdn-casestudy.py](Implementation/wdn-casestudy.py).

### Experiments

Algorithm 1 for generating a poisoned training data set is implemented in [Implementation/data_poisoning.py](Implementation/data_poisoning.py) and the experiments from the paper are implemented in:
- [Implementation/experiments.py](Implementation/experiments.py): Experiments for global and sub-group poisoning as described in Sections 6.3.2 - 6.3.3. This file also contains a flag (```weighted_sampling```) for running the ablation study as described in Section 6.3.6.
- [Implementation/experiments_local.py](Implementation/experiments_local.py): Experiments for a local poisoning as described in Section 6.3.4.
- [Implementation/experiments_defense.py](Implementation/experiments_defense.py): Experiments for evaluating the performance of outlier detection and data sanitization methods as described in Section 6.3.5. This file also contains a flag (```weighted_sampling```) for running the ablation study as described in Section 6.3.6. The data sanizization methods are implemented in [Implementation/datasanitization.py](Implementation/datasanitization.py).
- [Implementation/experiments_labelflipping.py](Implementation/experiments_labelflipping.py): Experiments for evaluting the affect of a label flipping as described in Section 6.3.1. The label flipping attack is implemented in [Implementation/label_flipping_attack.py](Implementation/label_flipping_attack.py).
- [Implementation/wdn-casestudy.py](Implementation/wdn-casestudy.py): The case study on water distribution systems as described in Section 5. Refer to [https://github.com/andreArtelt/EnsembleConsistentExplanations](https://github.com/andreArtelt/EnsembleConsistentExplanations) for more details.


## Requirements

- Python >=3.8
- Packages as listed in [Implementation/REQUIREMENTS.txt](Implementation/REQUIREMENTS.txt)

## License

MIT license - See [LICENSE](LICENSE).

## How to cite

Please cite the Journal version [10.1016/j.inffus.2026.104237](https://doi.org/10.1016/j.inffus.2026.104237) as follows:
```bibtex
@article{Artelt2026,
  title     = {{The Effect of Data Poisoning on Counterfactual Explanations}},
  author    = "Artelt, Andr{\'e} and Sharma, Shubham and Lecu{\'e}, Freddy and Hammer, Barbara",
  doi = {10.1016/j.inffus.2026.104237},
  url = {https://doi.org/10.1016/j.inffus.2026.104237},
  journal   = "Inf. Fusion",
  publisher = "Elsevier BV",
  number    =  104237,
  pages     = "104237",
  month     =  feb,
  year      =  2026,
  language  = "en"
}
