# Overall Evaluation of Democratizing Clustering Analyses: AutoML, Meta-Learning, and Ensemble Clustering to Support Novice Analysts

Prototypical Implementation in Python of the submitted Dissertation "Democratizing Clustering Analyses: AutoML, Meta-Learning, and Ensemble Clustering to Support Novice Analysts" at the University of Stuttgart.

## Overview

The main code is in the "src" folder. It contains the following modules:

- ``automlclustering``: Contains the code for AutoML4Clust [1] (Chapter 3) and ML2DAC [2] (Chapter 4).
- ``effens``: Code for EffEns - Efficient Ensemble Clustering [3] (Chapter 5).
- ``overall_evaluation``: Code for the overall evaluation and comparison of the three approaches AutoML4Clust, ML2DAC, and EffEns (Chapter 6).
- ``datagen_classification``: Code for the data generator and the subsequent evaluation of the three clustering approaches for subsequent classification [4] (Chapter 7).

## Installation

Our implementation is based on Python and we require Python 3.9.
Furthermore, as SMAC only runs on Linux, we also require a Linux system.
We have tested on Ubuntu 20.04.

Before installing EffEns, you first have to install the following that are required for some of the libraries:
- ``sudo apt-get install build-essential``
- ``sudo apt-get install gcc``

The easiest way of installing EffEns is to use Anaconda. Follow https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
to install Anaconda.
We will then create a prepared Python 3.9 environment:
- ``conda env create -f environment.yml``

This should create a conda environment with the name "automated_ensemble_clustering".
Then you have to install ib_base as it is not available as package: 

```git clone https://collaborating.tuhh.de/cip3725/ib_base.git
cd ib_base
python setup.py install
cd ..
```

After finishing this, you have to add the "src" folder of DissEval and the path to "ib_base" to your PYTHONPATH
You may also have to add them to your conda path
``gedit  ~/miniconda3/envs/automated_ensemble_clustering_39/lib/python3.9/site-packages/conda.pth``
or anaconda instead of miniconda.

Now everything should be setup and you can try to run ``python src/Experiments/SyntehticData/EffEns_Experiment_synthetic.py``.
This should run without any errors.

## References

[1] Fritz, M., Tschechlov, D.,& Schwarz, H. (2021). Efficient Exploratory  Clustering Analyses with Qualitative Approximations. Extending  Database Technology (EDBT), 337–342.

[2] Treder-Tschechlov, D., Fritz, M., Schwarz, H., & Mitschang, B. (2023).  ML2DAC: Meta-Learning to Democratize AutoML for Clustering Analysis.  Proceedings of the ACM on Management of Data (PACMMOD),  1(2), 1–26.

[3] Treder-Tschechlov, D., Fritz, M., Schwarz, H., & Mitschang, B. (2024).  Efficient Ensemble Clustering based on Meta-Learning and Hyperparameter  Optimization. In: To appear in Proc. VLDB Endow. 17, 11.

[4] Treder-Tschechlov, D., Reimann, P., Schwarz, H., & Mitschang, B. (2023). Approach to synthetic data generation for imbalanced multiclass  problems with heterogeneous groups. In: BTW 2023.
