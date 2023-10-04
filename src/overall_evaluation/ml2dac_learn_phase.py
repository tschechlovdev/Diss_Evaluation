from pathlib import Path

from automlclustering.MetaLearning import MetaFeatureExtractor

from automlclustering.MetaLearning.LearningPhase import run_learning_phase
from effens.Experiments.SyntheticData import DataGeneration

datasets = DataGeneration.generate_datasets(n_values=[1000,
                                                      5000,
                                                      10000,
                                                      50000
                                                      ])

d_names = datasets.keys()
X_list = [v[0] for v in datasets.values()]
y_list = [v[1] for v in datasets.values()]

ml2dac_mkr_path = Path("../automlclustering/MetaKnowledgeRepository")
run_learning_phase(training_datasets=X_list, training_data_labels=y_list,
                   training_dataset_names=d_names,
                   n_loops=100, mf_set=MetaFeatureExtractor.meta_feature_sets[5], time_limit=240 * 60,
                   mkr_path=ml2dac_mkr_path)
