import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

from automlclustering.ClusteringCS.ClusteringCS import build_config_space
from automlclustering.Helper.Helper import mf_set_to_string
from automlclustering.MetaLearning import LearningPhase
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from automlclustering.ClusterValidityIndices.CVIHandler import MLPCVI, CVICollection
from automlclustering.Optimizer.OptimizerSMAC import SMACOptimizer
from effens.EnsMetaLearning import MetaFeatureExtractor
from effens.Experiments.SyntheticData import DataGeneration
from effens.Utils.Utils import process_result_to_dataframe, clean_up_optimizer_directory


def get_similar_datasets(test_mfs_values, k):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(training_mfs.iloc[:, :-1].to_numpy())
    distances, indices = nbrs.kneighbors(test_mfs_values.reshape(1, -1))
    print(indices)
    return training_mfs.iloc[indices[0]]["dataset"]


runs = 10
ml2dac_mkr_path = Path("../automlclustering/MetaKnowledgeRepository")
effens_mkr_path = Path("../effens/EffEnsMKR")

eval_configs = pd.read_csv("../automlclustering/MetaKnowledgeRepository/evaluated_configs.csv")

print(eval_configs)
print(eval_configs.columns)

# Stats+General
mf_set = MetaFeatureExtractor.meta_feature_sets[5]

real_world_path = "/home/ubuntu/volume/real_world_data"
real_world_datasets = os.listdir(real_world_path)
print(real_world_datasets)
random_state = 1234

n_loops = 70
k_range = (2, 100)

datasets = DataGeneration.generate_datasets(n_values=[1000,
                                                      5000,
                                                      10000,
                                                      50000
                                                      ])

d_names = datasets.keys()
X_list = [v[0] for v in datasets.values()]
y_list = [v[1] for v in datasets.values()]
cvi_list = ['CH', 'DBI', 'SIL', 'DBCV',
            'DI', 'CJI', 'COP']
eval_configs = eval_configs.replace([np.inf, -np.inf], np.nan)
best_eval_configs = eval_configs.loc[eval_configs.groupby(["dataset"])["AMI"].idxmin()]
best_eval_configs["algorithm"] = best_eval_configs.apply(
    lambda x: ast.literal_eval(x["config"])["algorithm"],
    axis="columns")

result_path = Path(f"./baseline_results")

for mf_set in [MetaFeatureExtractor.meta_feature_sets[5],
               "autoclust"
               ]:
    # Extract meta-features for offline datasets
    LearningPhase.extract_all_datasets(mf_Set=mf_set, d_names=d_names, path=Path("./baseline_mfs"),
                                       datasets=X_list)

for run in range(10):
    rs = random_state * run

    # values of CVIs
    cvi_X = eval_configs[cvi_list].fillna(2147483647).to_numpy()
    cvi_y = eval_configs["AMI"]
    cvi_y = cvi_y.apply(lambda x: 1 if x > 1 else x)

    # Create and train MLP model
    mlp_model = MLPRegressor(random_state=random_state,
                             # Parameters from AutoClust paper
                             solver="adam",
                             max_iter=100,
                             activation="relu",
                             hidden_layer_sizes=(60, 30, 10)
                             )
    mlp_model.fit(cvi_X, cvi_y)

    print(f"Range of AMI values: {cvi_y.max()} - {cvi_y.min()}")
    print(mlp_model)
    print(mlp_model.predict(cvi_X[0:10]))
    print(cvi_y[0:10])
    mlp_cvi = MLPCVI(mlp_model=mlp_model)

    for mf_set in [  # MetaFeatureExtractor.meta_feature_sets[5],
        "autoclust"]:
        training_mfs = pd.read_csv(f"./baseline_mfs/{mf_set_to_string(mf_set)}_metafeatures.csv")
        print(training_mfs.to_numpy()[0].shape[0])
        for dataset in real_world_datasets:
            print("------------------------------------------------------")
            print(f"Running on dataset {dataset}")

            results = pd.DataFrame()
            if "mnist" in dataset:
                print(f"Continue for dataset {dataset}")
                continue
            df = pd.read_csv(Path(real_world_path) / dataset, header=None)
            X = df.iloc[:, :-1].to_numpy()
            y = df.iloc[:, -1].to_numpy()

            if mf_set == "autoclust":
                cvis = [mlp_cvi]
            else:
                cvis = [CVICollection.DENSITY_BASED_VALIDATION,
                        CVICollection.COP_SCORE]

            for cvi in cvis:
                method = f"AS->HPO ({mf_set_to_string(mf_set)}) - {cvi.get_abbrev()}"
                result_path_run = result_path / f"run_{run}" / method
                result_path_run.mkdir(parents=True, exist_ok=True)

                if (result_path_run / dataset).is_file():
                    print(f"Result for method = {method} already existing for {dataset}")
                    print(f"SKIPPING {dataset}")
                    continue
                additional_result_info = {"dataset": dataset,
                                          "cvi": cvi.get_abbrev(),
                                          "n": X.shape[0],
                                          "run": run
                                          }

                if mf_set == "autoclust" and X.shape[0] > 20000:
                    print(f"Continue AutoClust for now!")
                    print("We are setting default values as meanshift will not execute!")
                    test_mfs_values = np.full(training_mfs.drop("dataset", axis=1).to_numpy()[0].shape[0],
                                              2147483647)

                else:
                    mfs, test_mfs_values = MetaFeatureExtractor.extract_meta_features(X, mf_set)

                test_mfs_values = np.nan_to_num(test_mfs_values, nan=0)
                print(f"Similar Datasets:")
                similar_datasets = get_similar_datasets(test_mfs_values, k=10)

                best_conf_similar_datasets = best_eval_configs[
                    eval_configs["dataset"].isin(similar_datasets)]
                print(best_conf_similar_datasets[["dataset", "algorithm", "AMI"]])
                print(best_conf_similar_datasets["algorithm"].value_counts().idxmax())
                print(best_conf_similar_datasets["algorithm"].mode())

                best_algorithm = best_conf_similar_datasets["algorithm"].value_counts().idxmax()

                cs = build_config_space(clustering_algorithms=[best_algorithm], k_range=k_range,
                                        X_shape=X.shape)

                optimizer = SMACOptimizer(dataset=X,
                                          cvi=cvi,
                                          cs=cs,
                                          n_loops=n_loops,
                                          wallclock_limit=360 * 60,
                                          seed=random_state,
                                          limit_resources=False
                                          )
                optimizer.optimize()
                result_df = process_result_to_dataframe(optimizer,
                                                        additional_result_info,
                                                        ground_truth_clustering=y)
                clean_up_optimizer_directory(optimizer)
                result_df["Method"] = method
                results = pd.concat([results, result_df], ignore_index=True)
                results.to_csv(result_path_run / dataset, index=False)
