from sklearn.datasets import make_circles, make_blobs, fetch_openml, make_moons
# from sklearn.cluster import OPTICS
import pandas as pd
import numpy as np
import os

import ClusteringCS.ClusteringCS
from ClusterValidityIndices.CVIHandler import CVICollection
from EnsOptimizer.EnsembleOptimizer import EnsembleOptimizer
from ConsensusCS.ConsensusCS import build_consensus_cs
from Utils.Utils import process_result_to_dataframe, clean_up_optimizer_directory

if __name__ == '__main__':
    ## Parameters
    dataset_file_names = [file for file in os.listdir("/volume/datasets/synthetic") if ".csv" in file]

    k_range = (2, 100)
    n_consensus_loops = 30
    n_generation_loops = n_consensus_loops

    generation_cs = ClusteringCS.ClusteringCS.KMEANS_SPACE
    consensus_cs = build_consensus_cs(k_range=k_range)
    result_path = "../results/cf_optimization"

    # Todo: Iterate over different CVIs
    for data_file_name in dataset_file_names:

        # if not ("type=varied" in data_file_name and "n=10000-" in data_file_name):
        #     continue
        # if not ("type=moons" in data_file_name and "n=10000-" in data_file_name):
        #     continue

        if "type=varied" in data_file_name:
            cvi = CVICollection.CALINSKI_HARABASZ
        else:
            cvi = CVICollection.DENSITY_BASED_VALIDATION

        #cvi = CVICollection.DENSITY_BASED_VALIDATION
        additional_result_info = {"dataset": data_file_name, "cvi": cvi.get_abbrev()}

        df = pd.read_csv(f"/volume/datasets/synthetic/{data_file_name}", index_col=None, header=None)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        true_k = len(np.unique(y))
        n = X.shape[0]
        f = X.shape[1]
        noise = float(data_file_name.replace(".csv", "").split("-")[4].split("=")[-1])
#        print(noise)

        # TODO: Check if result already exists
        # if os.path.isfile(file_name):
        #     results = pd.read_csv(file_name)
        # else:

        ens_optimizer = EnsembleOptimizer(dataset=X, generation_cvi=cvi,
                                          cs_generation=generation_cs, cs_consensus=consensus_cs)

        gen_optimizer = ens_optimizer.optimize_generation(n_loops=n_generation_loops, k_range=k_range)
        generation_result_df = process_result_to_dataframe(gen_optimizer,
                                                           additional_result_info,
                                                           ground_truth_clustering=y)
        clean_up_optimizer_directory(gen_optimizer)
        generation_result_df["n"] = n
        generation_result_df["f"] = f
        generation_result_df["true_k"] = true_k
        generation_result_df["Method"] = "Generation"

        consensus_opt = ens_optimizer.optimize_consensus(n_loops=n_consensus_loops, k_range=k_range)
        consensus_result_df = process_result_to_dataframe(consensus_opt,
                                                          additional_result_info,
                                                          ground_truth_clustering=y)
        consensus_result_df["Method"] = "CC Optimization"
        clean_up_optimizer_directory(consensus_opt)
        result_df = pd.concat([generation_result_df, consensus_result_df])
        result_df["n"] = n
        result_df["f"] = f
        result_df["true_k"] = true_k
        result_df.to_csv(f"{result_path}/{data_file_name}", index=False)
