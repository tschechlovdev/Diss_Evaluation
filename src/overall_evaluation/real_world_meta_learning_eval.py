import os
from pathlib import Path

import pandas as pd
from effens.Utils import Utils

from automlclustering.Helper import Helper
from overall_evaluation.ExecuteClusteringApproaches import ApproachExecuter

from effens.EnsMetaLearning import MetaFeatureExtractor


def store_training_sets_mkr(path, file_name):
    df = pd.read_csv(path / file_name)

    for training_set in eval_training_sets:
        result_path = path / "+".join(training_set)
        result_path.mkdir(parents=True, exist_ok=True)
        (result_path / "meta_features").mkdir(parents=True, exist_ok=True)

        df["type"] = df["dataset"].apply(lambda x: Utils.get_type_from_dataset(x))
        df_training_set = df[df["type"].isin(training_set)]
        print(df_training_set["type"].unique())
        print(len(df_training_set["dataset"].unique()))
        df_training_set = df_training_set.drop("type", axis=1)
        df_training_set.to_csv(result_path / file_name, index=False)


if __name__ == '__main__':
    mf_set = MetaFeatureExtractor.meta_feature_sets[5]
    real_world_path = "/home/ubuntu/volume/real_world_data"
    datasets = os.listdir(real_world_path)
    print(datasets)
    random_state = 1234
    n_loops = 70
    n_warmstarts = 50
    n_loops_effens = 70

    eval_training_sets = [
        ["gaussian"],
        ["varied"],
        #["moons"],
        #["circles"],
        ["moons", "circles"],
        ["gaussian", "varied"],
        ["gaussian", "moons", "circles"],
        ["varied", "moons", "circles"],
        ["gaussian", "varied", "moons", "circles"],
    ]

    ml2dac_mkr_path = Path("../automlclustering/MetaKnowledgeRepository")
    effens_mkr_path = Path("../effens/EffEnsMKR")

    # Create new MKR paths, one path for each training set!
    for file_name in ["evaluated_configs.csv",
                      "meta_features/statistical+general_metafeatures.csv",
                      "optimal_cvi.csv"]:
        store_training_sets_mkr(ml2dac_mkr_path, file_name)

    for file_name in ["evaluated_ensemble.csv", "meta_features.csv"]:
        store_training_sets_mkr(effens_mkr_path, file_name)

    runs = 1
    ml2dac_mkr_path = Path("../automlclustering/MetaKnowledgeRepository")
    effens_mkr_path = Path("../effens/EffEnsMKR")

    for training_set in eval_training_sets:
        training_set_string = "+".join(training_set)
        print("-----------------------------")
        print(f"Running Training set {training_set}")
        result_path = Path("./eval_mtl_results") / training_set_string

        executor = ApproachExecuter(mf_set=mf_set, random_state=random_state,
                                    # Adapt paths of ML2DAC and EffEns for specific training set
                                    ml2dac_mkr_path=ml2dac_mkr_path / training_set_string,
                                    effens_mkr_path=effens_mkr_path / training_set_string,
                                    result_path=result_path,
                                    approaches_to_execute=["ml2dac", "effens"])

        executor.execute_all_approaches(datasets=datasets,
                                        n_loops=n_loops, n_warmstarts=n_warmstarts,
                                        path_to_datasets=real_world_path,
                                        runs=runs)
        print(f"Finished Training set {training_set}")
        print("-----------------------------")
        print("-----------------------------")

    print("CONGRATS! :-)")
    print("Script Finished! :) :) :) :) :) :) :) :) ")
