import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from effens.EnsMetaLearning import MetaFeatureExtractor
from overall_evaluation.ExecuteClusteringApproaches import ApproachExecuter

if __name__ == '__main__':
    runs = 10
    ml2dac_mkr_path = Path("../automlclustering/MetaKnowledgeRepository")
    effens_mkr_path = Path("../effens/EffEnsMKR")

    mf_set = MetaFeatureExtractor.meta_feature_sets[5]

    real_world_path = "/home/ubuntu/volume/real_world_data"
    datasets = os.listdir(real_world_path)
    print(datasets)
    random_state = 1234

    result_path = Path("./eval_results")

    n_loops = 70
    n_warmstarts = 50
    n_loops_effens = 70

    executor = ApproachExecuter(mf_set=mf_set, random_state=random_state, ml2dac_mkr_path=ml2dac_mkr_path,
                                effens_mkr_path=effens_mkr_path,
                                result_path=result_path)

    executor.execute_all_approaches(datasets=datasets, n_loops=n_loops, n_warmstarts=n_warmstarts,
                                    path_to_datasets=real_world_path, runs=10)
    print("CONGRATS! :-)")
    print("Script Finished! :) :) :) :) :) :) :) :) ")
