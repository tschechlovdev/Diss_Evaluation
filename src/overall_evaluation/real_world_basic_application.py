import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from automlclustering.MetaLearning.ML2DAC import ML2DAC
from effens.EnsMetaLearning import MetaFeatureExtractor
from effens.EnsMetaLearning.EffEns import EffEns
from effens.Utils.Utils import process_result_to_dataframe, clean_up_optimizer_directory, calculate_gen_info


def result_already_exists(results, test_dataset, approach, cvi):
    if "dataset" in results.columns:
        dataset_results = results[results["dataset"] == str(test_dataset)]
        if len(dataset_results[
                   (dataset_results["Method"] == approach) & (dataset_results["cvi"] == cvi.get_abbrev())]) > 0:
            return True
    return False


def _run_ml2dac(X, loops, warmstarts, cvi, result_path_, dataset, random_state):
    # if X.shape[0] > 20000 and cvi in ["DBCV", "COP"]:
    #     print(f"n={X.shape} and cvi={cvi}. This will result in OOM.")
    #     print(f"Using SIL instead")
    #     cvi = "SIL"
    print(f"Running {result_path_.name}")
    if warmstarts == 0:
        limit_cs = False
    else:
        limit_cs = True

    result_path_.mkdir(parents=True, exist_ok=True)
    ml2dac = ML2DAC(mf_set=mf_set, mkr_path=ml2dac_mkr_path, random_state=random_state)
    optimizer_instance, additional_info = ml2dac.optimize_with_meta_learning(X, dataset_name=dataset,
                                                                             cvi=CVICollection.get_cvi_by_abbrev(cvi),
                                                                             n_optimizer_loops=loops,
                                                                             n_warmstarts=warmstarts,
                                                                             limit_cs=limit_cs)
    additional_info["cvi"] = cvi
    additional_info["Method"] = result_path_.name

    optimizer_result_df = process_result_to_dataframe(optimizer_instance, additional_info, y)
    clean_up_optimizer_directory(optimizer_instance)
    optimizer_result_df.to_csv(result_path_ / dataset, index=False)


def _run_effens(X, n_loops_effens, cvi, result_path_, dataset, rs):
    # if X.shape[0] > 30000 and cvi in ["DBCV"]:
    #     print(f"n={X.shape} and cvi={cvi}. This will result in OOM.")
    #     print(f"Using SIL instead")
    #     cvi = "SIL"
    print(f"Running {result_path_.name}")
    result_path_.mkdir(parents=True, exist_ok=True)

    effens = EffEns(mkr_path=effens_mkr_path, mf_set=mf_set, random_state=rs)
    effens.mf_scaler = scaler
    consensus_optimizer, additional_info = effens.apply_ensemble_clustering(X=X, cvi=cvi,
                                                                            n_loops=n_loops_effens)
    additional_info["cvi"] = cvi
    selected_ens = effens.get_ensemble()
    gen_info = calculate_gen_info(X, y, effens, cvi)
    additional_info.update(gen_info)
    additional_info.update({"dataset": dataset,
                            "Method": result_path_.name,
                            "run": run,
                            "rs": rs})
    optimizer_result_df = process_result_to_dataframe(consensus_optimizer, additional_info, y)
    clean_up_optimizer_directory(consensus_optimizer)
    optimizer_result_df.to_csv(result_path_ / dataset, index=False)


runs = 10
ml2dac_mkr_path = Path("../automlclustering/MetaKnowledgeRepository")
effens_mkr_path = Path("../effens/EffEnsMKR")

mf_set = MetaFeatureExtractor.meta_feature_sets[5]

real_world_path = "/home/ubuntu/volume/real_world_data"
datasets = os.listdir(real_world_path)
print(datasets)
random_state = 1234

# Have to scale meta-features and use this scaling for future datasets
mfs = pd.read_csv(effens_mkr_path / "meta_features.csv")
mfs = mfs.drop("dataset", axis=1)
result_path = Path("./eval_results")

scaler = MinMaxScaler()
mfs = scaler.fit_transform(mfs.to_numpy())
n_loops = 100
n_warmstarts = 50
n_loops_effens = 70


def run_method(X, n_loops, n_warmstarts, cvi, result_path_, method, dataset, random_state):
    # Predict CVI for ML2DAC and EffEns

    if method == "ml2dac":
        _run_ml2dac(X, n_loops, n_warmstarts, cvi, result_path_ / method, dataset, random_state)
    elif method == "effens":
        if X.shape[0] > 30000 and cvi in ["DBCV"]:
            # COP should be possible with n=70k!
            cvi = "SIL"
        _run_effens(X, n_loops_effens, cvi, result_path_ / method, dataset, random_state)

    else:
        # Parse string "method (cvi)"
        method_run = method[0: method.find("(")]
        cvi = method[method.find("(") + 1:method.find(")")]

        if "effens" in method_run:
            _run_effens(X, n_loops_effens, cvi, result_path_ / method, dataset, random_state)
        elif "aml4c" in method_run:
            _run_ml2dac(X, n_loops, 0, cvi, result_path_ / method, dataset, random_state)
        else:
            raise ValueError(f"Unknown method {method}")
        # if cvi == "no warm":
        #     # special case where we want to run no warmstarts
        #     _run_ml2dac(X, n_loops, 0, cvi, result_path_ / method, dataset)
        # else:


for run in list(range(runs)):
    rs = random_state * run
    result_path_run = result_path / f"run_{run}"
    result_path_run.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        print("------------------------------------------------------")
        df = pd.read_csv(Path(real_world_path) / dataset, header=None)
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()

        print(f"Running on dataset: {dataset} with shape {X.shape}")
        n = X.shape[0]

        # if X.shape[0] > 20000:
        #     print(f"Skipping {dataset} for now as it has n={X.shape[0]} > 20k instances")
        #     continue

        mf_names, mf_X = MetaFeatureExtractor.extract_meta_features(mf_set=mf_set, dataset=X)
        mf_X = scaler.transform(mf_X.reshape(1, -1))
        print(mf_set)
        ml2dac = ML2DAC(mf_set=mf_set, mkr_path=ml2dac_mkr_path, random_state=rs)
        cvi = ml2dac.cvi_predictor.predict(mf_X.reshape(1, -1))[0]
        cvis_probs = ml2dac.cvi_predictor.predict_proba(mf_X.reshape(1, -1))[0]
        cvis = ml2dac.cvi_predictor.classes_

        # Todo: Wir könnten sagen, dass wir unter bestimmten Umständen eine andere CVI nehmen möchten
        cvis_ordered = [cvi for _, cvi in sorted(zip(cvis_probs, cvis), reverse=True)]
        print(f"Using cvi: {cvi}")
        print(sorted(zip(cvis_probs, cvis)))
        print(f"Predicted CVIs, ordered by model probability: {cvis_ordered}")

        for method in [
            "effens (DBCV)",
            "ml2dac", "effens",
            "effens (COP)",
            "aml4c (COP)",
            "aml4c (DBCV)",
            #          "aml4c (SIL)",
            #          "aml4c (CH),
            #         "effens (SIL)",
            # "effens (CH)",
            # "ml2dac (no warm)",
        ]:

            if (result_path_run / method / dataset).is_file():
                print(f"Result for method = {method} already existing for {dataset}")
                print(f"SKIPPING {dataset}")
            else:
                print(f"Running method: {method}")
                run_method(X, n_loops, n_warmstarts, cvi, result_path_run, method, dataset, random_state)
                print("------------------------------------------------------")
                print("------------------------------------------------------")

print("CONGRATS! :-)")
print("Script Finished! :) :) :) :) :) :) :) :) ")
