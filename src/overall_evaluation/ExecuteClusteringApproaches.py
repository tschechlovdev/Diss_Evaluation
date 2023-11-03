import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from automlclustering.MetaLearning.ML2DAC import ML2DAC
from effens.EnsMetaLearning import MetaFeatureExtractor
from effens.EnsMetaLearning.EffEns import EffEns
from effens.Utils.Utils import process_result_to_dataframe, clean_up_optimizer_directory, calculate_gen_info


class ApproachExecuter:
    default_approaches = [
        "ml2dac", "effens",
        "effens (COP)",
        "effens (DBCV)",
        "aml4c (COP)",
        "aml4c (DBCV)",
        #          "aml4c (SIL)",
        #          "aml4c (CH),
        #         "effens (SIL)",
        # "effens (CH)",
        # "ml2dac (no warm)",
    ]

    def __init__(self, mf_set=MetaFeatureExtractor.meta_feature_sets[5],
                 random_state=1234,
                 ml2dac_mkr_path=Path("../automlclustering/MetaKnowledgeRepository"),
                 effens_mkr_path=Path("../effens/EffEnsMKR"),
                 result_path=Path("./eval_results"),
                 approaches_to_execute=default_approaches
                 ):
        # Have to scale meta-features and use this scaling for future datasets
        mfs = pd.read_csv(effens_mkr_path / "meta_features.csv")
        mfs = mfs.drop("dataset", axis=1)
        print(mfs)

        scaler = MinMaxScaler()
        self.mfs = scaler.fit_transform(mfs.to_numpy())
        self.scaler = scaler
        self.result_path = result_path
        self.ml2dac_mkr_path = ml2dac_mkr_path
        self.effens_mkr_path = effens_mkr_path
        self.random_state = random_state
        self.mf_set = mf_set
        if approaches_to_execute:
            self.default_approaches = approaches_to_execute
        else:
            self.default_approaches = ApproachExecuter.default_approaches


    @staticmethod
    def result_already_exists(results, test_dataset, approach, cvi):
        if "dataset" in results.columns:
            dataset_results = results[results["dataset"] == str(test_dataset)]
            if len(dataset_results[
                       (dataset_results["Method"] == approach) & (dataset_results["cvi"] == cvi.get_abbrev())]) > 0:
                return True
        return False

    def _run_ml2dac(self, X, y, loops, warmstarts, cvi, result_path_, dataset, random_state):
        print(f"Running {result_path_.name}")
        if warmstarts == 0:
            limit_cs = False
        else:
            limit_cs = True

        result_path_.mkdir(parents=True, exist_ok=True)
        ml2dac = ML2DAC(mf_set=self.mf_set, mkr_path=self.ml2dac_mkr_path, random_state=random_state)
        optimizer_instance, additional_info = ml2dac.optimize_with_meta_learning(X, dataset_name=dataset,
                                                                                 cvi=CVICollection.get_cvi_by_abbrev(
                                                                                     cvi),
                                                                                 n_optimizer_loops=loops,
                                                                                 n_warmstarts=warmstarts,
                                                                                 limit_cs=limit_cs)
        additional_info["cvi"] = cvi
        additional_info["Method"] = result_path_.name
        additional_info["n_loops"] = loops

        optimizer_result_df = process_result_to_dataframe(optimizer_instance, additional_info, y)
        clean_up_optimizer_directory(optimizer_instance)
        optimizer_result_df.to_csv(result_path_ / dataset, index=False)
        return optimizer_instance, optimizer_result_df

    def _run_effens(self, X, y, n_loops_effens, cvi, result_path_, dataset, rs):
        print(f"Running {result_path_.name}")
        result_path_.mkdir(parents=True, exist_ok=True)

        effens = EffEns(mkr_path=self.effens_mkr_path, mf_set=self.mf_set, random_state=rs)
        effens.mf_scaler = self.scaler
        consensus_optimizer, additional_info = effens.apply_ensemble_clustering(X=X, cvi=cvi,
                                                                                n_loops=n_loops_effens)
        additional_info["cvi"] = cvi
        selected_ens = effens.get_ensemble()
        try:
            gen_info = calculate_gen_info(X, y, effens, cvi)
        except MemoryError as e:
            print(f"Error: {e}")
            gen_info = {"gen_error": str(e)}

        additional_info.update(gen_info)
        additional_info.update({"dataset": dataset,
                                "Method": result_path_.name,
                                "n_loops": n_loops_effens,
                                # "run": run,
                                "rs": rs})
        optimizer_result_df = process_result_to_dataframe(consensus_optimizer, additional_info, y)
        clean_up_optimizer_directory(consensus_optimizer)
        optimizer_result_df.to_csv(result_path_ / dataset, index=False)
        return consensus_optimizer, optimizer_result_df

    def run_method(self, X, y, n_loops, n_warmstarts, cvi, result_path_, method, dataset, random_state):
        if method == "ml2dac":
            optimizer_instance, optimizer_result_df = self._run_ml2dac(X, y, n_loops, n_warmstarts, cvi,
                                                                       result_path_ / method, dataset, random_state)
        elif method == "effens":
            optimizer_instance, optimizer_result_df = self._run_effens(X, y, n_loops, cvi,
                                                                       result_path_ / method, dataset, random_state)

        else:
            # Parse string "method (cvi)"
            method_run = method[0: method.find("(")]
            cvi = method[method.find("(") + 1:method.find(")")]

            if "effens" in method_run:
                optimizer_instance, optimizer_result_df = self._run_effens(X, y, n_loops, cvi, result_path_ / method,
                                                                           dataset, random_state)
            elif "aml4c" in method_run:
                optimizer_instance, optimizer_result_df = self._run_ml2dac(X, y, n_loops, 0, cvi, result_path_ / method,
                                                                           dataset, random_state)
            else:
                raise ValueError(f"Unknown method {method}")
            # if cvi == "no warm":
            #     # special case where we want to run no warmstarts
            #     _run_ml2dac(X, n_loops, 0, cvi, result_path_ / method, dataset)
            # else:
        return optimizer_instance, optimizer_result_df

    def predict_cvi(self, X, rs):
        mf_names, mf_X = MetaFeatureExtractor.extract_meta_features(mf_set=self.mf_set, dataset=X)
        mf_X = self.scaler.transform(mf_X.reshape(1, -1))
        print(self.mf_set)
        ml2dac = ML2DAC(mf_set=self.mf_set, mkr_path=self.ml2dac_mkr_path, random_state=rs)
        cvi = ml2dac.cvi_predictor.predict(mf_X.reshape(1, -1))[0]
        cvis_probs = ml2dac.cvi_predictor.predict_proba(mf_X.reshape(1, -1))[0]
        cvis = ml2dac.cvi_predictor.classes_

        # Todo: Wir könnten sagen, dass wir unter bestimmten Umständen eine andere CVI nehmen möchten
        cvis_ordered = [cvi for _, cvi in sorted(zip(cvis_probs, cvis), reverse=True)]
        print(f"Using cvi: {cvi}")
        print(sorted(zip(cvis_probs, cvis)))
        print(f"Predicted CVIs, ordered by model probability: {cvis_ordered}")
        return cvi, cvis_ordered

    def execute_all_approaches(self, datasets, path_to_datasets,
                               n_loops, n_warmstarts,
                               runs=10):
        for run in list(range(runs)):
            rs = self.random_state * run
            result_path_run = self.result_path / f"run_{run}"
            result_path_run.mkdir(parents=True, exist_ok=True)

            for dataset in datasets:
                print("------------------------------------------------------")
                df = pd.read_csv(Path(path_to_datasets) / dataset, header=None)
                X = df.iloc[:, :-1].to_numpy()
                y = df.iloc[:, -1].to_numpy()

                print(f"Running on dataset: {dataset} with shape {X.shape}")
                n = X.shape[0]

                if "mnist" in dataset:
                    print(f"Continue for dataset {dataset}")
                    continue

                if n > 10000 and "Fashion-MNIST" in dataset:
                    print(f"Continue for {n} > 10k for F-MNIST dataset for now!")
                    print(f"Continue for {dataset}")
                    continue

                cvi, cvis_ordered = self.predict_cvi(X, rs)

                for method in self.default_approaches:
                    if (result_path_run / method / dataset).is_file():
                        print(f"Result for method = {method} already existing for {dataset}")
                        print(f"SKIPPING {dataset}")
                    else:
                        print(f"Running method: {method}")
                        self.run_method(X, y, n_loops, n_warmstarts, cvi, result_path_run, method, dataset, rs)
                        print("------------------------------------------------------")
                print("------------------------------------------------------")
