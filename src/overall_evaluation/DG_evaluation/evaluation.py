import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import adjusted_mutual_info_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from automlclustering.Helper import Helper
from datagen_classification.Cluster_Classifier import BaseClassifier, ClustClassifier
from effens.EnsMetaLearning import MetaFeatureExtractor
from overall_evaluation.ExecuteClusteringApproaches import ApproachExecuter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

generated_data_path = Path("../../datagen_classification/generated")
datasets = os.listdir(generated_data_path)
runs = 10
ml2dac_mkr_path = Path("../../automlclustering/MetaKnowledgeRepository")
effens_mkr_path = Path("../../effens/EffEnsMKR")

mf_set = MetaFeatureExtractor.meta_feature_sets[5]

print(datasets)
random_state = 1234
result_path = Path("../eval_dg_results")
n_loops = 70
n_warmstarts = 25

n_loops_values = [
    # 5, 25, 50,
    70]
classifier_config_list = [{"classifier": RandomForestClassifier,
                           "parameters": {"random_state": random_state,
                                          "n_estimators": 10}},
                          {"classifier": LogisticRegression,
                           "parameters": {"random_state": random_state}},
                          # KNeighborsClassifier,
                          {"classifier": DecisionTreeClassifier,
                           "parameters": {"random_state": random_state}
                           },
                          {"classifier": DummyClassifier,
                           "parameters": {"strategy": "most_frequent",
                                          "random_state": random_state}}
                          ]


def run_classifier_approach(classifier_approach, classifier, clustering_labels, classifier_args):
    classifier_approach = classifier_approach(classifier=classifier, classifier_args=classifier_args)
    print(f"Running on {dataset}")
    classifier_approach.fit(X_train, y_train, cluster_labels=clustering_labels)
    predictions = classifier_approach.predict(X_test)

    report_df = classifier_approach.get_report_df(predictions=predictions, y_true=y_test)

    predictions_df = pd.DataFrame({"true_label": y_test, "pred_label": predictions})
    # predictions_df["accuracy"] = (df["true_labels"] == df["pred_labels"]).astype(int)

    # report_df["accuracy"] = df.groupby("true_labels")["accuracy"].mean().values

    classifier_approach.get_acc_f1_as_dic(predictions, y_test)

    accuracy_dic = classifier_approach.get_acc_f1_as_dic(predictions=predictions, y_true=y_test)
    accuracy_df = pd.DataFrame([accuracy_dic])

    print(f"Finished")
    print(accuracy_dic)
    print("-----------------------")
    return report_df, accuracy_df, predictions_df, classifier_approach


def run_classifier_configs(classifier_config_list, classifier_approach, clustering_approach, cluster_labels, rs,
                           result_path_run, dataset):
    classfier_path = result_path_run / "classification"

    for classifier_config in classifier_config_list:
        classifier = classifier_config["classifier"]
        classifier_args = classifier_config["parameters"]
        classifier_args["random_state"] = rs
        report_df, accuracy_df, \
            predictions_df, class_approach = run_classifier_approach(classifier_approach,
                                                                     classifier,
                                                                     clustering_labels=cluster_labels,
                                                                     classifier_args=classifier_args)

        for (df, path) in [(report_df, "report"), (accuracy_df, "accuracy"), (predictions_df, "predictions")]:
            (classfier_path / class_approach.classifier_name() / clustering_approach / path).mkdir(parents=True, exist_ok=True)
            df.to_csv(classfier_path / class_approach.classifier_name() / clustering_approach / path / dataset)

clust_approaches = ["effens"]
#ApproachExecuter.default_approaches
for run in list(range(runs)):
    rs = random_state * run
    result_path_run = result_path / f"run_{run}"
    result_path_run.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        print("-------------------------------------")
        print(f"Running on dataset: {dataset}")
        df = pd.read_csv(generated_data_path / dataset)

        f = int(dataset.split("_")[2].split("features")[-1])

        # Classes that have only one instance
        unique_classes = df['target'].value_counts()[df['target'].value_counts() == 1].index

        df_not_unique = df[~df['target'].isin(unique_classes)]
        df_only_unique = df[df['target'].isin(unique_classes)]

        df_train, df_test = train_test_split(df_not_unique, train_size=0.7, random_state=rs,
                                             stratify=df_not_unique["target"])
        df_train = pd.concat([df_train, df_only_unique])

        X_train = df_train[[f"F{i}" for i in range(f)]].to_numpy()
        X_test = df_test[[f"F{i}" for i in range(f)]].to_numpy()

        n = X_train.shape[0]
        assert f == X_train.shape[1]

        y_train = df_train["target"].values
        y_test = df_test["target"].values
        y_group = df_train["group"].values

        if n > 1000:
            print(f"Only using n=1k for now")
            print(f"Continue for dataset {dataset}")
            continue

        executor = ApproachExecuter(mf_set=mf_set,
                                    effens_mkr_path=effens_mkr_path,
                                    ml2dac_mkr_path=ml2dac_mkr_path,
                                    random_state=rs, result_path=result_path,
                                    approaches_to_execute=clust_approaches)

        # if (accuracy_path / dataset).is_file():
        #     print(f"Already have result {accuracy_path / dataset} for dataset {dataset}")
        #     print("skipping")
        #     print("-----------")
        #     continue

        run_classifier_configs(classifier_config_list,
                               BaseClassifier, clustering_approach="Base",
                               rs=rs, cluster_labels=None,
                               result_path_run=result_path_run,
                               dataset=dataset)

        for clustering_approach in clust_approaches:
            for n_loops in n_loops_values:
                print(f"Running {clustering_approach} on {dataset}")
                cvi, _ = executor.predict_cvi(X_train, rs)
                optimizer_instance, optimizer_result_df = executor.run_method(X=X_train, y=y_group, n_loops=n_loops,
                                                                              cvi=cvi,
                                                                              random_state=rs,
                                                                              # If we use 5 loops, also use only 5 warmstarts
                                                                              n_warmstarts=min(n_warmstarts, n_loops),
                                                                              dataset=dataset,
                                                                              result_path_=result_path_run / "clustering",
                                                                              method=clustering_approach,
                                                                              )
                cvi = optimizer_result_df["cvi"].values[0]
                # inc = optimizer_instance.get_incumbent_stats()
                result_df = optimizer_instance.get_runhistory_df()
                incs = result_df[result_df[cvi] == result_df[cvi].min()]
                cluster_labels = incs["labels"].values[0]
                print(f"AMI score for {clustering_approach} on {dataset}")
                print(adjusted_mutual_info_score(cluster_labels, y_group))
                print("------------------------------------------------------")
                # print(inc["labels"])

                run_classifier_configs(classifier_config_list, ClustClassifier,
                                       clustering_approach=clustering_approach, rs=rs,
                                       cluster_labels=cluster_labels,
                                       result_path_run=result_path_run,
                                       dataset=dataset)
