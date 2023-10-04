import argparse
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

from automlclustering.ClusterValidityIndices.CVIHandler import CVICollection
from automlclustering.ClusteringCS import ClusteringCS
from automlclustering.MetaLearningExperiments import DataGeneration
from automlclustering.MetaLearning import MetaFeatureExtractor
from automlclustering.MetaLearning.MetaFeatureExtractor import extract_all_datasets
from automlclustering.Optimizer.OptimizerSMAC import SMACOptimizer
from automlclustering.Helper import Helper
from automlclustering.Helper.Helper import mf_set_to_string
from effens.Utils.Utils import clean_up_optimizer_directory, get_n_from_dataset

# Has to be adjusted according to your working directory!
# mkr_path = Path("src/EffEnsMKR")
evaluated_configs_filename = "evaluated_configs.csv"
optimal_metric_file_name = "optimal_cvi.csv"

# Preparation, datasets to use and meta-features
meta_feature_sets = MetaFeatureExtractor.meta_feature_sets

# different_shape_sets = DataGeneration.generate_datasets()
# d_names = list(different_shape_sets.keys())
# datasets = [X for X, y in different_shape_sets.values()]
# true_labels = [y for X, y in different_shape_sets.values()]

# define random seed
np.random.seed(1234)


def get_best_cvi_for_dataset(meta_knowledge, dataset, n_top_results=None):
    """
    Returns the cvi that is best suited for the given dataset.
    We determine this by calculating the spearman correlation between the ARI and each CVI and return the CVI with the
     highest correlation.
    :param meta_knowledge: dataframe of the executed configurations and their evaluations.
     Might contain more than the information for the given dataset.
    :param dataset: name of the dataset to use.
    :return:(best_cvi, best_corr, correlation_by_metric);
     best_cvi: cvi abbrev of the best cvi, best_corr: spearman correlation of the best cvi, and
    , correlation_by_metric (dictionary with correlation for each cvi as value and
    cvi abbrev as key)
    """
    if dataset:
        meta_knowledge_for_data = meta_knowledge[meta_knowledge["dataset"] == dataset]
    else:
        meta_knowledge_for_data = meta_knowledge

    best_corr = -np.infty
    best_cvi = ""
    correlation_by_metric = {}
    for cvi in CVICollection.internal_cvis:
        if n_top_results:
            top_cvi_configs = meta_knowledge_for_data.sort_values(cvi.get_abbrev(), ascending=True)[
                              0:n_top_results]
        else:
            top_cvi_configs = meta_knowledge_for_data

        ari_values = top_cvi_configs["AMI"].to_numpy()
        cvi_values = top_cvi_configs[cvi.get_abbrev()].to_numpy()
        corr, _ = spearmanr(ari_values, cvi_values, nan_policy='omit')
        correlation_by_metric[cvi.get_abbrev()] = corr

        if np.isnan(corr) & (len(ari_values) == 1) & (ari_values[0] == -1.0):
            corr = 1

        if corr > best_corr:
            best_corr = corr
            best_cvi = cvi.get_abbrev()
    return best_cvi, best_corr, correlation_by_metric


# TODO: We might test instead of ranking to select the CVI with best AMI value (what about ties?)
def select_best_cvi(mkr_path, ranking=True):
    """
    Determines the best metric for each dataset and stores the result in a csv file.
    :return: dataset_optimal_cvi: pd.DataFrame that contains for each dataset the metric that is best suited.
    """
    evaluated_configs = pd.read_csv(mkr_path / evaluated_configs_filename, index_col=0)
    dataset_optimal_cvi = pd.DataFrame(columns=["cvi", "dataset", "correlations"])

    for dataset in evaluated_configs["dataset"].unique():
        if ranking:
            best_cvi, best_corr, correlation_by_metric = get_best_cvi_for_dataset(evaluated_configs, dataset)

        else:
            ec_dataset = evaluated_configs[evaluated_configs["dataset"] == dataset]
            best_cvi = ""
            best_ami_cvi = np.infty
            ami_per_cvi = {}
            for cvi in CVICollection.internal_cvis:
                cvi = cvi.get_abbrev()
                ami_value = ec_dataset[ec_dataset[cvi] == ec_dataset[cvi].min()]["AMI"].min()
                if ami_value < best_ami_cvi:
                    best_ami_cvi = ami_value
                    best_cvi = cvi
                ami_per_cvi[cvi] = best_ami_cvi

            correlation_by_metric = ami_per_cvi

        print(f"best cvi for {dataset} is: {best_cvi}")
        dataset_optimal_cvi = pd.concat([dataset_optimal_cvi,
                                         pd.DataFrame([{"cvi": best_cvi, "dataset": dataset,
                                                        "correlations": correlation_by_metric}])],
                                        ignore_index=True)

    dataset_optimal_cvi.to_csv(mkr_path / optimal_metric_file_name)
    return dataset_optimal_cvi


def train_model_not_for_dataset(dataset, mf_set, name_meta_feature_set, optimal_cvi, prediction_cols, classifier,
                                mkr_path, random_state=1234):
    # Prepare Training data for this dataset (i.e. all datasets except this dataset)
    mf_set_for_dataset = mf_set[mf_set["dataset"] != dataset]
    optimal_cvi_for_dataset = optimal_cvi[optimal_cvi["dataset"] != dataset]
    X = mf_set_for_dataset[prediction_cols]
    X = np.nan_to_num(X, 0)
    y = optimal_cvi_for_dataset["cvi"].to_numpy()

    # instantiate and train classifier
    classifier_instance = classifier(random_state=random_state)
    model_name = Helper.get_model_name(classifier_instance)
    classifier_instance.fit(X, y)

    # save trained classifer
    classifier_directory = mkr_path / "models" / model_name / name_meta_feature_set
    classifier_directory.mkdir(exist_ok=True, parents=True)
    with open(f'{classifier_directory}/{dataset}', 'wb') as f:
        joblib.dump(classifier_instance, f)
        return classifier_instance


def train_classifier(mkr_path, classifier=RandomForestClassifier, mf_set=MetaFeatureExtractor.meta_feature_sets[5],
                     random_state=1234):
    """
    Trains a classifier on the meta-features as X and the best metric for the according dataset are the labels.
    The classifier can be used to predict for new datasets, based on their meta-features, the best CVI.
    We train one classifier for all datasets, everytime except one dataset (i.e., L-O-O cross validation).
    We store the results by the name of the dataset that is left out. This way, we can retrieve the classifier
    in the application phase for the new unseen dataset by its name.
    :param classifier: sklearn estimator class
    :return:
    """
    optimal_cvi = pd.read_csv(mkr_path / optimal_metric_file_name)

    meta_feature_path = mkr_path / "meta_features"

    name_meta_feature_set = mf_set_to_string(mf_set)
    mf_set = pd.read_csv(f"{meta_feature_path}/{name_meta_feature_set}_metafeatures.csv")

    mf_set['dataset'] = pd.Categorical(mf_set['dataset'], optimal_cvi["dataset"])
    mf_set = mf_set.sort_values("dataset")
    prediction_cols = [col for col in mf_set.columns if col not in ["dataset", "mfe"]]

    # Train a model for all datasets
    # We use this for "new" application phases
    clf = train_model_not_for_dataset(dataset=None, mf_set=mf_set, name_meta_feature_set=name_meta_feature_set,
                                      optimal_cvi=optimal_cvi, prediction_cols=prediction_cols,
                                      classifier=classifier,
                                      mkr_path=mkr_path, random_state=random_state)
    return clf


def evalute_configurations(n_loops, time_limit, datasets, dataset_names, dataset_labels, mkr_path,
                           skip_existing_results=True):
    if (mkr_path / evaluated_configs_filename).is_file():
        print(f"got file {mkr_path / evaluated_configs_filename} - skipping results if possible")
        evaluated_configs_df = pd.read_csv(mkr_path / evaluated_configs_filename)
    else:
        evaluated_configs_df = pd.DataFrame()

    for X, y, dataset_name in zip(datasets, dataset_labels, dataset_names):
        sys.stdout.flush()
        print("-------------------")
        print(f"Running on dataset: {dataset_name}")

        if skip_existing_results and "dataset" in evaluated_configs_df.columns:
            if dataset_name in evaluated_configs_df["dataset"].unique():
                print(f"Result for {dataset_name} already existing - skipping")
                continue
            else:
                print(f"Result for {dataset_name} does not exist yet")
        # min_max_scaler = preprocessing.MinMaxScaler()
        # X = min_max_scaler.fit_transform(X)

        # we build a config space with the hyperparameters for each algorithm separately
        cs_all_algos = ClusteringCS.build_all_algos_space(X_shape=X.shape)
        opt_result_df = pd.DataFrame()

        # RunOptimizationProcedure
        opt_instance = SMACOptimizer(dataset=X, true_labels=y, cvi=CVICollection.ADJUSTED_MUTUAL,
                                     n_loops=n_loops, cs=cs_all_algos, wallclock_limit=time_limit)
        opt_instance.optimize()
        algo_result_df = opt_instance.get_runhistory_df()
        algo_result_df['dataset'] = dataset_name

        # todo: don't need labels for AutoCluster
        algo_result_df = algo_result_df.drop("labels", axis=1)

        opt_result_df = pd.concat([opt_result_df, algo_result_df])
        evaluated_configs_df = pd.concat([evaluated_configs_df, opt_result_df])

        # save after each execution of algorithm for each dataset --> Can have intermediate results if one run crashes
        evaluated_configs_df.to_csv(mkr_path / evaluated_configs_filename, index=False)
        clean_up_optimizer_directory(optimizer_instance=opt_instance)
        print(f"Finished dataset: {dataset_name}")
        # print(f"Finished dataset number {dataset_names.index(dataset_name)}/{len(dataset_names)}")
        print("---------------------------------")


def run_learning_phase(training_datasets, training_data_labels, training_dataset_names, n_loops=100,
                       time_limit=120 * 60, mf_set="statistical", skip_mf_extraction=False,
                       mkr_path=Path("")):
    """
    Runs learning phase of our approach
    :param extract_mfs: Boolean, whether to extract the meta-features or not
    :param n_loops: number of optimizer loops to perform for each dataset
    :param time_limit: Time limit of the optimization procedure. The default is two hours
    :param mf_path: Path where to store the meta-features

    Args:
        mkr_path:
    """
    meta_feature_path = mkr_path / "meta_features"

    #########################################################################################################
    # L1: ExtractMeta-features
    if not skip_mf_extraction:
        print(training_dataset_names)
        extract_all_datasets(datasets=training_datasets, path=meta_feature_path,
                             mf_Set=mf_set,
                             d_names=training_dataset_names,
                             save_metafeatures=True)
    #########################################################################################################

    #########################################################################################################
    # L2: Evaluate Configurations
    evalute_configurations(n_loops=n_loops,
                           time_limit=time_limit,
                           datasets=training_datasets,
                           dataset_names=training_dataset_names,
                           dataset_labels=training_data_labels,
                           mkr_path=mkr_path)
    #########################################################################################################

    #########################################################################################################
    # L3: Select Best CVI
    select_best_cvi(mkr_path=mkr_path)
    #########################################################################################################

    #########################################################################################################
    # L4: Train Classification Model
    train_classifier(mkr_path)
    #########################################################################################################

    print("Finished Learning Phase!")
    sys.stdout.flush()


if __name__ == '__main__':
    #########################################################################################################
    ############################### Parameter Specification #################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", help="Option for running on specific dataset types. "
                                               "Per default, all synthetic dataset types are used.",
                        nargs='+', default=DataGeneration.DATASET_TYPES,
                        choices=DataGeneration.DATASET_TYPES + ['all'])
    parser.add_argument("--n_loops",
                        help="Specifies the number of configurations/optimizer loops that should be executed."
                             " Default are 100 loops", default=100)
    parser.add_argument("--time_limit", help="Defines the runtime of the optimization procedure."
                                             " Per default the time limit is 2 hours (120 * 60 s)",
                        default=120 * 60)
    # parser.add_argument("--output_dir", help="Specify output directory of the results.", default="", required=False)
    args = parser.parse_args()

    dataset_types = args.dataset_type
    if dataset_types == ['all']:
        dataset_types = DataGeneration.DATASET_TYPES

    n_loops = args.n_loops
    time_limit = args.time_limit
    #########################################################################################################

    # Use this meta-feature set
    mf_set: list[str] = ["statistical", "info-theory", "general"]

    # for mf_set in MetaFeatureExtractor.meta_feature_sets:
    # Run the learning phase
    # run_learning_phase(mf_set=mf_set, n_loops=n_loops,
    #                    training_datasets=datasets,
    #                    training_dataset_names=d_names,
    #                    training_data_labels=true_labels)
