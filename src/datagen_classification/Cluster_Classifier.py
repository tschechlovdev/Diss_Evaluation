import numpy as np
import pandas as pd
import sklearn.base
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, balanced_accuracy_score, precision_score, \
    recall_score, auc, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


class BaseClassifier:
    def __init__(self, classifier: sklearn.base.BaseEstimator.__class__, classifier_args):
        self.classifier = classifier
        self.classifier_args = classifier_args
        self.clustering = False

    def fit(self, X_train, y_train, cluster_labels):
        self.classifier = self.classifier(**self.classifier_args)
        print(f"Fitting {self.classifier} with parameters {self.classifier_args}")
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def get_report_df(self, predictions, y_true):
        report = classification_report(y_pred=predictions, y_true=y_true, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        return report_df

    def get_acc_f1_as_dic(self, predictions, y_true):
        return {"accuracy": accuracy_score(predictions, y_true),
                "F1_score": f1_score(y_pred=predictions, y_true=y_true, average="macro",
                                     labels=np.unique(predictions)),
                "balanced_acc": balanced_accuracy_score(y_pred=predictions, y_true=y_true),
                "precision": precision_score(y_pred=predictions, y_true=y_true, average="macro",
                                             labels=np.unique(predictions)),
                "recall": recall_score(y_pred=predictions, y_true=y_true, average="macro",
                                       labels=np.unique(predictions)),
                # "auc": roc_auc_score(y_score=predictions, y_true=y_true)
                }

    def name(self):
        return self.__class__.__name__ + f" ({self.classifier.__class__.__name__})"

    def classifier_name(self):
        return self.classifier.__class__.__name__


class ClustClassifier(BaseClassifier):
    def __init__(self, classifier: sklearn.base.BaseEstimator.__class__,
                 cluster_predictor: sklearn.base.BaseEstimator.__class__ = KNeighborsClassifier,
                 classifier_args=None):
        super().__init__(classifier, classifier_args=classifier_args)
        self.cluster_predictor = cluster_predictor()
        self.model_repo = {}
        self.clustering = True

    def fit(self, X_train, y_train, cluster_labels):
        self.cluster_predictor.fit(X_train, cluster_labels)

        self.model_repo = {}
        for cluster_i in np.unique(cluster_labels):
            cluster_mask = np.where(cluster_labels == cluster_i)
            cluster_X = X_train[cluster_mask]
            y_class_cluster = y_train[cluster_mask]

            if len(np.unique(y_class_cluster)) == 1:
                model = DummyClassifier(strategy="most_frequent")
            else:
                model = self.classifier(**self.classifier_args)
            model.fit(cluster_X, y_class_cluster)
            self.model_repo[cluster_i] = model

    def predict(self, X_test):
        predictions = []
        for test_sample in X_test:
            cluster_pred = self.cluster_predictor.predict(test_sample.reshape(1, -1))
            if isinstance(cluster_pred, list) or isinstance(cluster_pred, np.ndarray):
                cluster_pred = cluster_pred[0]
            print(self.model_repo)
            cluster_model = self.model_repo[cluster_pred]
            y_pred = cluster_model.predict(test_sample.reshape(1, -1))
            predictions.append(y_pred)
        return predictions

    def name(self):
        return self.__class__.__name__ + f" ({self.classifier.__name__})"

    def classifier_name(self):
        return self.classifier.__name__
