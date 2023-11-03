from sklearn.datasets import make_blobs
from effens.EnsMetaLearning.EffEns import EffEns
from automlclustering.ClusterValidityIndices import CVIHandler
from effens.Utils.Utils import process_result_to_dataframe

X,y = make_blobs()
effens = EffEns(mkr_path="../src/effens/EffEnsMKR/")

cvi = CVIHandler.CVICollection.CALINSKI_HARABASZ
result, _ = effens.apply_ensemble_clustering(X, cvi=cvi, n_loops=5)
result = process_result_to_dataframe(result, {"cvi": cvi.get_abbrev()}, ground_truth_clustering=y)
print(result[["iteration","config", "CVI score", "Best NMI"]])
