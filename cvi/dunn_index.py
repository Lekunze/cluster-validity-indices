import scipy.spatial
import numpy as np
import warnings


def dunn_index(data, m, **kwargs):
	"""
		Dunn's index is an internal cluster validity measure
		It compares the ratio of MIN. inter-cluster distance to MAX. intra-cluster distanc

		Parameters
		----------
		m: array of cluster assignments
		data: array of data points
		**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

		Returns
		-------
		score:	Maximum value represents good partition
	"""

	m = np.array(m)
	data = np.array(data)

	if len(m) != len(data):
		warnings.warn('Failed! Dimensions of data and cluster labels are unequal')
		return np.NaN

	intra_cdist = list()
	inter_cdist = list()

	for i in set(m):

		idx_i = [idx for idx, cx in enumerate(m) if cx == i]
		cluster_i = data[idx_i, :]

		# Get maximum within-cluster distance
		intra_cdist.append(max(scipy.spatial.distance.pdist(cluster_i, **kwargs)))

		# Get minimum between-cluster distance
		for j in list(set(m) ^ {i}):

			idx_j = [idx for idx, cx in enumerate(m) if cx == j]
			cluster_j = data[idx_j, :]

			pw_dist = scipy.spatial.distance.cdist(np.array(cluster_i), np.array(cluster_j))
			inter_cdist.append(min(pw_dist.flatten()))

	return min(inter_cdist) / max(intra_cdist)

