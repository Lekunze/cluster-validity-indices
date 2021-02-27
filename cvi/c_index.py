import scipy.spatial.distance as dist
import numpy as np


def c_index(data, m, **kwargs):
	"""
		C-Index is defined as a normalized cohesion estimator, represented by:
			S — S_min / S_max — S_min

		Where,
			S: Sum of distances from point pairs in the same cluster
			S_min: Sum of smallest distances
			S_max: Sum of largest distances

		Parameters
		----------
		m: array of cluster assignments
		data: array of data points
		**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

		Returns
		-------
		score:	Minimum value represents good partition
	"""

	m = np.array(m)
	data = np.array(data)

	s = 0
	nw = 0

	for k in set(m):
		idx_k = [idx for idx, cx in enumerate(m) if cx == k]
		cluster_k = data[idx_k, :]

		pw_distance_k = dist.pdist(cluster_k, **kwargs)
		nw += len(pw_distance_k)
		s += np.sum(pw_distance_k)

	pw_distance = dist.pdist(data, **kwargs)
	s_min = np.sum(sorted(pw_distance)[0:nw])
	s_max = np.sum(sorted(pw_distance, reverse=True)[0:nw])

	return (s-s_min)/(s_max-s_min)