import scipy.spatial.distance as dist
import numpy as np


def cop(data, m, **kwargs):
	"""
		COP Index is a cohesion-separation ratio, where:
			Cohesion — Distance from points to cluster centroid
			Separation — Distance from furthest neighbor

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

	N = len(data)
	cop_k = 0

	for k in set(m):
		idx_k = [idx for idx, cx in enumerate(m) if cx == k]
		cluster_k = data[idx_k, :]

		# Centroid for cluster k
		centroid_k = np.mean(cluster_k, axis=0)

		# Size of cluster k
		n_k = len(cluster_k)

		# Intra-Cluster distance from centroid
		intra_cdist = 0

		for instance in cluster_k:
			intra_cdist += dist.pdist([instance, centroid_k], **kwargs)

		alt_clusters = list(set(m) ^ {k})

		intra_cop = (1/n_k) * intra_cdist
		inter_cop = min(separation(cluster_k, data, alt_clusters, **kwargs))

		cop_k += (n_k * (intra_cop/inter_cop))

	return cop_k/N


def separation(cluster_i, data, m, **kwargs):
	"""
		Compute maximum separation or farthest distance between points in clusters

		Parameters
		----------
		cluster_i: reference cluster, from which other distances are computed
		m: array of cluster assignments
		data: array of data points
		**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

		Returns
		-------
		list:	Maximum point-distances from cluster comparisons
	"""

	# Inter-cluster distance from points in other cluster
	inter_cdist = list()

	for j in m:
		idx_j = [idx for idx, cx in enumerate(m) if cx == j]
		cluster_j = data[idx_j, :]

		inter_cdist.append(max(dist.cdist(cluster_i, cluster_j, **kwargs)))

	return inter_cdist
