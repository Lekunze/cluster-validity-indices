import scipy.spatial.distance as dist
import numpy as np
import warnings


class GeneralizedDunn:

	"""
		GeneralizedDunn is a class for computing different variations of
		the Popular Dunn's Index, with different expressions for:

		(1) Cohesion Estimator — Noted as Big Delta
		(2) Separation Estimator — Noted as Small Delta

	"""

	def __init__(self, data, m):

		if len(m) != len(data):
			warnings.warn('Failed! Dimensions of data and cluster labels are unequal')
			return np.NaN

		self.data = np.array(data)
		self.m = np.array(m)

	def big_delta(self, cluster_k, i, **kwargs):

		if i == 1:
			return max(dist.pdist(cluster_k, **kwargs))

		elif i == 3:

			n_k = len(cluster_k)
			centroid_k = np.mean(cluster_k, axis=0)

			inter_cdist = 0
			for j in cluster_k:
				inter_cdist += dist.pdist([j, centroid_k], **kwargs)

			return (2 * inter_cdist)/n_k

	def small_delta(self, cluster_k, cluster_l, i, **kwargs):

		if i == 3:
			n_k = len(cluster_k)
			n_l = len(cluster_l)

			pw_dist = dist.cdist(cluster_k, cluster_l, **kwargs)

			return np.sum(pw_dist) / (n_k * n_l)

		elif i == 4:

			centroid_k = np.mean(cluster_k, axis=0)
			centroid_l = np.mean(cluster_l, axis=0)

			return dist.pdist([centroid_k, centroid_l], **kwargs)

		elif i == 5:

			n_k = len(cluster_k)
			n_l = len(cluster_l)

			centroid_k = np.mean(cluster_k, axis=0)
			centroid_l = np.mean(cluster_l, axis=0)

			inter_cdist_k = 0
			for k in cluster_k:
				inter_cdist_k += dist.pdist([k, centroid_k], **kwargs)

			inter_cdist_l = 0
			for k in cluster_l:
				inter_cdist_l += dist.pdist([k, centroid_l], **kwargs)

			return(inter_cdist_k + inter_cdist_l) / (n_k + n_l)

	def generalized_exp(self, bd=1, sd=1, **kwargs):

		"""
			Generalized Function for Dunn Index Variants
			Available are: gD33, gD43, gD53

			Parameters
			----------
			bd: int, big delta
			sd: int, small delta
			**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

			Returns
			-------
			score:	float, maximum value represents good partition for all variations
		"""

		intra_cdist = list()
		inter_cdist = list()

		for k in set(self.m):

			# Members of Cluster K
			idx_k = [idx for idx, cx in enumerate(self.m) if cx == k]
			cluster_k = self.data[idx_k, :]

			# Get maximum within-cluster distance
			intra_cdist.append(self.big_delta(cluster_k, bd, **kwargs))

			alt_clusters = list(set(self.m) ^ {k})

			for j in alt_clusters:

				idx_j = [idx for idx, cx in enumerate(self.m) if cx == j]
				cluster_j = self.data[idx_j, :]

				# Get minimum between-cluster distance
				inter_cdist.append(self.small_delta(cluster_k, cluster_j, sd, **kwargs))

		return min(inter_cdist) / max(intra_cdist)


