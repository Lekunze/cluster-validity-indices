import numpy as np
import scipy.spatial.distance as dist
from sklearn.neighbors import KDTree
import warnings


class gSym:
	"""
		Implements a new class of cluster metrics based on point symmetry distance, not Euclidean
		Each individual metric mostly modifies an existing CVI, eg. Dunn Index, Davies Bouldin,
		resulting in indices such as Sym-DB, Sym-Dunn Index, etc,
	"""

	def __init__(self, data, m):
		if len(m) != len(data):
			warnings.warn('Failed! Dimensions of data and cluster labels are unequal')
			return np.NaN

		self.data = np.array(data)
		self.m = np.array(m)
		self.K = len(set(m))

	@staticmethod
	def sym_point(point, centroid):
		"""
			Reflected / Symmetrical point with respect to centroid

			Parameters
			---------
			point: numpy array
			centroid: numpy array

			Returns
			-------
			symmetrical point: numpy array of reflected point
		"""
		return 2 * centroid - point

	@staticmethod
	def ps_distance(point, centroid, cluster, n_neighbors=2, **kwargs):
		"""
			Computes point symmetrical distance of point with respect to the centroid

			Parameters
			---------
			point: numpy array
			centroid: numpy array
			cluster: numpy array of cluster point belongs to
			n_neighbors: int, number of neighbors in KD-Tree
			**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

			Returns
			-------
			symmetrical point: numpy array of reflected point
		"""

		# Get x* — symmetrical point w.r.t centroid
		x_sym = gSym.sym_point(point, centroid)

		# KD-Tree Nearest Neighbor search
		tree = KDTree(cluster, leaf_size=2)
		distance, ind = tree.query(np.array([x_sym]), k=n_neighbors)
		d_sym = np.sum(distance) / n_neighbors

		# Intra-cluster distance from centroid
		intra_cdist = dist.pdist([point, centroid], **kwargs)

		return d_sym * intra_cdist

	def Sym(self, **kwargs):
		"""
			Sym-Index: Symmetrical variation of I-Index
			Computed as ratio of maximum inter-centroid distances,
			divided by sum of point symmetry distances multiplied by number of clusters

			Parameters
			---------
			**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

			Returns
			-------
			score:	float, maximum value represents good partition

		"""

		inter_cdist = list()
		ps_distance = 0

		for k in set(self.m):
			# Members of Cluster K
			idx_k = [idx for idx, cx in enumerate(self.m) if cx == k]
			cluster_k = self.data[idx_k, :]
			centroid_k = np.mean(cluster_k, axis=0)

			# Sums point symmetry distance for all points in cluster, for all clusters
			for instance in cluster_k:
				ps_distance += gSym.ps_distance(instance, centroid_k, cluster_k, **kwargs)

			# Get other clusters
			alt_clusters = list(set(self.m) ^ {k})

			# Get euclidean / regular distance measure between cluster centroids
			for l in alt_clusters:
				idx_l = [idx for idx, cx in enumerate(self.m) if cx == l]
				cluster_l = self.data[idx_l, :]
				centroid_l = np.mean(cluster_l, axis=0)

				inter_cdist.append(dist.pdist([centroid_k, centroid_l], **kwargs))

		return (max(inter_cdist) / (self.K * ps_distance))[0]

	def SymDB(self, **kwargs):
		"""
			Sym-DB: Symmetry-Based Davies-Bouldin Index
			Computed as DB Index, with Scatter modified to be average sum of all
			point-symmetry distances within clusters

			Parameters
			---------
			**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

			Returns
			-------
			score:	float, minimum value represents good partition

		"""

		R = list()

		for k in set(self.m):
			# Members of Cluster K
			idx_k = [idx for idx, cx in enumerate(self.m) if cx == k]
			cluster_k = self.data[idx_k, :]
			centroid_k = np.mean(cluster_k, axis=0)

			# Sums point symmetry distance for all points in Cluster K
			ps_distance_k = 0
			for instance in cluster_k:
				ps_distance_k += gSym.ps_distance(instance, centroid_k, cluster_k, **kwargs)

			ps_distance_k = ps_distance_k/len(cluster_k)

			# Get other clusters
			alt_clusters = list(set(self.m) ^ {k})

			# Get euclidean / regular distance measure between cluster centroids
			# Get point symmetry distance for all points in Alt Clusters
			for l in alt_clusters:
				idx_l = [idx for idx, cx in enumerate(self.m) if cx == l]
				cluster_l = self.data[idx_l, :]
				centroid_l = np.mean(cluster_l, axis=0)

				inter_cdist = dist.pdist([centroid_k, centroid_l], **kwargs)

				ps_distance_l = 0
				for l in cluster_l:
					ps_distance_l += gSym.ps_distance(l, centroid_l, cluster_l, **kwargs)

				ps_distance_l = ps_distance_l/len(cluster_l)

				R.append(max((ps_distance_k + ps_distance_l)/inter_cdist))

		return np.sum(R)/self.K

	def Sym33(self, **kwargs):

		"""
			Sym-Index: Symmetrical variation of gD33 (Varied Dunn's Index)
			Cohesion estimator modified as sum of point symmetry distances for cluster,
			divided by Cluster size and multiplied by 2


			Parameters
			---------
			**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

			Returns
			-------
			score:	float, maximum value represents good partition

		"""

		intra_cdist = list()
		inter_cdist = list()

		for k in set(self.m):

			# Cluster K
			idx_k = [idx for idx, cx in enumerate(self.m) if cx == k]
			cluster_k = self.data[idx_k, :]
			centroid_k = np.mean(cluster_k, axis=0)

			# Get maximum within-cluster distance, computed as PS-distance
			ps_distance_k = 0
			for instance in cluster_k:
				ps_distance_k += gSym.ps_distance(instance, centroid_k, cluster_k, **kwargs)
			intra_cdist.append((2 * ps_distance_k) / len(cluster_k))

			# Get other clusters
			alt_clusters = list(set(self.m) ^ {k})

			for j in alt_clusters:
				idx_j = [idx for idx, cx in enumerate(self.m) if cx == j]
				cluster_j = self.data[idx_j, :]

				# Get minimum between-cluster distance
				inter_cdist.append(gSym.small_delta(cluster_k, cluster_j, **kwargs))

		return (min(inter_cdist) / max(intra_cdist))[0]

	@staticmethod
	def small_delta(cluster_k, cluster_l, **kwargs):
		"""
			Small delta computation of separation estimator for gD33 index

			Parameters
			---------
			cluster_k: numpy array of first cluster
			cluster_l: numpy array of second cluster
			**kwargs: See scipy.spatial.distance **kwargs; pass necessary added parameters, eg. V- value in seuclidean

			Returns
			-------
			score:	between-cluster distance between cluster_k and cluster_l
		"""
		n_k = len(cluster_k)
		n_l = len(cluster_l)

		pw_dist = dist.cdist(cluster_k, cluster_l, **kwargs)

		return np.sum(pw_dist) / (n_k * n_l)


if __name__ == '__main__':
	from sklearn.datasets import load_iris, load_wine
	import sklearn.cluster as cluster
	import time

	iris = load_iris()
	clustering = cluster.KMeans().fit(iris.data)
	labels = clustering.labels_


	# start = time.time()
	gSym = gSym(iris.data, labels)
	# print('Sym Point: ', gSym.sym_point(a, c))
	print('SymDB: ', gSym.SymDB())
	print('Sym33: ', gSym.Sym33())
	print('Sym: ', gSym.Sym())

	# end = time.time()
	# print('SDBW Index: ', sdbw.score())
	# print('Scatter: ', sdbw.scatter())
	# print('PET: ', end - start)
