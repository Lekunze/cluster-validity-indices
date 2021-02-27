import numpy as np
import scipy.spatial.distance as dist
import scipy.linalg
import math
import warnings


class SDbw:

	def __init__(self, data, m):
		if len(m) != len(data):
			warnings.warn('Failed! Dimensions of data and cluster labels are unequal')
			return np.NaN

		self.data = np.array(data)
		self.m = np.array(m)
		self.K = len(set(m))
		self.stdev = self.ave_cluster_stdev()

	def ave_cluster_stdev(self):

		cluster_sd = 0

		for k in set(self.m):

			# Cluster K
			idx_k = [idx for idx, cx in enumerate(self.m) if cx == k]
			cluster_k = self.data[idx_k, :]

			cluster_k_sd = np.std(cluster_k, axis=0)
			cluster_k_sd_norm = scipy.linalg.norm(cluster_k_sd)

			cluster_sd += cluster_k_sd_norm

		return np.sqrt(cluster_sd) / self.K

	def density(self, cluster_k, cluster_l=None, **kwargs):
		centroid_k = np.mean(cluster_k, axis=0)

		if cluster_l is None:
			inter_cdist = 0
			for i in cluster_k:
				distance_i = dist.pdist([i, centroid_k], **kwargs)
				inter_cdist += SDbw.piecewise_f(distance_i, self.stdev)
			return inter_cdist

		else:
			centroid_l = np.mean(cluster_l, axis=0)
			inter_cdist = 0
			for i in np.concatenate((cluster_k, cluster_l)):
				distance_i = dist.pdist([i, (centroid_k + centroid_l)/2], **kwargs)
				inter_cdist += SDbw.piecewise_f(distance_i, self.stdev)

			return inter_cdist

	@staticmethod
	def piecewise_f(distance, sd):
		if distance > sd:
			return 0
		else:
			return 1

	def scatter(self):

		scatter_sum = 0
		X_sd_norm = scipy.linalg.norm(np.std(self.data, axis=0))

		for k in set(self.m):

			# Cluster K
			idx_k = [idx for idx, cx in enumerate(self.m) if cx == k]
			cluster_k = self.data[idx_k, :]

			cluster_k_sd_norm = scipy.linalg.norm(np.std(cluster_k, axis=0))

			scatter_sum += (cluster_k_sd_norm/X_sd_norm)

		return scatter_sum / self.K

	def density_bw(self, **kwargs):
		"""

		:param kwargs:
		:return:
		"""

		density_ratio = 0
		sd_ratios = list()

		for k in set(self.m):

			# Cluster K
			idx_k = [idx for idx, cx in enumerate(self.m) if cx == k]
			cluster_k = self.data[idx_k, :]

			alt_clusters = list(set(self.m) ^ {k})

			for l in alt_clusters:
				idx_l = [idx for idx, cx in enumerate(self.m) if cx == l]
				cluster_l = self.data[idx_l, :]

				paired_densities = self.density(cluster_k, cluster_l, **kwargs)
				point_density_k = self.density(cluster_k, **kwargs)
				point_density_l = self.density(cluster_l, **kwargs)

				print('Density K', point_density_k)
				print('Density L', point_density_l)
				density_ratio += paired_densities/(max(point_density_k, point_density_l))

			return density_ratio / (math.pow(self.K, 2) - self.K)

	def score(self):
		return self.scatter() + self.density_bw()


if __name__ == '__main__':
	from sklearn.datasets import load_iris, load_wine
	import sklearn.cluster as cluster
	import time

	iris = load_wine()
	clustering = cluster.KMeans().fit(iris.data)
	labels = clustering.labels_

	start = time.time()
	sdbw = SDbw(iris.data, labels)
	end = time.time()
	print('SDBW Index: ', sdbw.score())
	print('Scatter: ', sdbw.scatter())
	print('PET: ', end-start)


