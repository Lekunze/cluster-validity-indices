import unittest

import sklearn.cluster as cluster
from sklearn.datasets import load_iris

import cvi.dunn_index


class TestDunnIndex(unittest.TestCase):

	def test_dunn_index(self):
		iris = load_iris()
		clustering = cluster.Birch().fit(iris.data)
		labels = clustering.labels_
		self.assertAlmostEqual(cvi.dunn_index.dunn_index(iris.data, labels), 0.08714893406611904)


if __name__ == '__main__':
	unittest.main()
