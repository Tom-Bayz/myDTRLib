import numpy as np
from Decision_tree_regressor import DTR
from matplotlib import pyplot as plt
import copy

class Random_Forest(object):

	def __init__(self,tree_num=100 ,max_deapth=5, random_seed=None, sample_ratio=0.5, max_feature_ratio=1):
		self.tree_num = tree_num
		self.max_deapth = max_deapth
		self.sample_ratio = sample_ratio
		self.Forest = []
		self.max_feature_ratio = max_feature_ratio
		np.random.seed(random_seed)

	def fit(self, trainX, trainY):

		n,d = np.shape(trainX)

		for t in range(self.tree_num):
			sample = np.random.choice(range(n),int(n*self.sample_ratio),replace=True)
			sampleX = trainX[sample,:]
			sampleY = trainY[sample]

			tree = DTR(max_deapth=self.max_deapth,max_feature_ratio=self.max_feature_ratio)
			tree.fit(sampleX,sampleY)

			self.Forest.append(copy.copy(tree))

	def predict(self, testX):

		pred = np.zeros(np.shape(testX)[0])

		for t in range(self.tree_num):
			pred += self.Forest[t].predict(testX)

		return pred/self.tree_num


if __name__ == "__main__":

	mesh = 500
	train_ratio = 0.4

	X = np.linspace(0,10,mesh)
	np.random.shuffle(X)
	X = X[:,np.newaxis]

	Y = X*np.sin(X)/100
	true = np.copy(Y)
	Y = Y[:,0] + np.array(np.random.normal(0, 7*1e-3, mesh))


	trainX = X[:int(mesh*train_ratio),:]
	trainY = Y[:int(mesh*train_ratio)]

	testX = X[int(mesh*train_ratio):,:]
	testY = Y[int(mesh*train_ratio):]

	clf = Random_Forest(tree_num=300)
	clf.fit(trainX,trainY)
	pred = clf.predict(testX)

	#"""
	#yz.set_font()
	plt.grid(True)
	plt.ylim(-0.065,0.1)
	plt.xlabel(r"$x$",fontsize=20)
	plt.ylabel(r"$f(x)$",fontsize=20)
	plt.plot(trainX,trainY,"o",color="black",marker="+",label="data: "r"$\mathcal{D}$")
	plt.plot(np.sort(X[:,0]),true[np.argsort(X[:,0])],"--",color="red",label="Unknown function: "+r"$f$")
	#plt.plot(testX,testY,"o",color="red",marker="+",label="test data")
	plt.plot(testX[np.argsort(testX[:,0]),:],pred[np.argsort(testX[:,0])],"-",color="blue",label="RF prediction")
	plt.legend(fontsize=10)
	plt.tight_layout()
	plt.show()
	#"""
