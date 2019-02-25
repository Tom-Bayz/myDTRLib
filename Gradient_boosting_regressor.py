import numpy as np
from Decision_tree_regressor import DTR
from matplotlib import pyplot as plt
import copy

class Gradient_Boosting(object):

	def __init__(self,tree_num=100 ,max_deapth=5, random_seed=None, max_feature_ratio=1):
		self.tree_num = tree_num
		self.max_deapth = max_deapth
		self.Forest = []
		self.max_feature_ratio = max_feature_ratio
		np.random.seed(random_seed)

	def fit(self, trainX, trainY):

		n,d = np.shape(trainX)

		tree = DTR(max_deapth=self.max_deapth,max_feature_ratio=self.max_feature_ratio)
		tree.fit(trainX,trainY)
		pred = tree.predict(trainX)

		self.Forest.append(copy.copy(tree))

		gradient = trainY - pred

		for t in range(self.tree_num-1):

			tree = DTR(max_deapth=self.max_deapth,max_feature_ratio=self.max_feature_ratio)
			tree.fit(trainX,gradient)
			pred += tree.predict(trainX)

			self.Forest.append(copy.copy(tree))

			gradient = trainY - pred

	def predict(self, testX):

		pred = np.zeros(np.shape(testX)[0])

		for t in range(self.tree_num):
			pred += self.Forest[t].predict(testX)

		return pred


if __name__ == "__main__":

	mesh = 500
	train_ratio = 0.3

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

	for i in range(1,50):
		regressor = Gradient_Boosting(tree_num=i)
		regressor.fit(trainX,trainY)
		pred = regressor.predict(testX)

		#"""
		plt.grid(True)
		plt.title("# of tree:"+str(i))
		plt.ylim(-0.065,0.1)
		#plt.plot(trainX,trainY,"o",color="black",marker="+",label="train")
		#plt.plot(X,true,"o",color="black",label="true function")
		plt.plot(testX,testY,"o",color="red",marker="+",label="test")
		plt.plot(testX[np.argsort(testX[:,0]),:],pred[np.argsort(testX[:,0])],"-",color="blue",marker="*",label="pred")
		plt.legend()
		plt.pause(0.001)
		plt.close()
		#"""
