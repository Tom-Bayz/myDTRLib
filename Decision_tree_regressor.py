import numpy as np
from TreeBuilder import _Node
from matplotlib import pyplot as plt

class DTR(object):

	def __init__(self,criterion="mse",max_deapth=5, max_feature_ratio=1):
		self.criterion = criterion
		self.max_deapth = max_deapth
		self.root = None
		self.max_feature_ratio = max_feature_ratio

	def fit(self, trainX, trainY):

		self.root = _Node(self.max_deapth,self.max_feature_ratio)
		self.root.build(trainX, trainY, 0)

	def predict(self, testX):

		pred = []
		test_n, _ = np.shape(testX)

		for n in range(test_n):
			now_node = self.root

			while (now_node.left is not None) & (now_node.right is not None):

				if testX[n,now_node.feature] < now_node.threshold:
					now_node = now_node.left
					#print("left")

				elif testX[n,now_node.feature] >= now_node.threshold:
					now_node = now_node.right
					#print("right")

				else:
					print("Value Error")
					return

			#print(np.mean(now_node.target))
			pred.append(np.mean(now_node.target))

		return np.array(pred)

	def plot_threshold(self, node=None):

		if node is None:
			node = self.root

		plt.plot([node.threshold,node.threshold],[-50,50],"-",color="red",alpha=0.2)

		if node.left is not None:
			self.plot_threshold(node.left)

		if node.right is not None:
			self.plot_threshold(node.right)


if __name__ == "__main__":

	mesh = 100
	train_ratio = 0.7

	X = np.linspace(0,10,mesh)
	np.random.shuffle(X)
	X = X[:,np.newaxis]

	Y = X*np.sin(X)/100
	Y = Y[:,0] + np.array(np.random.normal(0, 7*1e-3, mesh))

	#print(X)
	#print(Y)

	trainX = X[:int(mesh*train_ratio),:]
	trainY = Y[:int(mesh*train_ratio)]

	testX = X[int(mesh*train_ratio):,:]
	testY = Y[int(mesh*train_ratio):]

	"""
	plt.grid(True)
	plt.plot(trainX,trainY,"o",color="black",label="train")
	plt.plot(testX,testY,"o",color="red",marker="+",label="test")
	plt.show()
	"""

	clf = DTR(max_deapth=6)
	clf.fit(trainX,trainY)
	pred = clf.predict(testX)

	#"""
	#plt.grid(True)
	plt.ylim(-0.065,0.1)
	clf.plot_threshold()
	plt.plot(trainX,trainY,"o",color="black",marker="+",label="train")
	plt.plot(testX,testY,"o",color="red",marker="+",label="test")
	plt.plot(testX,pred,"o",color="blue",marker="*",label="pred")
	plt.legend()
	plt.show()
	#"""
