import numpy as np

class _Node(object):

	def __init__(self, max_deapth, min_dataNum=2, max_feature_ratio=1):

		self.left = None
		self.right = None
		self.feature = None
		self.threshold = None
		self.target = None
		self.max_deapth = max_deapth
		self.deapth = None
		self.data_num = None
		self.I = -np.inf
		self.min_dataNum = min_dataNum
		self.max_feature_ratio = max_feature_ratio

	def build(self, X, Y, deapth):

		self.target = Y
		self.deapth = deapth
		self.data_num, data_dim = (np.shape(np.atleast_2d(X)))
		self.data_num = float(self.data_num)

		I = self.get_MSE(self.target,"Parent")

		if (self.deapth == self.max_deapth) or (np.shape(np.atleast_1d(np.unique(Y)))[0] <= self.min_dataNum):
			return

		#feature_sample = np.random.choice(range(data_dim),int(data_dim*self.max_feature_ratio))

		for d in range(data_dim):

			dth_dim = np.unique(X[:, d])
			points = (dth_dim[1:] + dth_dim[:-1])/2
			#print(points)

			for threshold in points:

				target_l = Y[X[:, d] <  threshold]
				num_l = np.shape(target_l)[0]

				target_r = Y[X[:, d] >= threshold]
				num_r = np.shape(target_r)[0]

				I_l = self.get_MSE(target_l,"l_child")
				I_r = self.get_MSE(target_r,"r_child")

				I_tmp = I - (num_l/self.data_num)*I_l - (num_r/self.data_num)*I_r
				#I_tmp = (num_l/self.data_num)*I_l + (num_r/self.data_num)*I_r

				#print("	--")
				#print("	I => "+str(I_tmp))
				#print("	(num_l/self.data_num)*I_l => " +str((num_l/self.data_num)*I_l) )
				#print("	(num_r/self.data_num)*I_r => " +str((num_r/self.data_num)*I_r))


				if I_tmp > self.I:
					self.I = I_tmp
					self.feature = d
					self.threshold = threshold

		"""
		print("|## deapth"+str(self.deapth)+" ##")
		print "|  Best threshold: ",
		print([points[0], points[-1]])
		print("|		 => "+str(self.threshold))
		print("|")
		print("|  MSE improvement: "+str(self.I))
		print("|"+"="*65)
		"""

		##### recursive process #####
		# left child
		data_l = X[X[:, self.feature] <  self.threshold]
		target_l = Y[X[:, self.feature] <  self.threshold]
		self.left = _Node(self.max_deapth, self.max_feature_ratio)
		self.left.build(data_l, target_l, self.deapth+1)

		# right child
		data_r = X[X[:, self.feature] >= self.threshold]
		target_r = Y[X[:, self.feature] >= self.threshold]
		self.right = _Node(self.max_deapth, self.max_feature_ratio)
		self.right.build(data_r, target_r, self.deapth+1)

	def get_MSE(self, target,label="None"):

		#print(label+" => "+str(np.mean((target - np.mean(target))**2.0)))
		return np.mean((target - np.mean(target))**2.0)
