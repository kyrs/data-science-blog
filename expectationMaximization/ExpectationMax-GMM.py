"""
__author__ :kumar shubham
__desc__   : expectation maximization over single dim gaussian 

ref : http://www.math.uwaterloo.ca/~hwolkowi//matrixcookbook.pdf
https://zhiyzuo.github.io/EM/

"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal as Gauss
plt.style.use('ggplot')


np.random.seed(100)

def dataGen():

	N = 100
	## following function genereate dataset to test the 
	mu = np.vstack((4,9))
	std = np.vstack((2.3,5))
	
	data1 = std[0]*np.random.randn(N,1)+mu[0]
	data2 = std[1]*np.random.randn(N,1)+mu[1]
	fullData = np.append(data1,data2)

	
	return np.array(fullData).reshape(-1,1)


class GMM(object):
	def __init__(self,x,z):

		## setting up the data and z values 
		self.z = z
		self.data = x.copy()
		m,n = x.shape

		self.mean = np.random.random((z,n))
		self.covar = np.array([np.asmatrix(np.identity(n)) for _ in range(self.z)])
		self.w = (np.ones(self.z)/self.z)
		self.dist = np.vstack([np.empty(self.z)/self.z for _ in range(m)])
		
	def eStep(self):
		## doing the e step of the expectation maximization
		m,n = self.data.shape
		for i in range(m):
			normFact = 0
			for zIdx in range(self.z):
				val = self.w[zIdx] * Gauss(mean=self.mean[zIdx,:].T,cov=self.covar[zIdx,:],allow_singular=True).pdf(self.data[i,:])
				self.dist[i,zIdx] =val
				normFact+=val
			self.dist[i,:]/=normFact

			assert np.sum(self.dist[i,:])-1 < 1e-2


	def mStep (self):
		## doing the z step of the expectation maximization

		m,n = self.data.shape

		

		
		# ## updating the variance term 
		# for k in range(self.z):
		# 	variance = np.zeros((n,n))
		# 	for idx in range(m):
		# 		imdVal = (self.data[idx,:]-self.mean[k,:])
		# 		symMat = imdVal*imdVal

		# 		variance+=self.dist[idx,k]*symMat
		# 	self.covar[k]=variance/np.sum(self.dist[:,k])

		# 	print (self.covar[k])
		

		for k in range(self.z):
			self.covar[k] = np.cov(self.data.T, aweights=self.dist[:,k])

		## weights update 
		## updating the weights 
		for i in range(self.z):
			self.w[i] = np.sum(self.dist[:,i])/m
		
		## updating the mean term 
		for k in range(self.z):
			val =np.zeros(n)

			for idx in range(m):
				val+=self.dist[idx,k]*self.data[idx,:]
			self.mean[k]=val/np.sum(self.dist[:,k])
			


	def ELBO(self):
		## calculating the elbo of the given distribution 
		m,n = self.data.shape
		elbo = 0
		for i in range(m):
			
			for zIdx in range(self.z):
				val = self.w[zIdx] * Gauss(mean=self.mean[zIdx,:].T,cov=self.covar[zIdx,:],allow_singular=True).pdf(self.data[i,:])
				elbo+= np.log(val/self.dist[i,zIdx])*self.dist[i,zIdx]
		return elbo

	def fit(self):
		## fitting the model on given dataset
		self.eStep()
		self.mStep()

	
	def Run(self):
		## running the model
		prevELBO = 0
		currELBO = 1
		count = 0
		while(currELBO-prevELBO > 1e-8):

			self.fit()
			prevELBO = self.ELBO()
			self.fit()
			currELBO = self.ELBO()

			print("count : %d elbo:  %f diff : %f"%(count,currELBO,(currELBO-prevELBO)))

			count+=1



	def draw(self):
		pass

if __name__=="__main__":
	
	X = dataGen()
	obj = GMM(x=X,z=2)
	N = 100
	obj.Run()
	resultClass1 = []
	resultClass2 = []
	for x,prob in zip(X,obj.dist):
		if prob[0]>prob[1]:
			resultClass1.append(x)
		else:
			resultClass2.append(x)

	figure,ax = plt.subplots(1,2)
	ax[0].scatter(X[:N],np.zeros(N),c="blue")
	ax[0].scatter(X[N:],np.zeros(N),c="orange")
	ax[0].set_title('original data')
	sns.distplot(X[:N],color="blue",ax=ax[0])
	sns.distplot(X[N:],color="orange",ax=ax[0])


	ax[1].scatter(resultClass1,np.zeros(len(resultClass1)),c="blue")
	ax[1].scatter(resultClass2,np.zeros(len(resultClass2)),c="orange")
	ax[1].set_title('GMM-output')
	sns.distplot(resultClass1,color="blue",ax=ax[1])
	sns.distplot(resultClass2,color="orange",ax=ax[1])

	plt.show()
