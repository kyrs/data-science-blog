"""
__author__ : Kumar shubham
__Desc__   : code for Gibbs sampling experiment for 2 dimensional matrix
__date__   : 30-03-2019
"""
import seaborn as sns 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def biVariantDataGen():
	"""
	in current code we are gonna sample data from a bivariant gaussian with some defined mean and variance
	"""
	dataMean1 = 0
	dataMean2 = 0

	sampleLen = 1000

	dataVar1 = 1
	dataVar2 = 1
	dataVar12 = 0.5

	muVector = np.vstack((dataMean1,dataMean2)) ## mean vector 
	covarMatrix = np.vstack(((dataVar1,dataVar12),(dataVar12,dataVar2))) ## covariance matrix

	## according to sampling technique any gaussian can be sampled with formulation std*N(0,1)+mu

	stdDev = np.linalg.cholesky(covarMatrix) ## formulation for computing std for given covariance matrix

	## no of data point for calculation
	dataPoint = stdDev@np.random.randn(2,sampleLen) + muVector
	
	return muVector,covarMatrix,dataPoint

def conditional12(D, mean, covar,varIndex):
	"""
	D : dimesion of the data 
	mean : mean vector 
	covar : covariance vector
	varIndex : index for which conditioning is done. i.e. for P(a,b,c) if P(c|a,b) the varIndex =2 
	formula : http://www.inf.ed.ac.uk/teaching/courses/mlpr/2017/notes/w7c_gaussian_processes.html
	"""
	assert D==2 # only going to work for 2 Dim data 

	a = mean[varIndex]
	b= mean[~varIndex]

	A = covar[varIndex,varIndex]
	B = covar[~varIndex,~varIndex]
	C = covar[varIndex,~varIndex]

	BInv = 1/B ## as it is a one dim matrix 



	def newDist(evidence):
	## evidence is what is being observed 
		mu = a+ C*BInv*(evidence-b)
		std = np.sqrt(A-C*BInv*C)

		sample  = std*np.random.randn(1)+ mu

		return sample 
	return newDist


def GibbsSampling():

	noSampleGen = 1000
	
	sample = np.zeros((2,noSampleGen))
	D = 2 ## n of sample to generate
	mean,covar,x = biVariantDataGen()
	fnList =[]
	## genereating a list of all conditional probability 
	fnList = [conditional12(D, mean, covar,varIndex)  for varIndex in range(D)]

	# starting with some value
	sample[:,0] = [-2,1]

	for i in range(1,noSampleGen):
		sample[:,i] = sample[:,i-1]

		d = i%D
		sample[d,i] = fnList[d](sample[~d,i-1])

	figure,axes = plt.subplots(1,3)
	
	count = 0
	for i in range(0,noSampleGen):
		axes[0].plot(sample[0][i:i+2],sample[1][i:i+2])
	axes[0].set_title('Gibbs sampling step')

	axes[1].scatter(sample[0],sample[1])
	axes[1].set_title('Gibbs samples')

	axes[2].scatter(x[0],x[1])
	axes[2].set_title('original data')
	
	plt.show()

if __name__ =="__main__":
	GibbsSampling()