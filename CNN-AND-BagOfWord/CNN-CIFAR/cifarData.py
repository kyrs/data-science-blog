"""
__author__ : Kumar shubham 
__date__   : 13 Sept 2018
__Desc__   : loading the CIFAR data in memory
__roll__    : MS2018008
"""

import tensorflow as tf 
import numpy as np 
import pickle
import  matplotlib.pyplot  as plt
import os 

class Cifar10Loader(object):
	def __init__(self,fileLocation):
		self.imageSize = 32
		self.fileLocation = fileLocation
		self.noClass = 10
		self.numFile = 1
		self.imagePerFile = 10000
		self.channel = 3
		self.noTrinImage = self.imagePerFile*self.numFile		


	def __rawImageConverter(self,image):
		## converting an image into raw pixel array
		# format NHWC [Batch, Height,Width, channel]
		floatImage = np.array(image,dtype=float)/255.0
		imageReshaped = floatImage.reshape(-1,self.channel,self.imageSize,self.imageSize)
		finalImage = imageReshaped.transpose(0,2,3,1)
		return finalImage

	def __unPickledDataLoader(self,fileAdd=""):
		## providing the address of folder file.
		with open(fileAdd,"rb") as fileNameReader:
			print ("Loading the File : %s"%(fileAdd))
			dictToLoad = pickle.load(fileNameReader,encoding='bytes')
			trainingData = self.__rawImageConverter(dictToLoad[b"data"])
			trainingLabels = np.array(dictToLoad[b"labels"])	
			return trainingData,trainingLabels


	def loadTrainingData(self):
		## function to load the training data.
		trainingData = np.zeros([self.imagePerFile*self.numFile,self.imageSize,self.imageSize,self.channel],dtype=float)
		trainingLabel = np.zeros([self.imagePerFile*self.numFile],dtype=float)

		startIdx = 0
		for i in range(self.numFile):
			newFileLocation = self.fileLocation+"/data_batch_"+str(i+1)
			dataList,lablelList = self.__unPickledDataLoader(fileAdd = newFileLocation)
			
			## len represent the first dimension in numpy calculation
			sizeOfList = len(dataList)
			trainingData[startIdx:startIdx+sizeOfList, :] = dataList
			trainingLabel[startIdx:startIdx+sizeOfList] = lablelList 
			startIdx = startIdx+sizeOfList


		return trainingData,trainingLabel,self.oneHotEncoding(trainingLabel)

	def loadTestingData(self):
		## load the test data within the system
		testingData = np.zeros([self.imagePerFile,self.imageSize,self.imageSize,self.channel],dtype=float)
		testingLabel = np.zeros([self.imagePerFile],dtype=float)
		newFileLocation = self.fileLocation+"/test_batch"
		dataList,lablelList = self.__unPickledDataLoader(fileAdd=newFileLocation)
		testingData[0:self.imagePerFile,:] = dataList 
		testingLabel[0:self.imagePerFile] = lablelList

		return testingData,testingLabel, self.oneHotEncoding(testingLabel)



	def oneHotEncoding(self,labelList):
		## label list for the given dataset
		listIdx = [ int(i) for  i in labelList ]
		out =np.eye(self.noClass,dtype=float)[listIdx]  
		return out


if __name__ == "__main__":
	obj = Cifar10Loader("./data/cifar-10-batches-py")
	obj.loadTrainingData()