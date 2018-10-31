"""
__name__ : Kumar Shubham
__date__ : 16-09-2018
__desc__ : Code for testing AlexNetModel for Feature comparision
"""


import tensorflow as tf 
import numpy as np 
from AlexNetModel import AlexNetArchitecture
from sklearn import svm
import os 
import pickle
class AlexNetFeatureExt(AlexNetArchitecture):
	def __init__(self,objModel,trainDir,testDir,modelDirToSave):
		## code for running alexnet 
		self.objModel = objModel
		self.trainDir = trainDir
		self.testDir = testDir
		self.modelDirToSave = modelDirToSave
		## initializing the parameter of the alexnet feature extractor 
		super(AlexNetFeatureExt,self).__init__(self.objModel)
		self.classLabel={'airplanes':0, 'chair':1, 'crocodile':2, 'cup':3, 'elephant':4, 'headphone':5, 'helicopter':6, 'lotus':7, 'motorbikes':8, 'watch':9}

	def loadTrain(self):
		## loading the trainingDir
		trainImage = []
		trainLabels = []

		for root,dirs,file in os.walk(self.trainDir):
			for name in file:
				fileName =os.path.join(root,name)
				try:
					data = self.imageFormatter(fileName)
					className = root.split("\\")[-1]
					classLabel = self.classLabel[className]
					trainImage.append(data)
					trainLabels.append(classLabel)
				except Exception as e:
					print(e)
		return trainImage,trainLabels


	def loadTest(self):
		## loading the testingDir
		testImage = []
		testLabels = []
		for root,dirs,file in os.walk(self.testDir):
			for name in file:
				fileName =os.path.join(root,name)
				try:
					data = self.imageFormatter(fileName)
					className = root.split("\\")[-1]
					classLabel = self.classLabel[className]
					testImage.append(data)
					testLabels.append(classLabel)
				except Exception as e:
					print(e)
		return testImage,testLabels


	def sessionBuild(self):
		## building session for running through the images 
		init = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(init)

	def TrainAlexNetFeatureVector(self):
		#### function for training and saving the modelPath

		self.build()## building the network
		self.sessionBuild()## building the session for data processing 

		dataTrain,trainLabels = self.loadTrain()
		listOfTrainDataFeature = []

		for idx,image in enumerate (dataTrain):
			print ("Loading Image for training SVM : %d"%(idx))

			featureVector = self.sess.run(self.fc7, feed_dict = {self.inputLayer:[image]})## fc7 declaration was done in super class of Alexnet
			listOfTrainDataFeature.append(featureVector.ravel())

		trainDataFeature  = np.array(listOfTrainDataFeature)
		trainLabels = np.array(trainLabels)

		# print (trainDataFeature.shape)
		###### training the data ############
		pred = svm.SVC()

		pred.fit(trainDataFeature,trainLabels)
		out = pred.predict(trainDataFeature)
		print ( "Training accuracy : %d"%(np.sum(out==trainLabels)) )

		fileName = open(os.path.join(self.modelDirToSave,"SVMmodelAlexNet.sav"),"wb")

		pickle.dump(pred,fileName)
		print ("saved the SVM Model...")
		fileName.close()

	def TestAlexNetFeatureVector(self):
		self.build()## building the network
		self.sessionBuild()## building the session for data processing 

		listOfTestData = []
		testData, testLabels = self.loadTest()



		for idx,image in enumerate (testData):
			print ("Loading Image for testing SVM : %d"%(idx))

			featureVector = self.sess.run(self.fc7, feed_dict = {self.inputLayer:[image]})## fc7 declaration was done in super class of Alexnet
			listOfTestData.append(featureVector.ravel())


		fileNameSVM = open(os.path.join(self.modelDirToSave,"SVMmodelAlexNet.sav"),"rb")
		svmModel = pickle.load(fileNameSVM)


		testDataFeature  = np.array(listOfTestData)
		testLabels = np.array(testLabels)

		############ testing the data ###########
		out = svmModel.predict(testDataFeature)
		correctPred = np.sum(out==testLabels) 
		print ("correct prediction : {0} total examples : {1} accuracy : {2}%".format(correctPred,len(out), (correctPred/float(len(out))*100 ) ) )		

		fileNameSVM.close()

if __name__ =="__main__":
	obj = AlexNetFeatureExt(objModel="./model/bvlc_alexnet.npy",trainDir="./caltechSampleTrainData/Train",testDir="./caltechSampleTrainData/Test",modelDirToSave="./model")
	
	######### 				step -2 : train SVM model for the input training data 		##########################
	
	# obj.TrainAlexNetFeatureVector()

	#################		step -3 : test SVM model for the input testing  data 		##########################

	obj.TestAlexNetFeatureVector()

