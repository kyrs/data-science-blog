"""
__name__ : Kumar Shubham
__desc__ : Code for implementing bagOfword using SIFT
"""
import cv2
import numpy as np 
import os 
from sklearn.cluster import KMeans
import pickle 
from collections import Counter
from sklearn import svm

class BagOfWord(object):
	def __init__(self,trainDir,testDir,modelDir):
		## path of image
		self.trainDir = trainDir
		self.testDir = testDir
		self.classLabel={'airplanes':0, 'chair':1, 'crocodile':2, 'cup':3, 'elephant':4, 'headphone':5, 'helicopter':6, 'lotus':7, 'motorbikes':8, 'watch':9}
		self.noCluster = 200
		self.modelDir = modelDir 
		self.sift = cv2.xfeatures2d.SIFT_create()

	def loadTrain(self):
		## loading the trainingDir
		trainImage = []
		trainLabels = []

		for root,dirs,file in os.walk(self.trainDir):
			for name in file:
				fileName =os.path.join(root,name)
				data = cv2.imread(fileName)
				className = root.split("\\")[-1]
				classLabel = self.classLabel[className]
				trainImage.append(data)
				trainLabels.append(classLabel)
		return trainImage,trainLabels


	def loadTest(self):
		## loading the testingDir
		testImage = []
		testLabels = []

		for root,dirs,file in os.walk(self.testDir):
			for name in file:
				fileName =os.path.join(root,name)
				data = cv2.imread(fileName)
				className = root.split("\\")[-1]
				classLabel = self.classLabel[className]
				testImage.append(data)
				testLabels.append(classLabel)
		return testImage,testLabels


	def siftFeatureDesc(self,image):
		## extract the features from the  object image
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		(kps, descs) = self.sift.detectAndCompute(gray, None)
		return kps,descs


	def HistogramTrainingfn(self):
		## used for K mean clustering of the data 
		trainData, _ = self.loadTrain()
		listOfFeatures = []
		for idx, image in enumerate(trainData):
			print ("Extracting feature for K-mean clustering : %d "%(idx))
			_,desc = self.siftFeatureDesc(image)
			listOfFeatures.extend(desc)
		featureSetForTrain = np.array(listOfFeatures)
		print(featureSetForTrain.shape)
		print("starting the training .. \n")
		kmeanOverData = KMeans(n_clusters=self.noCluster,random_state=0,verbose=1,max_iter =100).fit(featureSetForTrain)
		fileName = open(os.path.join(self.modelDir,"knnCluster.sav"),"wb")

		pickle.dump(kmeanOverData,fileName)
		print ("saved the Knn File.")
		fileName.close()

	def trainingClassifier(self):
		## code for training the model
		fileName = open(os.path.join(self.modelDir,"knnCluster.sav"),"rb")
		knnModel = pickle.load(fileName)
		
		listOfTrainData = []
		trainData, trainLabels = self.loadTrain()

		for idx,image in enumerate (trainData):
			print ("Loading Image for training SVM : %d"%(idx))

			featureHist =self._histogramGenerate(knnModel,image)
			listOfTrainData.append(featureHist)

		trainDataFeature  = np.array(listOfTrainData)
		trainLabels = np.array(trainLabels)

		###### training the data ############
		pred = svm.SVC()
		pred.fit(trainDataFeature,trainLabels)
		out = pred.predict(trainDataFeature)
		print ( "Training accuracy : %d"%(np.sum(out==trainLabels)) )

		fileName = open(os.path.join(self.modelDir,"SVMmodel.sav"),"wb")

		pickle.dump(pred,fileName)
		print ("saved the SVM Model...")
		fileName.close()

	def testingClassifier(self):
		## method for testing the classifier accuracy 

		fileNameKNN = open(os.path.join(self.modelDir,"knnCluster.sav"),"rb")
		knnModel = pickle.load(fileNameKNN)
		
		listOfTestData = []
		testData, testLabels = self.loadTest()

		for idx,image in enumerate (testData):
			print ("Loading Image for testing SVM : %d"%(idx))

			featureHist =self._histogramGenerate(knnModel,image)
			listOfTestData.append(featureHist)

		testDataFeature  = np.array(listOfTestData)
		testLabels = np.array(testLabels)

		fileNameSVM = open(os.path.join(self.modelDir,"SVMmodel.sav"),"rb")
		svmModel = pickle.load(fileNameSVM)


		############ testing the data ###########
		out = svmModel.predict(testDataFeature)
		correctPred = np.sum(out==testLabels) 
		print ("correct prediction : {0} total examples : {1} accuracy : {2}%".format(correctPred,len(out), (correctPred/float(len(out))*100 ) ) )

		fileNameSVM.close()
		fileNameKNN.close()

	def _histogramGenerate(self,knnModel,image):
		## genereating the histogram for classification 
		kps,desc =  self.siftFeatureDesc(image)
		nearestCluster = knnModel.predict(desc)
		counterData = Counter(nearestCluster)
		featureHist = np.zeros(self.noCluster)

		## setting the value in the counter data 
		for key in counterData:
			featureHist[key] = counterData[key]
		return featureHist



		pass
if __name__ =="__main__":
	obj = BagOfWord(trainDir="./caltechSampleTrainData/Train",testDir="./caltechSampleTrainData/Test",modelDir="./model")
	######## step -1 : creating the KNN model for Bag Of word (if you already have then don't run it.)############
	
	# obj.HistogramTrainingfn()

	######### 				step -2 : train SVM model for the input training data 		##########################
	
	# obj.trainingClassifier()

	#################		step -3 : test SVM model for the input testing  data 		##########################

	obj.testingClassifier()