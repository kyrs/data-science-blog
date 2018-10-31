"""
__author__ : Kumar shubham 
__date__   : 13 Sept 2018
__Desc__   : training convnet model
__roll__    : MS2018008
"""

import tensorflow as tf 
import numpy as np 
from cifarData import Cifar10Loader
import os 

import time

class ConvNetStructure(Cifar10Loader):
	def __init__(self,dataDir,batchSize,activation,withMomentum=True,withNormalization=True,withVariableLr=True,skipStep=50,noEpoch =20):
		self.batchSize = batchSize
		self.fileLocation = dataDir
		self.activation = activation
		self.withMomentum = withMomentum
		self.batchNormalizeFlag = withNormalization
		self.noEpoch =noEpoch

		self.skipStep=skipStep
		self.startLearningRate = 0.01
		self.testSample = 10000

		self.gstep = tf.Variable(0, dtype=tf.int32, 
								trainable=False, name='global_step')

		self.variableLearningRate =withVariableLr

		super(ConvNetStructure,self).__init__(self.fileLocation)


	def dataSliceTf(self):
		
		## slicing the data based on 
		trainDataSet, trainDataLabel, trainOneHotEncLabel = self.loadTrainingData()
		testDataSet, testDataLabel, testOneHotEncLabel = self.loadTestingData()

		trainDataTfFormat = tf.data.Dataset.from_tensor_slices((trainDataSet,trainOneHotEncLabel))
		trainDataTfFormat = trainDataTfFormat.shuffle(10000)
		trainDataTfFormat = trainDataTfFormat.batch(self.batchSize)

		testDataTfFormat = tf.data.Dataset.from_tensor_slices((testDataSet,testOneHotEncLabel))
		testDataTfFormat = testDataTfFormat.batch(self.batchSize)

		return trainDataTfFormat,testDataTfFormat



	def getDataForProcessing(self):
		## creating the batch for processing 
		with tf.name_scope("CifarData"):
			trainDataTfFormat,testDataTfFormat = self.dataSliceTf()
			iterator = tf.data.Iterator.from_structure(trainDataTfFormat.output_types,trainDataTfFormat.output_shapes)

			img,self.label = iterator.get_next()

			self.img = tf.reshape(img,shape=[-1,self.imageSize,self.imageSize,self.channel])

			# self.img=tf.cast(img,tf.float32)
			## initializer for the data iteration 
			self.train_init = iterator.make_initializer(trainDataTfFormat)
			self.test_init = iterator.make_initializer(testDataTfFormat)

	def graphStructure(self):
		## defining the graphical structure 

		conv1 = tf.layers.conv2d(inputs=self.img, filters= 32,kernel_size=[5,5],padding="SAME", name="conv1",activation= self.activation) ## first convolution layer
		self.isTraining = tf.placeholder(tf.bool, name='training')
		if self.batchNormalizeFlag:
			conv1Cast = tf.cast(conv1,tf.float32)
			conv1Bn= tf.contrib.layers.batch_norm(conv1Cast, 
                                          center=True, scale=True, 
                                          is_training=self.isTraining,
                                          scope='bn1')

			# conv1Bn = tf.contrib.layers(x=conv1,name= "conv2_bn",mean=mean1,variance=variance1, variance_epsilon=1e-5, offset=0, scale=0.95)
			pool1 = tf.layers.max_pooling2d(inputs=conv1Bn,pool_size=[2,2],strides =2,name = "pool1") ## pooling layer for computation
		else:
			pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides =2,name = "pool1") ## pooling layer for computation

		

		if self.batchNormalizeFlag:
			

			conv2 = tf.layers.conv2d(inputs=pool1, filters= 16,kernel_size=[2,2],padding="SAME", name="conv2",activation= self.activation) ## second convolution layer
			conv2Cast = tf.cast(conv2,tf.float32)
			conv2Bn = tf.contrib.layers.batch_norm(conv2Cast, 
                                          center=True, scale=True, 
                                          is_training=self.isTraining,
                                          scope='bn2')

			# conv2Bn = tf.nn.batch_normalization(x=conv2,name= "conv2_bn",mean=mean2,variance=variance2, variance_epsilon=1e-5, offset=0, scale=0.95)
			finalOutputLayer2 = conv2Bn
		else:
			conv2 = tf.layers.conv2d(inputs=pool1, filters= 16,kernel_size=[2,2],padding="SAME", name="conv2",activation= self.activation) ## second convolution layer
			finalOutputLayer2 = conv2

		pool2 = tf.layers.max_pooling2d(inputs=finalOutputLayer2,pool_size=[2,2],strides =2,name = "pool2") ## pooling layer for computation

		featureDim = pool2.shape[1]*pool2.shape[2]*pool2.shape[3]
		## flatenning the output
		pool2Flatten = tf.reshape(pool2,[-1,featureDim])


		fc1 = tf.layers.dense(pool2Flatten,1024,activation=self.activation, name = "fc1") ## first fully convolution layer over input data 
		fc2 = tf.layers.dense(fc1,512,activation=self.activation, name = "fc2") ## second fully convolution layer over input data 
		self.logits = tf.layers.dense(fc2,self.noClass,name = "logits")


	def lossComputation(self):
		
		
		with tf.name_scope('loss'):
			entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=self.logits)
			self.loss = tf.reduce_mean(entropy,name="loss")

	def optimizer(self):
		## optimizer for rthe given system
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)## necessary for batch normalization

		with tf.control_dependencies(update_ops):
			if self.variableLearningRate:
				learningRate = tf.train.exponential_decay(self.startLearningRate,self.gstep,3000,0.96,staircase=True)
			else:
				learningRate = self.startLearningRate

			if self.withMomentum:
				self.opt =tf.train.MomentumOptimizer(learningRate,0.9).minimize(self.loss,global_step=self.gstep)
			else:
				self.opt =tf.train.GradientDescentOptimizer(learningRate).minimize(self.loss,global_step=self.gstep)


	def summary(self):
		## merging the summary operation 
		with tf.name_scope("summary"):
			tf.summary.scalar('loss',self.loss)
			tf.summary.scalar('accuracy',self.accuracy)
			tf.summary.histogram('histogram loss', self.loss)
			self.summary_op = tf.summary.merge_all()


	def eval(self):
		## evaluating over the input data point 
		with tf.name_scope("predict"):
			preds = tf.nn.softmax(self.logits)
			correctPrediction = tf.equal(tf.argmax(preds,1), tf.argmax(self.label,1))
			self.accuracy = tf.reduce_sum(tf.cast(correctPrediction,tf.float32))


	def build(self):
		## building the full graph Structure 
		self.getDataForProcessing()
		self.graphStructure()
		self.lossComputation()
		self.optimizer()
		self.eval()
		self.summary()
		

	def TrainPerEpoch(self,sess,saver,init,writer,epoch,step):
		## training per epoch
		startTime = time.time()
		sess.run(init)
		totalLoss = 0
		totalBatch = 0
		
		try:
			while(True):
				_,l,summary = sess.run([self.opt,self.loss,self.summary_op],feed_dict={self.isTraining:True})
				writer.add_summary(summary,global_step=step)
				if ((step+1)% self.skipStep ==0):
					print (" epoch : {0} Total step : {1} current Loss : {2} ".format(epoch,step,l))

				totalLoss+=l 
				step+=1
				totalBatch+=1
		except tf.errors.OutOfRangeError:
			pass

		return step 
		# saver.save(sess,"./model/toSaveModel",step)
		print ("Total time taken: ", time.time()-startTime)
		return step
	def evalOnce(self, sess, init, writer, epoch, step):
		startTime = time.time()
		sess.run(init)
		
		correctPreds = 0
		try:
			while True:
				accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op],{self.isTraining:False})
				writer.add_summary(summaries, global_step=step)
				correctPreds += accuracy_batch
		except tf.errors.OutOfRangeError:
			pass

		print('Accuracy at epoch {0}: {1} '.format(epoch, correctPreds/self.testSample))
		print('Took: {0} seconds'.format(time.time() - startTime))

	def Run(self):

		self.build()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()

			ckpt = tf.train.get_checkpoint_state(os.path.dirname('./model/fineTuneModel'))
			writer = tf.summary.FileWriter('./model/summaryWriter', tf.get_default_graph())

			# if ckpt and ckpt.model_checkpoint_path:
			# 	saver.restore(sess, ckpt.model_checkpoint_path)

			step = self.gstep.eval()
			count = 0
			for epoch in range(self.noEpoch):
				start = time.time()
				step = self.TrainPerEpoch(sess, saver, self.train_init, writer, epoch, step)
				end = time.time()
				count += end-start

				self.evalOnce(sess, self.test_init, writer, epoch, step)
		writer.close()

		print ("Total time taken in training process: ", count)



if __name__=="__main__":

	################ Model-1 : ConvNet with momentum and relu activation ########################## 
	obj = ConvNetStructure(dataDir="./data/cifar-10-batches-py",batchSize=16,activation=tf.nn.sigmoid,withMomentum=True,withNormalization=False,withVariableLr=True)
	obj.Run()