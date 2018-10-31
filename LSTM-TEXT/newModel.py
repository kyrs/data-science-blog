"""
__Author__ : Kumar shubham 
__Date__   : 29/ 10/2018
__desc__   : writing the code for the model creation
__source__ : https://github.com/campdav/text-rnn-tensorflow/blob/master/simple_model.py
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import random
import numpy as np

class Model():

	def __init__(self,dataDir,inputEncoding,logDir,saveDir,rnnSize,numLayers,model,batchSize,seqLength,noEpoch,
				saveAfterEvery,gradClip,learningRate,decayRate,gpuMemory,initFrom,vocabSize,infer=False):

		"""
		dataDir : Dir where data resides
		inputEncoding : encoding format of the input data
		logDir : dir to save the log genereated by the model
		saveDir : dir where checkpoint of the model are saved
		rnnSize : size of the rnn
		numLayers : no of layers in RNN network

		model :

		batchSize: size of a single batch
		seqLength : lenghth of each seq in input data 
		noEpoch  : no of epoch to run
		saveAfterEvery : save checkpoint after every x iteration
		gradClip : clipping the gradient with given factor
		learningRate: learning rate of the given training process
		decay rate : decayRate
		gpuMemory : how much memory being used by the given model
		initFrom : initialize from given network
		vocabSize : size of vocabaluary
		infer : whether infering form given model or not
		##############################################################
		Task - genereating the model for training


		"""

		## if inference 
		if infer:
			batchSize = 1
			seqLength = 1


		### setting up the architecture
		# 'gru': cell_fn = rnn.GRUCell
		# 'basic rnn': cell_fn = rnn.BasicRNNCell
		cellFn = rnn.BasicLSTMCell
		cells = []
		for _ in range(numLayers):
			cell = cellFn(rnnSize)
			cells.append(cell)

		
		self.cell = cell = rnn.MultiRNNCell(cells) ## new network which is being created 



		## input data model creation 

		self.inputData = tf.placeholder(tf.int32,[batchSize,seqLength],name="inputData")
		self.targets = tf.placeholder(tf.int32,[batchSize,seqLength],name = "targets")

		## defining the initial state of the integer over the size of the batch
		self.initialState = cell.zero_state(batchSize, tf.float32)

		## batch pointer handler for the overall training
		self.batchPointer = tf.Variable(0,name="batchPointer",trainable=False,dtype =tf.int32)
		self.incBatchPointerOp = tf.assign(self.batchPointer,self.batchPointer+1)

		self.epochPointer = tf.Variable(0,name="epochPointer",trainable=False)
		self.batchTime = tf.Variable(0.0,name="batchTime",trainable=False)

		tf.summary.scalar("time_batch",self.batchTime)

		def variableSummaries(var):
			mean = tf.reduce_mean(var)
			#create scalar values:
			tf.summary.scalar('mean', mean)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))

		##Gets an existing variable with these parameters or create a new one.
		with tf.variable_scope("rnnlm"):
			softmaxWeight = tf.get_variable("softmax_w",[rnnSize,vocabSize])
			variableSummaries(softmaxWeight)
			softmaxBias = tf.get_variable("softmax_b",[vocabSize])
			variableSummaries(softmaxBias)


			"""
			embedding_lookup function retrieves rows of the params tensor. 
			The behavior is similar to using indexing with arrays in numpy. E.g.

			matrix = np.random.random([1024, 64])  # 64-dimensional embeddings
			ids = np.array([0, 5, 17, 33])
			print matrix[ids]  # prints a matrix of shape [4, 64] 
			params argument can be also a list of tensors in which case the ids will be distributed among the tensors.
			For example, given a list of 3 tensors [2, 64], 
			the default behavior is that they will represent ids: [0, 3], [1, 4], [2, 5].

			partition_strategy controls the way how the ids are distributed among the list.
			The partitioning is useful for larger scale problems when the matrix might be too large to keep in one piece.
			"""

			## tf.squeeze : Removes dimensions of size 1 from the shape of a tensor. (deprecated arguments)
			##tf.split    :  Splits a tensor into sub tensors.
			"""
			# 'value' is a tensor with shape [5, 30]
				# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
				split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
				tf.shape(split0)  # [5, 4]
				tf.shape(split1)  # [5, 15]
				tf.shape(split2)  # [5, 11]
				# Split 'value' into 3 tensors along dimension 1
				split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
				tf.shape(split0)  # [5, 10]
			"""

		with tf.device("/cpu:0"):
			## check this for detail about embedding  https://web.stanford.edu/class/cs20si/2017/lectures/notes_04.pdf
			embedding = tf.get_variable("embedding",[vocabSize,rnnSize])
			inputs = tf.split(tf.nn.embedding_lookup(embedding, self.inputData), seqLength, 1)
			inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
		

		## defining the loop function for processing 

		####################################################
			"""
			tf.stop_gradient
			Defined in generated file: tensorflow/python/ops/gen_array_ops.py.

			Stops gradient computation.

			When executed in a graph, this op outputs its input tensor as-is.

			When building ops to compute gradients, this op prevents the contribution of its
			 inputs to be taken into account. Normally, the gradient generator adds ops to a graph to 
			 compute the derivatives of a specified 'loss' by recursively finding out inputs that contributed 
			 to its computation. If you insert this op in the graph it inputs are masked from the gradient generator. 
			 They are not taken into account for computing gradients.

			This is useful any time you want to compute a value with TensorFlow but need to pretend that the value
			 was a constant. Some examples include:

			The EM algorithm where the M-step should not involve backpropagation through the output of the E-step.
			Contrastive divergence training of Boltzmann machines where, when differentiating the energy function,
			 the training must not backpropagate through the graph that generated the samples from the model.
			Adversarial training, where no backprop should happen through the adversarial example generation process.

			"""


		def loop(prev,_):
			prev = tf.matmul(prev, softmaxWeight)+softmaxBias
			prevSymbol = tf.stop_gradient(tf.argmax(prev,1))
			return tf.nn.embedding_lookup(embedding, prevSymbol)

		####################################################


		## RNN Decoder initilization
		#---------------------------
		#RNN decoder for the sequence-to-sequence model. It requires:
			# - inputs,
			# - initial_state,
			# - cell function and size,
			# - loop_function,
			# - scope scope for the created subgraph
		#this function returns:
			# - outputs: (the generated outputs, a list of the same length as inputs)
			# - last_state : the state of each cell at the final time-step.

		outputs, lastState = legacy_seq2seq.rnn_decoder(inputs,self.initialState,cell,loop_function=loop,scope = 'rnnlm')

		output = tf.reshape(tf.concat(outputs, 1), [-1, rnnSize])

		self.logits = tf.matmul(output,softmaxWeight)+softmaxBias

		self.probs = tf.nn.softmax(self.logits)

		#We want to minimize the average negative log probability of the target words:
		#first, define the loss:
		loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
			[tf.reshape(self.targets, [-1])],
			[tf.ones([batchSize * seqLength])],
			vocabSize)

		self.cost = tf.reduce_sum(loss) / batchSize / seqLength

		tf.summary.scalar("cost",self.cost)

		self.finalState = lastState

		## defining the learning rate of the exp 

		self.lr = tf.Variable(0.0,trainable=False)

		## defining the gradinet clip

		tfvar = tf.trainable_variables()

		grads,_ = tf.clip_by_global_norm(tf.gradients(self.cost, tfvar),
			gradClip)

		optimizer = tf.train.AdamOptimizer(self.lr)


			#Apply gradients to variables.
		#-----------------------------
		self.trainOp = optimizer.apply_gradients(zip(grads, tfvar))

	def sample(self,sess,words,vocab,num,prime='first',samplingType=1):
		'''
		This function is used to generate text, based on a saved model, with
		a text as input.
		It returns a string, composed of words chosen one by one by the model.
		'''
		## https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.searchsorted.html

		## https://stackoverflow.com/questions/35875652/lstm-inputs-for-tensorflow
		def weightPick(weights): ## use to pick from a distribution randomly 
			t = np.cumsum(weights)
			s = np.sum(weights)
			return(int(np.searchsorted(t, np.random.rand(1)*s)))


		ret = ''
		## defining the state for the inference 
		state = sess.run(self.cell.zero_state(1, tf.float32))
		
		ret = prime

		word = prime.split()[-1] ## picking the last word of the waord sentence

		## looping throught he words to be genereated

		for n in range(num):
			x = np.zeros((1,1))

			x[0,0] = vocab.get(word,0)

			## feed the data 

			feed = {self.inputData:x,self.initialState:state}

			[probs,state] = sess.run([self.probs,self.finalState],feed)

			p = probs[0]

			## different sampling type for the formulation
			if samplingType ==0:
				sample = np.argmax(p)
			elif samplingType ==2 :
				if word == "\n":
					sample = weightPick(p)
				else:
					sample = np.argmax(p)
			else:
				sample = weightPick(p)

			pred = words[sample]
			ret += ' '+pred

			word = pred
		return ret 




