import tensorflow as tf 
import numpy as np 
from model import helperFn


FLAGS = tf.flags.FLAGS 


def getEmbeddings(hparams):
	if hparams.glove_path or hparams.vocab_path:
		tf.logging.info("Loading the Glove embedding..")
		vocabArray,vocabDict = helperFn.loadVocab(hparams.vocab_path)
		gloveVectors,gloveDict = helperFn.loadGloveVectors(hparams.glove_path,vocab = set(vocabArray))
		initializer = helperFn.buildInitialEmbeddingMatrix(vocabDict, gloveDict, gloveVectors, hparams.embedding_dim)
	else:
		tf.logging.info("No glove/ vocab file found")
		initializer = tf.random_uniform_initializer(-0.25,0.25)
	return tf.get_variable(
		"word_embeddings",
		shape=[hparams.vocab_size, hparams.embedding_dim],
		initializer=initializer)


def dualEncoderModel(hparams,mode,context,contextLen,utterance,utteranceLen,targets):
	## making the model for dual encoder 

	## intialize the embedding randomly or with the pre trained model 
	embeddingW = getEmbeddings(hparams)

	## converting the embedding into the vector format 
	contextEmbedded = tf.nn.embedding_lookup(embeddingW,context,name = "embeded_context")
	uttaranceEmbedded = tf.nn.embedding_lookup(embeddingW,utterance,name = "embeded_utterance")

	with tf.variable_scope("rnn") as vs:
		cell = tf.contrib.rnn.LSTMCell(
	        hparams.rnn_dim,
	        forget_bias=2.0,
	        use_peepholes=True,
	        state_is_tuple=True)

	"""
		tf.nn.dynamic_rnn(
    cell,
    inputs,
    sequence_length=None,
    initial_state=None,
    dtype=None,
    parallel_iterations=None,
    swap_memory=False,
    time_major=False,
    scope=None
)
	"""
	rnnOutput,rnnState = tf.nn.dynamic_rnn(
						cell,
						tf.concat([contextEmbedded,uttaranceEmbedded],0),
						sequence_length =tf.concat([contextLen,utteranceLen],0),
						dtype = tf.float32 
						)
	encodedContext,encodedUtternace = tf.split(rnnState.h,2,0)

	with tf.variable_scope("prediction") as vs:
		M = tf.get_variable("M",shape = [hparams.rnn_dim,hparams.rnn_dim],initializer=tf.truncated_normal_initializer())


	genereatedResponse = tf.matmul(encodedContext,M)
	genereatedResponse = tf.expand_dims(genereatedResponse,2)

	encodingUttrerance = tf.expand_dims(encodedUtternace,2)



	## genereating the logit
	logits = tf.matmul(genereatedResponse, encodingUttrerance, True) 
	logits = tf.squeeze(logits, [2])

	probs = tf.sigmoid(logits)


	if mode == tf.contrib.learn.ModeKeys.INFER:
		return probs, None

	losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(targets), logits=logits)

	meanLoss = tf.reduce_mean(losses, name="mean_loss")

	return probs, meanLoss