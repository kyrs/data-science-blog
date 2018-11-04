import os 
import time 
import itertools
import tensorflow as tf
import udcModel
import udcMetrics
import paramInit
import loadingTfRecod
from model.dualEncoder import dualEncoderModel
from model.helperFn import loadVocab
import sys 
import numpy as np 

tf.flags.DEFINE_string("model_dir", './runs/1541325014', "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "D:\\data\\data\\vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
	print ("you must specify the model dir")
	sys.exit()

def tokenizer_fn(iterator):
	return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

INPUT_CONTEXT = "Example context"
POTENTIAL_RESPONSES = ["Response 1", "Response 2"]

def getFeatures(context, utterance):
	contextMatrix = np.array(list(vp.transform([context])))
	utteranceMatrix = np.array(list(vp.transform([utterance])))
	contextLen = len(context.split(" "))
	utteranceLen = len(utterance.split(" "))
	features = {
	"context": tf.convert_to_tensor(contextMatrix, dtype=tf.int64),
	"context_len": tf.constant(contextLen, shape=[1,1], dtype=tf.int64),
	"utterance": tf.convert_to_tensor(utteranceMatrix, dtype=tf.int64),
	"utterance_len": tf.constant(utteranceLen, shape=[1,1], dtype=tf.int64),
	}
	return features, None

if __name__ == "__main__":
	hparams = paramInit.createHparams()
	modelFn = udcModel.createModelFn(hparams, modelImpl=dualEncoderModel)
	estimator = tf.contrib.learn.Estimator(model_fn=modelFn, model_dir=FLAGS.model_dir)

	
	print("Context: {}".format(INPUT_CONTEXT))
	for r in POTENTIAL_RESPONSES:
		prob = estimator.predict(input_fn=lambda: getFeatures(INPUT_CONTEXT, r))
		print(list(prob))