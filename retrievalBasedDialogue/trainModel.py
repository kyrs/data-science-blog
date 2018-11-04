import os 
import time 
import itertools
import tensorflow as tf
import udcModel
import udcMetrics
import paramInit
import loadingTfRecod
from model.dualEncoder import dualEncoderModel


tf.flags.DEFINE_string("input_dir", 'D:\\data\\data', "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 2000, "Evaluate after this many train steps")
FLAGS = tf.flags.FLAGS

TIMESTAMP = int(time.time())

if FLAGS.model_dir:
	modelDir =  FLAGS.model_dir
else:
	modelDir = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))

TrainFile= os.path.abspath(os.path.join(FLAGS.input_dir, "train.tfrecords"))
ValidationFile = os.path.abspath(os.path.join(FLAGS.input_dir, "validation.tfrecords"))

tf.logging.set_verbosity(FLAGS.loglevel)

def main(unusedArgv):
	hparams = paramInit.createHparams()
	modelFn = udcModel.createModelFn(hparams,modelImpl=dualEncoderModel)



	estimator = tf.contrib.learn.Estimator(
		model_fn=modelFn,
		model_dir=modelDir,
		config=tf.contrib.learn.RunConfig())

	inputFnTrain = loadingTfRecod.create_input_fn(
		mode = tf.contrib.learn.ModeKeys.TRAIN,
		input_files = TrainFile,
		batch_size = hparams.batch_size,
		num_epochs=FLAGS.num_epochs 
		)

	inputFnEval = loadingTfRecod.create_input_fn(
		mode = tf.contrib.learn.ModeKeys.EVAL,
		input_files = ValidationFile,
		batch_size = hparams.batch_size,
		num_epochs=FLAGS.num_epochs 
		)

	evalMetrics = udcMetrics.create_evaluation_metrics()

	evalMonitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=inputFnEval,
        every_n_steps=FLAGS.eval_every,
        metrics=evalMetrics)


	estimator.fit(input_fn=inputFnTrain, steps=None, monitors=[evalMonitor])

if __name__ == "__main__":
	tf.app.run()
	