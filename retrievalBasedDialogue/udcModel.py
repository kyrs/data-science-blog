import tensorflow as tf 
import sys 

def getIdFeature(features,key,lenKey,maxLen):
	ids = features[key]
	idsLen = tf.squeeze(features[lenKey], [1])
	idsLen = tf.minimum(idsLen,tf.constant(maxLen,tf.int64))

	return ids,idsLen


def createTrainOp(loss,hparams):
	trainOp = tf.contrib.layers.optimize_loss(
		loss = loss,
		global_step = tf.contrib.framework.get_global_step(),
		learning_rate=hparams.learning_rate,
		clip_gradients=10.0,
		optimizer=hparams.optimizer

		)

	return trainOp

def createModelFn(hparams,modelImpl):
	def modelFn(features,targets,mode):
		context,contextLen = getIdFeature(features,"context","context_len",hparams.max_context_len)

		utterance,utteranceLen = getIdFeature(features,"utterance","utterance_len",hparams.max_utterance_len)

		if targets is not None:
			batchSize = targets.get_shape().as_list()[0]


		if mode ==tf.contrib.learn.ModeKeys.TRAIN:
			probs, loss = modelImpl(
			hparams,
			mode,
			context,
			contextLen,
			utterance,
			utteranceLen,
			targets)
			trainOp = createTrainOp(loss, hparams)
			return probs, loss, trainOp

		if mode ==tf.contrib.learn.ModeKeys.INFER:
			probs, loss = modelImpl(
			hparams,
			mode,
			context,
			contextLen,
			utterance,
			utteranceLen,
			None)
			return probs,0.0,None

		if mode == tf.contrib.learn.ModeKeys.EVAL:
			allContexts = [context]
			allContextLen = [contextLen]
			allUtterance = [utterance]
			allUtteranceLen = [utteranceLen]
			allTarget = [tf.ones([batchSize,1],dtype = tf.int64)]


			for i in range(9):
				distractor,distractorLen = getIdFeature(
						features,
						"distractor_{}".format(i),
						"distractor_{}_len".format(i),
						hparams.max_utterance_len

					)

				allContexts.append(context)
				allContextLen.append(contextLen)
				allUtterance.append(distractor)
				allUtteranceLen.append(distractorLen)

				allTarget.append(tf.zeros([batchSize,1],dtype=tf.int64))

			
			prob,loss = modelImpl(

					hparams,
					mode,
					tf.concat(allContexts,0),
					tf.concat(allContextLen,0),
					tf.concat(allUtterance,0),
					tf.concat(allUtteranceLen,0),
					tf.concat(allTarget,0)
				)	
			splitPorb = tf.split(prob,10,0)

			shapedProb = tf.concat(splitPorb,1)

			tf.summary.histogram("eval_correct_probs_hist", split_probs[0])
			tf.summary.scalar("eval_correct_probs_average", tf.reduce_mean(split_probs[0]))	
			tf.summary.histogram("eval_incorrect_probs_hist", split_probs[1])	    
			tf.summary.scalar("eval_correct_probs_average", tf.reduce_mean(split_probs[0]))
			return shapedProb,loss,None
	return modelFn