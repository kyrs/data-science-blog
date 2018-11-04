import array
import numpy as np 
import tensorflow as tf 
from collections import defaultdict

def loadVocab(filename):
	vocab = None
	with open(filename) as f:
		vocab = f.read().splitlines()
		dct = defaultdict(int)
		for idx, word in enumerate(vocab):
			dct[word] = idx
	return [vocab, dct]

def loadGloveVectors(fileName,vocab):
	## loading the glove vectors 

	dct = {}
	vectors = array.array('d')
	currentIdx = 0
	with open(fileName,'r',encoding="utf-8") as f:
		for _,line in ennumerate(f):
			tokens = line.split(" ")
			word = token[0]
			entries = token[1:]
			if not vocab or word in vocab:
				dct[word] = currentIdx
				vectors.extend(float(x) for x in entries)
				currentIdx+=1
			else:
				pass
		wordDim = len(entries)
		numVectors = len(dct)
		tf.logging.info("Found {} out of {} vectors in Glove".format(numVectors, len(vocab)))
		return [np.array(vectors).reshape(numVectors, wordDim), dct]

def buildInitialEmbeddingMatrix(vocabDict, gloveDict, gloveVectors, embeddingDim):
	initialEmbeddings = np.random.uniform(-0.25, 0.25, (len(vocabDict), embeddingDim)).astype("float32")
	for word, gloveWordIdx in gloveDict.items():
		wordIdx = vocab_dict.get(word)
		initialEmbeddings[wordIdx, :] = gloveVectors[gloveWordIdx]
	return initialEmbeddings