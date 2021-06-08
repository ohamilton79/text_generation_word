# LSTM Network to generate text inspired by Oliver Twist by Charles Dickens
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import np_utils
import re
import math

from model import loadModel
from dataset import getCleanWords, getCleanSequences
from dataset import condenseTexts, generator
from dataset import getEmbeddingWeights
from rnn_test import performTest

class ValidationCallback(Callback):
    #At the end of each epoch, test the network
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 1 == 0:
            #print(self.model.layers[0].get_weights()[0])
            #print(self.model.layers[1].get_weights()[0])
            #print(self.model.layers[2].get_weights()[0])
            #print(self.model.layers[3].get_weights()[0])
            performTest("data/weights-{}.hdf5".format(epoch+1))

#Constants
filename = "corpus.txt"
maxSeqLength = 30
seedLength = 30
stride = 1
#maskValue = -10

#Allow memory to grow on GPU
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#Load ASCII text and return a mapping between characters and integers,
#and retrieve the number of total characters and unique characters
nVocab = 5000
#embeddings = getGloveEmbeddings()
cleanWords = getCleanWords(filename)
#print(cleanWords[0:30])
#wordToInt = getVocabulary(cleanWords, nVocab, embeddings)
#print(len(wordToInt))
seqs = getCleanSequences(cleanWords, maxSeqLength, stride)
#nVocab = len(list(set(cleanWords)))
#rawText, tokenizer, nWords = summariseDataset(filename, nVocab)
#wordToInt = tokenizer.word_index
sentences, nextWords, tokenizer, wordVecs = condenseTexts(seqs, maxSeqLength, nVocab)
#print(tokenizer.sequences_to_texts(sentences[0:4]))
#print(tokenizer.sequences_to_texts([nextWords[0:4]]))
#Get the training data from the raw text
#sentences, nextWords = getCleanSequences(rawText.split(), maxSeqLength, wordToInt, stride)
#vocab = getVocab(tokenizer, nVocab)
#Get word embeddings
#embeddings = getWordEmbeddings(tokenizer, seqs, nVocab)
embeddingWeights = getEmbeddingWeights(cleanWords, tokenizer, wordVecs)

#Get the RNN model and output a summary
model = loadModel(maxSeqLength, nVocab, embeddingWeights)
model.summary()

#Define the checkpoint to save the best weights
filepath="data/weights-{epoch:d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='min')

#Fit the model using the training data
batchSize = 128
model.fit_generator(generator(sentences, nextWords, batchSize, nVocab, maxSeqLength), steps_per_epoch=math.ceil(len(sentences) / batchSize), epochs=500, callbacks=[checkpoint, ValidationCallback()])
