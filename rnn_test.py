import sys
import re
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from model import loadModel
from dataset import getCleanWords, getCleanSequences
from dataset import condenseTexts, generator, getTestData
from dataset import getEmbeddingWeights

def sample(preds, temperature=1.0):
    """Use the temperature to add variety to the predicted
    sentences
    """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def performTest(weightsFilename):
    #Constants
    """filename = "corpus.txt"
    maxSeqLength = 20
    seedLength = 8"""
    temperatures = [0.20, 0.25, 0.30, 0.35, 0.40]
    #maskValue = -10
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
    #Get the training data from the raw text
    #sentences, nextWords = getCleanSequences(rawText.split(), maxSeqLength, wordToInt, stride)
    #vocab = getVocab(tokenizer, nVocab)
    #Get word embeddings
    #embeddings = getWordEmbeddings(tokenizer, seqs, nVocab)
    embeddingWeights = getEmbeddingWeights(cleanWords, tokenizer, wordVecs)

    #Get the RNN model and output a summary
    model = loadModel(maxSeqLength, nVocab, embeddingWeights)

    #Allow memory to grow on GPU
    #gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    #for device in gpu_devices:
        #tf.config.experimental.set_memory_growth(device, True)

    #Load ASCII text and return a mapping from characters to integers,
    #and retrieve the number of total characters and unique characters
    """rawText, wordToInt, nWords, nVocab = summariseDataset(filename)
    #Create a mapping from integers to characters
    intToWord = dict((i, c) for c, i in wordToInt.items())
    #print(pattern)
    #print(pattern.shape)
    #Get word embeddings
    embeddings = getWordEmbeddings(list(set(rawText.split())))

    #Get the RNN model
    model = loadModel(maxSeqLength, nVocab, embeddings)"""

    #Load the network weights
    filename = weightsFilename
    #"weights-improvement-35-1.1789-bigger.hdf5"#"weights-improvement-50-1.0037-bigger.hdf5"
    model.load_weights(filename)


    #Generate characters using the seed until the end is reached or the max size is exceeded
    currentWord = None
    #wordIndex = seedLength
    maxPredictionLen = 100
    basePattern = getTestData(cleanWords, seedLength, maxSeqLength, tokenizer)
    for temperature in temperatures:
        #Get a random piece of test data
        pattern = np.copy(basePattern)
        wordIndex = seedLength
        #result = ' '.join([vocab[pattern[0, i]-1] for i in range(seedLength)])
        #print(result)
        result = ' '.join(tokenizer.sequences_to_texts(pattern)) + " "
        while wordIndex < maxPredictionLen:
    #not re.search('(?<!(.{1}\smr|\smrs|.{2}\sm|\smme|mlle))\\.\s', result) and len(result) < maxPredictionLen:
        #Predict the next character using the RNN
        #paddedPattern = np.pad(pattern, ((0, 0), (0, maxSeqLength - len(result)), (0, 0)), constant_values=maskValue)
            prediction = model.predict(pattern, verbose=0)[0]
            #print(len(prediction))
            #prediction = prediction[0]
            #print(len(prediction))
            #print("Prediction: {}".format(prediction))
            index = sample(prediction, temperature)
            #index = model.predict_classes(pattern, verbose=0)[0]
            #print(index)
            #index = index[0]
            #print("Index: {}".format(index))
            #print("Current word: {}".format(tokenizer.sequences_to_texts([[index+1]])))
            currentWord = tokenizer.sequences_to_texts([[index+1]])[0]
        #One-hot encode the outputted character
        #oneHot = np.zeros((nVocab))
        #oneHot[index] = 1.0
        #print(pattern)
        #Update pattern using newly generated character
        #pattern[0:seqLength-1] = pattern[1:seqLength]
            if wordIndex >= maxSeqLength:
                pattern[0, 0:maxSeqLength-1] = pattern[0, 1:maxSeqLength]
                pattern[0, maxSeqLength-1] = index + 1
            #print(pattern)
            else:
                pattern[0, wordIndex] = index + 1

            #print("Updated pattern: {}".format(tokenizer.sequences_to_texts(pattern)))
        #print(pattern)
            result += (currentWord + " ")
            wordIndex += 1

        print("Generated text with temperature {}:\n{}".format(temperature, result.strip()))
