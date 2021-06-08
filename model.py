import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, BatchNormalization
from keras.optimizers import RMSprop
from keras.constraints import max_norm

'''def loadEmbeddings():
    """
    Load the GloVE embeddings
    """
    #wordVecs = {}
    #count = 0

    embeddings = np.load("glove/embeddings.npy", mmap_mode='r')
    #with open("glove/embeddings.vocab", "r", encoding="utf8") as fileRead:
        #for index, word in enumerate(fileRead):
            #wordVecs[word.strip()] = loadedVectors[index]

    return embeddings
'''
def loadModel(seqLength, nVocab, embeddingWeights):
    print(embeddingWeights)
    model = Sequential()
    #model.add(Masking(mask_value=maskValue, input_shape=(seqLength, nVocab)))
    model.add(Embedding(nVocab+1, 50, input_shape=(seqLength,)))
    #model.add(LSTM(128, return_sequences=True))
    #model.add(Dropout(0.20))
    model.add(LSTM(256, kernel_constraint=max_norm(3), recurrent_constraint=max_norm(3), bias_constraint=max_norm(3), return_sequences=True))
    #model.add(Dropout(0.20))
    #model.add(BatchNormalization())
    model.add(LSTM(256, kernel_constraint=max_norm(3), recurrent_constraint=max_norm(3), bias_constraint=max_norm(3)))
    #model.add(Dropout(0.20))
    #model.add(Dense(128, activation="relu"))
    #model.add(Dropout(0.20))
    model.add(Dense(nVocab, kernel_constraint=max_norm(3), bias_constraint=max_norm(3),  activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
    
    return model
