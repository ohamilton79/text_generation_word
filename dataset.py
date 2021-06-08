import numpy as np
import re
from random import randint
from keras.preprocessing.text import Tokenizer

def getGloveEmbeddings():
    """Load the 50-dimensional GloVE embeddings
    """
    wordVecs = {}
    
    embeddings = np.load("glove/embeddings.npy", mmap_mode='r')
    with open("glove/embeddings.vocab", "r", encoding="utf8") as fileRead:
        for index, word in enumerate(fileRead):
            wordVecs[word.strip()] = embeddings[index]

    return wordVecs

def getCleanWords(filename):
    #Read the dataset file
    rawText = open(filename, 'r', encoding='utf-8').read()
    #Convert text to lowercase
    rawText = rawText.lower()
    #Separate punctation with spaces from words
    punctuation = '!"£$%^&*()-;:\'\’#?,.`‘“”_—'
    for char in punctuation:
        #print(char)
        rawText = rawText.replace(char, ' {} '.format(char))
    #rawText = re.sub('[!\?£$%^\&*()\;:#\.,"`]', '', rawText)
    #rawText = re.sub(' - ', ' ', rawText)
    #rawText = re.sub(' -', ' ', rawText)
    #rawText = re.sub('- ', ' ', rawText)
    #Remove line breaks and indentation
    #rawText = rawText.replace('\n', ' ')
    #rawText = rawText.replace('\t', '')
    #Remove any double spaces created when adjusting punctuation
    while '  ' in rawText:
        rawText = rawText.replace('  ', ' ')
    #Create a mapping from words to integers
    #print(rawText)
    #words = sorted(list(set(rawText.split())))
    #wordToInt = dict((c, i) for i, c in enumerate(words))
    #tokenizer = Tokenizer(num_words=nWords)
    #tokenizer.fit_on_texts(rawText.split())
    #Get the total number of words in the corpus
    #nWords = len(rawText.split())
    #nVocab = len(words)
    #Convert text into a list of words
    cleanWords = rawText.split()

    return cleanWords

#Get the vocabulary from the tokenizer by selecting the most common words
"""def getVocab(tokenizer, nVocab):
    vocab = sorted(tokenizer.word_docs.items(), key=lambda x: x[1], reverse=True)[0:nVocab]
    return [item[0] for item in vocab]"""

'''def getVocabulary(rawText, nVocab, embeddings):
    """Remove any words that don't have GloVE embeddings to
    form the vocabulary
    """
    vocabWords = []
    for word in cleanWords:
        if word in embeddings:
            vocabWords.append(word)

    #Tokenise words
    tokenizer = Tokenizer(num_words=nVocab)
    tokenizer.fit_on_texts(vocabWords)

    #Return the mapping from words to integers
    return tokenizer.word_index'''

def getEmbeddingWeights(cleanWords, tokenizer, wordVecs):
    vocab = tokenizer.sequences_to_texts(tokenizer.texts_to_sequences(list(set(cleanWords))))
    #Flatten
    #vocab = [entry for sublist in vocab for entry in sublist]
    #Remove empty entries
    trimmedVocab = [entry for entry in vocab if entry]
    #print(trimmedVocab)
    weights = np.zeros((len(trimmedVocab)+1, 50))
    #print(len(wordToInt))
    #index = 0
    #print(vocab)
    for word in list(set(cleanWords)):
        if word in vocab and word in wordVecs:
            #print(word)
            index = tokenizer.texts_to_sequences([word])
            weights[index[0][0]] = wordVecs[word]
            #index += 1
            #print(index)
            #print(index[0][0])

    return weights

#Remove sequences that contain words which aren't in the vocabulary or don't have embeddings
def getCleanSequences(cleanWords, seqLength, stride):
    #wordVecs = loadEmbeddingMatrix()
    seqs = []
    #sentences = []
    #nextWords = []
    #newWords = [None]*len(cleanWords)
    #print("B", len(cleanWords))
    i = 0
    while i < len(cleanWords) - seqLength:
        #validSequence = True
        seq = cleanWords[i:i + seqLength+1]
        #seqOut = cleanWords[i + seqLength]
        #for word in seq:
            #if not word in embeddings:
                #validSequence = False

            #elif not word in newWords:
                #newWords.append(word)

        #if validSequence:
        seqs.append(seq)
        #newWords[i:i + seqLength+1] = seq

        #if 0 in sentences[-1]:
            #print(sentences[-1])
        #Get next character sequence
        i += stride
        #Get next sequence length
        #seqLength = randint(seedLength, maxSeqLength)
        #print(len(seqIn))

    #Remove null words from words list
    #newWords = [word for word in newWords if word]

    return seqs

def condenseTexts(seqs, seqLength, nVocab):
    sentences = []
    nextWords = []

    wordVecs = getGloveEmbeddings()
    
    tokenizer = Tokenizer(num_words=nVocab+1, filters='\t\n')
    tokenizer.fit_on_texts(seqs)

    #embeddings = getWordEmbeddings(tokenizer, seqs, nVocab)
    #print("Embeddings: {}".format(embeddings))
    validSeqs = []
    #Remove any sequences that don't have corresponding word vectors
    for seq in seqs:
        validSeq = True
        for word in seq:
            if not word in wordVecs:
                validSeq = False
        if validSeq:
            validSeqs.append(seq)

    sequences = tokenizer.texts_to_sequences(validSeqs)
    #Shuffle sequences
    np.random.shuffle(sequences)
    #print(sequences)
    #print("AAA")
    #print(sequences)
    #print(sequences[0:10])
    for index in range(len(sequences)):
        #print(index)
        #print(sequences[index])
        if len(sequences[index]) == seqLength + 1:
            sentences.append(sequences[index][0:seqLength])
            nextWords.append(sequences[index][seqLength])
        #else:
            #print("Caught!")
    #sentences = [sequence[0:seqLength] for sequence in sequences if len(sequence) == seqLength + 1]
    #nextWords = [sequence[seqLength] for sequence in sequences if len(sequence) == seqLength + 1]

    return sentences, nextWords, tokenizer, wordVecs

def generator(sentences, nextWords, batchSize, nVocab, seqLength):
    #nVocab = len(wordToInt)
    #print(nVocab)
    #tokenizer = Tokenizer(num_words=nVocab+1, filters='\t\n')
    
    #tokenizer.fit_on_texts(seqs)
    #sequences = tokenizer.texts_to_sequences(seqs)
    #sentences = [sequence[0:seqLength] for sequence in sequences if len(sequence) == seqLength + 1]
    #nextWords = [sequence[seqLength] for sequence in sequences if len(sequence) == seqLength + 1]

    index = 0
    #print(sentences, nextWords)
    while True:
        x = np.zeros((batchSize, seqLength), dtype=np.int32)
        y = np.zeros((batchSize, nVocab), dtype=np.float32)
        #Generate data items in the batch
        for i in range(batchSize):
            #print(i, index)
            #if len(sequences[index]) == seqLength + 1:
                #for t, w in enumerate(sequences[index]):
                #if len(sequences[index]) == seqLength + 1:
            x[i] = np.array(sentences[index])

            y[i, nextWords[index]-1] = 1.0

            index += 1
            #Restart once end of dataset reached
            if index == len(sentences):
                index = 0
        #print(x, y)
        yield x, y

'''
def getWordEmbeddings(tokenizer, seqs, nVocab):
    """Load the GloVE word embeddings for the words in the training dataset
    """
    nDims = 50
    wordVecs = getGloveEmbeddings()
    embeddings = np.zeros((nVocab + 1, 50))
    uniqueWords = list(set([word for seq in seqs for word in seq]))
    #wordIndices = list(wordVecs.keys())
    index = 0
    for index, token in enumerate(tokenizer.texts_to_sequences(uniqueWords)):
        #print(index, token, uniqueWords[index])
        if uniqueWords[index] in wordVecs and token:
            #print("True!")
            #embeddingIndex = wordIndices.index(word)
            embeddings[token] = wordVecs[uniqueWords[index]]

        #if word:
        index += 1
            #print(index, word)

        #else:
            #print(word)

    return embeddings
'''
def getTestData(cleanWords, seedLength, maxSeqLength, tokenizer):
    #print(rawText[0:1000])
    #Split the corpus into a list of sentences, ignoring periods for title abbreviations
    #sentences = re.sub("(?<!(.{1}\smr|\smrs|.{2}\sm|\smme|mlle))\\.\s", ".//", rawText).split("//")
    #Keep only the sentences of at least the required length
    #sentences = [sentence for sentence in sentences if len(sentence) > seedLength]
    #print(sentences[0:5])
    vocab = tokenizer.sequences_to_texts(tokenizer.texts_to_sequences(cleanWords))
    #Pick a random sentence index to choose
    valid = False
    while not valid:
        valid = True
        #print(vocabulary)
        i = np.random.randint(0, len(cleanWords)-1-seedLength)
        for word in cleanWords[i:i+seedLength]:
            #print(word)
            if not word in vocab:
                valid = False
                #print(word)

    #print(sentences[i])
    #Use the random number to get a random seed
    seqIn = cleanWords[i:i+seedLength]
    #print(seqIn)
    #seqOut = rawText.split()[i+seedLength]
    sentence = tokenizer.texts_to_sequences(seqIn)
    #sentence = [vocabulary.index(word) + 1 for word in seqIn]
    #sentence = [wordToInt[word]+1 for word in seqIn]

    pattern = np.zeros((1, len(sentence)), dtype=np.int32)
    pattern[0] = np.array(sentence).flatten()
    #One hot encode the network inputs
    #print(pattern)
    #for j, char in enumerate(sentence):
        #pattern[0, j, char] = 1.0
    #Pad the pattern so it matches the maximum sequence length
    paddedPattern = np.pad(pattern, ((0, 0), (0, maxSeqLength - seedLength)))
    return paddedPattern
