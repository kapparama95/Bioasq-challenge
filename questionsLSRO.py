from string import punctuation
from os import listdir
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import np_utils
from keras import callbacks
import csv
from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd


def load_docs(filename, vocab, map, nrows):
	text = list()
	label = list()
	loadedRows = 0
	with open(filename, encoding='utf8') as f:
		reader = csv.reader(f,delimiter = "\t")
		#next(reader)
		counts = [0,0,0,0]
		for row in reader:
			if len(row) == 2:
				text.append(clean_doc(row[0], vocab))
				lbl = -1
				for i in range(0,len(map)):
					if map[i] == row[1].replace('"',''):
						lbl = i
						label.append(i)
						counts[i] = counts[i]+1
						loadedRows += 1
				if lbl == -1:
					print("error on: " + row[1])
			if loadedRows == nrows-1:
				break
	return text, label, counts

def addDocs(filename,vocab,map,dataset,labels):
	with open(filename) as f:
		reader = csv.reader(f,delimiter = ";")
		next(reader)
		for row in reader:
			dataset.append(clean_doc(row[0], vocab))
			lbl = -1
			for i in range(0,len(map)):
				if map[i] == row[1]:
					lbl = i
					labels.append(i)
			if lbl == -1:
				print("error on: " + row[1])

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding="cp1252")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens


# load all docs in a directory
def process_docs(directory, vocab):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents


# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename, 'r', encoding="UTF8")
	lines = file.readlines()
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
	return embedding


# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = np.zeros((vocab_size, embeddingSize))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		vector = embedding.get(word)
		if vector is not None:
			weight_matrix[i] = vector
	return weight_matrix


# size of the embedding that will be used
embeddingSize = 100
# load the vocabulary
vocab_filename = 'questionsVocab2d.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

#define mapping from label to number
map = ["yesno","factoid","summary","list"]
# load all training reviews
unlabelled_samples  = (2252-400)*3
[all_docs, labels, counts] = load_docs('BigDataset.tsv', vocab, map,unlabelled_samples)

addDocs('Questions.CSV',vocab,map,all_docs,labels)

print('n_values: ' + str(len(all_docs)))
print('/*-- instance count --*\\')
print(map)
print(counts)

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(all_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(all_docs)
print(tokenizer)
# pad sequences
max_length = max([len(s.split()) for s in all_docs])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(len(padded_docs))
Xtest = padded_docs[-400:]
Ytest = labels[-400:]
padded_docs = padded_docs[:-401]
labels = labels[:-401]

yTrain_c = np_utils.to_categorical(labels, 4)

#LSRO
for i in range(0,unlabelled_samples):
	for j in range(0,4):
		yTrain_c[i][j] = 0.25

Xtrain,yTrain_c = sklearn.utils.shuffle(padded_docs,yTrain_c)

#Xtrain, Xtest, Ytrain, Ytest = train_test_split(padded_docs, labels, test_size=0.1)

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
# load embedding from file
raw_embedding = load_embedding('glove.6B/glove.6B.' + str(embeddingSize) + 'd.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, embeddingSize, weights=[embedding_vectors], input_length=max_length,
							trainable=False)




# define model
nn = Sequential()
nn.add(embedding_layer)
nn.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
nn.add(MaxPooling1D(pool_size=2))
nn.add(Flatten())
nn.add(Dropout(0.4))
nn.add(Dense(64, activation='relu'))
nn.add(Dense(4, activation='softmax'))
print(nn.summary())
# compile network
nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], )
# fit network

callback = callbacks.EarlyStopping(monitor='val_acc',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')
history = nn.fit(Xtrain, yTrain_c, epochs=5, validation_split=0.10)

##Store Plots
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# No validation loss in this example
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('output/model_accuracy.pdf')
plt.close()
# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('output/model_loss.pdf')

# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Results
# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

# Compute probabilities
Y_pred = nn.predict(Xtest)
# Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)

# Plot statistics
print('Analysis of results')
target_names = map
print(classification_report(Ytest, y_pred, target_names=target_names))
print(confusion_matrix(Ytest, y_pred))

nn_json = nn.to_json()
with open('output/nn.json', 'w') as json_file:
	json_file.write(nn_json)
weights_file = "output/weights" + ".hdf5"
nn.save_weights(weights_file, overwrite=True)