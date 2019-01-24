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
import pandas as pd


def load_docs(filename, vocab, map):
	text = list()
	label = list()
	with open(filename, encoding='utf8') as f:
		reader = csv.reader(f,delimiter = "\t")
		#next(reader)
		counts = [0,0,0,0]
		originals = list()
		for row in reader:
			if len(row) == 2:
				originalText = row[0]
				text.append(clean_doc(row[0], vocab))
				lbl = -1
				for i in range(0,len(map)):
					if map[i] == row[1].replace('"',''):
						lbl = i
						label.append(i)
						counts[i] = counts[i]+1
						originals.append(originalText)
				if lbl == -1:
					print("error on: " + row[1])
	return text, label, counts, originals


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

[all_docs, labels, counts, originals] = load_docs('BigDataset.tsv', vocab, map)

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
# pad sequences
max_length = max([len(s.split()) for s in all_docs])
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(len(padded_docs))

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
# load embedding from file
raw_embedding = load_embedding('glove.6B/glove.6B.' + str(embeddingSize) + 'd.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
# Load model
from keras.models import model_from_json

# Loading model and weights
json_file = open('final_network/nn.json', 'r')
nn_json = json_file.read()
json_file.close()
model = model_from_json(nn_json)
model.load_weights("final_network/weights.hdf5")

# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Results
# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

# Compute probabilities
Y_pred = model.predict(padded_docs)
# Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)

with open('finalDataset.tsv','w',encoding='utf8')as f:
	f.write('Question\tType\n')
	for j in range(0,len(padded_docs)):
		f.write(originals[j] + '\t' + map[y_pred[j]]+'\n')


