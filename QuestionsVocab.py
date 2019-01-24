from string import punctuation
from os import listdir
from collections import Counter
import csv
# load doc into memory
def process_docs(filename, vocab):
	with open(filename,encoding='UTF8') as f:
		reader = csv.reader(f)
		next(reader)
		for row in reader:
			vocab.update(clean_doc(row[0]))

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens



# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('DT_SVMnewdata.tsv', vocab)

# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
# keep tokens with a min occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))
# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()

# save tokens to a vocabulary file
save_list(tokens, 'questionsVocab3.txt')