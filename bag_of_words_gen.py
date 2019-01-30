import csv
import numpy as np
import random
#import sklearn

from sklearn.feature_extraction.text import CountVectorizer
np.set_printoptions(threshold='nan')

class DataIngestion():

	'''Data Ingestion classes are based on models. Each model requires specific data in a specific format. Each instance of DataIngestion() will format data
	for a specific model type and application. That type and application are given by an a model type, and ID. Both are optional, and defaults to our
	Generic model type with no ID. If this is the case it will assume it's parsing english sentences. It will then associate decisions as binary.'''

	def __init__(self):
		#self.data needs to be a bag of words. I.e. {"For":3, "Me":1, "hello":4}. References the number of times a word is used in a set of data
		self.data = {}
		#self.index references the specific project. 'none' refers to no project, other values refer to a project ID
		self.index = 'none'
		#self.decision needs to be the 'key' or 'decoder' for values to meaningful information. I.e. {"KB01039439":1, "KB10239853":2}. A 1 in the 'label' or 'decision' refers to KB01039439
		self.decision = {}
		self.train = True
		self.test = False
		self.predict = False


	def num_decisions(self, num):
		'''Create matrix decisions for the number of possible decisions <num>. I.e. 3 possible decisions yields [1,0,0], [0,1,0], [0,0,1]'''
		#num = len(self.decision.keys())
		decision = []
		for i in range(0, num):
			temp = np.zeros(num)
			temp[i] = 1
			decision.append(temp)
			del temp
		return decision


	def get_data(self, incidentData, num):
		'''Grabs data from <incidentData> file. Must be in ./modlel folder, and deliminitaed by commas. See ./model/incidents for specific formatting'''
		'''<num> is the number of decisions that need to be made. I.e. 3, 4, etc...'''
		words = []
		labels = []
		decisions = self.num_decisions(num)
		with open('./model/{}'.format(incidentData), 'rt') as incidents:
			spamreader = csv.reader(incidents, delimiter=',')
			for row in spamreader:
				#Add word to words list
				words.append(str(row[1]))
				#Add decision based on index in decision matrix
				labels.append(list(decisions[int(row[2])-1]))

				# if row[2] == '1':
				# 	labels.append([1,0,0,0,0,0])
				# elif row[2] =='2':
				# 	labels.append([0,1,0,0,0,0])
				# elif row[2] == '3':
				# 	labels.append([0,0,1,0,0,0])
				# elif row[2] == '4':
				# 	labels.append([0,0,0,1,0,0])
				# elif row[2] == '5':
				# 	labels.append([0,0,0,0,1,0])
				# elif row[2] == '6':
				# 	labels.append([0,0,0,0,0,1])
				# else:
				# 	labels.append([0,0,0,0,0,0])

		return words, labels

	def shuffle(self, words, labels):
		'''Shuffles words and labels so they are random. Words and labels are still linked by index of list/array'''
		z = list(zip(words, labels))
		random.shuffle(z)
		words[:], labels[:] = zip(*z)
		return words, labels

	def master_bow(self, referenceIncidents, num):
		''' Will create the reference bag of words to be used for analysis'''
		words, labels = self.get_data(referenceIncidents, num)
		words, labels = self.shuffle(words, labels)
		vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_df=1., min_df=0.000001, max_features=625)
		master_vocab = vectorizer.fit(words)
		return master_vocab.vocabulary_

	def transform_to_ingest(self, words, labels, referenceIncidents, num):
		''' Transforms a set of words and labels to be ingested by the model based on the master_bow() function creating the master bag of words'''
		master_vocab = self.master_bow(referenceIncidents, num)
		vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_df=1., min_df=0.000001, max_features=625, vocabulary=master_vocab)
		token_counts = vectorizer.transform(words)
		token_counts_matrix = token_counts.toarray()
		master_dict = {"words": token_counts_matrix, "labels": labels}
		return master_dict


	def test_model_ingestion(self, trainData, analyzeData, num):
		'''Creates the model ingestion for testing the model. Takes in training data file, and analyze data file and sends the information to the model
			based on <trainData> for the master list of words, and analyzeData will be used as the set of data to predict on'''
		historical_words, historical_labels = self.get_data(trainData, num)
		new_words, new_labels = self.get_data(analyzeData, num)
		historical_data = self.transform_to_ingest(historical_words, historical_labels, trainData, num)
		new_data = self.transform_to_ingest(new_words, new_labels, trainData, num)
		return historical_data, new_data

# if __name__ == '__main__':
#     test_model_ingestion()
