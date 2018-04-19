import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import re
import pandas as pd


def feature_generate(example_title, example_description, example_requirements, example_city, example_state):
	def cleanhtml(raw_html):
		cleanr = re.compile('<.*?>')
		cleantext = re.sub(cleanr, ' ', raw_html) # Using a regex to clean everything inside <> : including </b> </p> \r <p>  <b>
		cleantext = cleantext.replace('\r', ' ').replace('\n', ' ').replace('&nbsp;', ' ') #remove &nbsp; \r \n
		cleantext = cleantext.replace('Please refer to the Job Description to view the requirements for this job', '')
		cleantext = cleantext.rstrip() #remove redundant space
		return cleantext.lower() # convert string to lowercase

	#example_title = df.iloc[i]['Title']
	#example_state = df.iloc[i]['State']
	#example_city = df.iloc[i]['City']
	#example_requirements = df.iloc[i]['Requirements']
	#example_description = df.iloc[i]['Description']

	tokenizer = RegexpTokenizer(r'\w+') # get rid of punctuation using NLTK tokenizer. 
	#nltk.download('stopwords')
	#nltk.download('punkt')
	stop_words = set(stopwords.words("english"))# how to strip not only 'you' but also 'YOU'?
	stop_words.add('r') # it is hard to remove '\r' from the raw text, so manually add 'r' as a stop word
	stop_words.add('n') # same, get rid of '\n'
	stop_words.add('etc') # etc.
	#print(stop_words)

	def clean_tokenize_filteredStopWords(raw_str):
		raw_str = cleanhtml(raw_str)
		token_list = tokenizer.tokenize(raw_str)
		filtered_text = []
		filtered_text = [w for w in token_list if not w in stop_words]
		return filtered_text

	filtered_description = clean_tokenize_filteredStopWords(example_description)
	filtered_requirements = clean_tokenize_filteredStopWords(example_requirements)
	filtered_title = clean_tokenize_filteredStopWords(example_title)
	filtered_state = clean_tokenize_filteredStopWords(example_state)
	filtered_city = clean_tokenize_filteredStopWords(example_city)


	filtered_result = filtered_description + filtered_requirements
	length = len(filtered_description) + len(filtered_requirements)

	title_weight = 1 + length // 100
	for i in list(range(0, title_weight)):
		filtered_result.extend(filtered_title) # extend not append. See: https://stackoverflow.com/questions/252703/difference-between-append-vs-extend-list-methods-in-python

	city_weight = 1 + length // 200
	for i in list(range(0, city_weight)):
		filtered_result.extend(filtered_city)

	state_weight = 1 + length // 300
	for i in list(range(0, state_weight)):
		filtered_result.extend(filtered_state)

	#print(filtered_result)
	# Alternative method
	# for w in words:
	# 	if w not in stop_words:
	# 		filtered_sentence.append(w)

	#Stemming using Porter Stemming Algorithm

	# ps = PorterStemmer()
	# stemmed_filtered_result = [ps.stem(w) for w in filtered_result]

	return " ".join(filtered_result)


