from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import scipy.sparse
import pickle

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus
vectorizer = TfidfVectorizer(
	analyzer='word', 
	ngram_range=(1, 2), # 'machine learning' is a bigram. Single words sometimes can be confused.
	min_df=0.009, # I do not want to lose almost any rare word, except some spelling mistakes
	max_df=0.4, # can remove words such as 'skills',
	max_features=5000)

#feature_dic_df = pd.read_csv('feature_dic_project.tsv', dtype={'JobID':int}, delimiter="\t", error_bad_lines=False)

# the following line solve out-of-memory problems perfectly. Also see: https://stackoverflow.com/questions/40515194/memory-error-while-loading-csv-file
feature_dic_df = pd.concat((chunk for chunk in pd.read_csv('feature_dic_project.tsv', delimiter="\t", chunksize=10**4)),
               ignore_index=True)

#feature_dic_df = feature_dic_df.drop(feature_dic_df.columns[0], axis=1) # drop the first unnamed column


#feature_dic_df = feature_dic_df[:60000] # 60000 is the maximum cap

#feature_dic_df.to_csv('feature_dic_5.tsv', sep='\t')

# def make_corpus(doc_files):
# 	for doc in doc_files:
# 		df = pd.read_csv(doc, delimiter="\t")
# 		yield ' '.join(df['Feature'].values)

# file_list = ['feature_dic_0.tsv', 'feature_dic_1.tsv', 'feature_dic_2.tsv', 'feature_dic_3.tsv', 
# 'feature_dic_4.tsv', 'feature_dic_5.tsv']
# corpus = make_corpus(file_list)

job_ids = feature_dic_df['JobID'].tolist()

# with open("job_ids_list.txt", "wb") as fp:   #Pickling
# 	pickle.dump(job_ids, fp)

# convert corpus to tfidf matrix
tfidf_matrix = vectorizer.fit_transform(feature_dic_df['Feature'])
#tfidf_matrix = vectorizer.fit_transform(corpus)

#scipy.sparse.save_npz('tfidf_matrix.npz', tfidf_matrix) # Store sparse matrix to disk

# sparse_matrix = scipy.sparse.load_npz('tfidf_matrix.npz') # load the sparse matrix

tfidf_feature_names = vectorizer.get_feature_names()
#print(tfidf_feature_names) # 3847 <class 'list'>

#with open("tfidf_feature_names_list.txt", "wb") as fp:   #Pickling
	#pickle.dump(tfidf_feature_names, fp)

#with open("tfidf_feature_names_list.txt", "rb") as fp:   # Unpickling
	#tfidf_feature_names = pickle.load(fp)

print(type(tfidf_matrix)) # (46739, 3847) <class 'scipy.sparse.csr.csr_matrix'>