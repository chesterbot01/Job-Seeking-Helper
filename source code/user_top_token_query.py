import pickle
import scipy.sparse
import pandas as pd

with open("users_profiles.data", "rb") as myFile:
	users_profiles = pickle.load(myFile)

#tfidf_matrix = scipy.sparse.load_npz('tfidf_matrix.npz')

with open("tfidf_feature_names_list.txt", "rb") as fp:   # Unpickling
	tfidf_feature_names = pickle.load(fp)

high_tokens = pd.DataFrame(sorted(zip(tfidf_feature_names, users_profiles[20360].flatten().tolist()), 
	key=lambda x: -x[1])[:100], columns=['token', 'relevance'])
print(high_tokens)
