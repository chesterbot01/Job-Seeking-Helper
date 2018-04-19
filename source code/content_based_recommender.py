import pandas as pd
import scipy.sparse
import pickle
from sklearn.metrics.pairwise import cosine_similarity

with open("users_profiles.data", "rb") as myFile:
	users_profiles = pickle.load(myFile)

tfidf_matrix = scipy.sparse.load_npz('tfidf_matrix.npz')

with open("job_ids_list.txt", "rb") as fp:   # Unpickling
	job_ids = pickle.load(fp)

class ContentBasedRecommender:
	
	MODEL_NAME = 'Content-Based'

	def __init__(self, job_ids, jobs_df=None):
		self.job_ids = job_ids
		self.jobs_df = jobs_df

	def get_model_name(self):
		return self.MODEL_NAME

	def _get_similar_jobs_to_user_profile(self, user_id, topn=50000):
		# computes the cosine similarity between the user profile and all job profiles in the corpus
		cosine_similarities = cosine_similarity(users_profiles[user_id], tfidf_matrix) # shape (n_samples_X, n_samples_Y) = (1, 40000+)
		# gets the top similar items
		similar_indices = cosine_similarities.argsort().flatten()[-topn:]
		# sort the similar jobs by similarity
		similar_jobs = sorted([(job_ids[i], cosine_similarities[0,i]) for i in similar_indices], 
			key=lambda x: -x[1]) # sort the tuple; [0, i] -> only has one row.

		return similar_jobs

	def recommend_items(self, user_id, jobs_applied_ignore=[], topn=10, verbose=False):
		similar_jobs = self._get_similar_jobs_to_user_profile(user_id)
		similar_jobs_filtered = list(filter(lambda x: x[0] not in jobs_applied_ignore, similar_jobs))
		recommendations_df = pd.DataFrame(similar_jobs_filtered, columns=['JobID', 'recStrength']).head(topn)

		if verbose:
			if self.jobs_df is not None:
				recommendations_df = recommendations_df.merge(self.jobs_df, how = 'left', left_on = 'JobID', right_on = 'JobID')[['recStrength', 'JobID', 'Title', 'Description', 'Requirements', 'City', 'State']]

		return recommendations_df

# a simple test case
#jobs1_df = pd.read_csv('splitjobs/jobs1.tsv', delimiter="\t", na_filter=False, error_bad_lines=False)
#content_based_recommender_model = ContentBasedRecommender(job_ids, jobs1_df)
#print(content_based_recommender_model.recommend_items(8189, [406104, 758462, 472338, 278925, 79368], 100, True))

