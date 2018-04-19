import pandas as pd
import pickle
import scipy.sparse
import numpy as np
import sklearn.preprocessing

with open("job_ids_list.txt", "rb") as fp:   # Unpickling
	job_ids = pickle.load(fp)

tfidf_matrix = scipy.sparse.load_npz('tfidf_matrix.npz') # load the sparse matrix

def get_job_profile(job_id):
	idx = job_ids.index(job_id)
	job_profile = tfidf_matrix[idx:idx+1]
	return job_profile # shape is (1, 3847)

def get_job_profiles(job_ids):
	job_profiles_list = [get_job_profile(job_id) for job_id in job_ids]
	job_profiles = scipy.sparse.vstack(job_profiles_list) # Stack sparse matrices vertically (row wise)
	return job_profiles

def build_user_profile(user_id, apps_df):
	apps_df = apps_df[apps_df['UserID'] == user_id]
	print(apps_df)
	user_job_profiles = get_job_profiles(apps_df['JobID'])
	# average of job profiles by the interactions strength
	user_job_avg = np.sum(user_job_profiles, axis=0) / user_job_profiles.shape[0]
	user_profile_norm = sklearn.preprocessing.normalize(user_job_avg)
	return user_profile_norm