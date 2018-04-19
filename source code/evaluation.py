import pandas as pd
from content_based_recommender import ContentBasedRecommender
import pickle
import random

full_apps_df = pd.read_csv('jobs2app_app_project.tsv', delimiter="\t")
apps_train_df = pd.read_csv('train_app_project.tsv', delimiter="\t")
apps_test_df = pd.read_csv('test_app_project.tsv', delimiter="\t")
final_test_users_df = pd.read_csv('final_test_users.tsv', delimiter="\t")
users_in_apps_test = pd.unique(apps_test_df['UserID'])
jobs_2app_df = pd.read_csv('jobs_2app_project.tsv', delimiter="\t")

jobs1_df = pd.read_csv('splitjobs/jobs1.tsv', delimiter="\t", na_filter=False, error_bad_lines=False)

with open("job_ids_list.txt", "rb") as fp:   # Unpickling
	job_ids = pickle.load(fp)

# can be used for retrieving training or test data
def get_jobs_applied(user_id, apps_df):
	applied_jobs = apps_df[apps_df['UserID'] == user_id]['JobID']
	return set(applied_jobs if type(applied_jobs) == pd.Series else [applied_jobs])

TOP_NON_APPLIED_JOBS_COUNT = 100

class ModelEvaluator:

	def get_not_applied_jobs_samples(self, user_id, sample_size, seed=23):
		applied_jobs = get_jobs_applied(user_id, full_apps_df)
		all_jobs = set(jobs_2app_df['JobID'])
		non_applied_jobs = all_jobs - applied_jobs

		random.seed(seed)
		non_applied_jobs_sample = random.sample(non_applied_jobs, sample_size)
		#print('aaaaaaaaaaaaaaaaa')
		#print(len(non_applied_jobs_sample))  no bug for this method
		return set(non_applied_jobs_sample)

	def _hit_top_rank(self, job_id, recommended_jobs):
		for idx, value in enumerate(recommended_jobs): # a list of (seq_num, element)
			if value == job_id:
				return idx
		return -1

	def evaluate_model_for_user(self, model, user_id):
		# get the jobs in test set
		test_applied_jobs = apps_test_df[apps_test_df['UserID'] == user_id]
		test_applied_jobs = test_applied_jobs[test_applied_jobs['JobID'].isin(jobs_2app_df['JobID'])]
		if type(test_applied_jobs['JobID']) == pd.Series:
			user_applied_jobs_testset = set(test_applied_jobs['JobID'])
		else:
			user_applied_jobs_testset = set([int(test_applied_jobs['JobID'])])
		applied_jobs_count_testset = len(user_applied_jobs_testset)

		# get a ranked recommendation list from a model for a given user
		user_recs_df = model.recommend_items(user_id, 
			jobs_applied_ignore=get_jobs_applied(user_id, apps_train_df), topn=50000)
		# print('###########')
		# print(user_recs_df) # no bug here

		hits_at_3_count = 0
		hits_at_5_count = 0
		hits_at_10_count = 0
		hits_at_15_count = 0
		hits_at_20_count = 0
		hits_at_25_count = 0

		for job_id in user_applied_jobs_testset:
			# get a random list of non-applied jobs for the user
			non_applied_jobs_sample = self.get_not_applied_jobs_samples(user_id, 
				sample_size=TOP_NON_APPLIED_JOBS_COUNT, seed=job_id%(2**32))

			# combine the current applied job with the above 100 random non-applied jobs
			jobs_to_filter_recs = non_applied_jobs_sample.union(set([job_id]))
			# filter only recommendations that are either the applied job or from a random sample of 100 non-applied jobs
			valid_recs_df = user_recs_df[user_recs_df['JobID'].isin(jobs_to_filter_recs)]
			valid_recs = valid_recs_df['JobID'].values
			#print('cccccccccccccccc') # bug!
			#print(valid_recs)
			# verify if the current applied job is among the Top-N recommended jobs
			index = self._hit_top_rank(job_id, valid_recs)
			if index == -1:
				print('*******************')
				print(user_id)
				print(job_id)
				print('$$$$$$$$$$$$$$$$$$$')
			if index >= 0 and index < 3:
				hits_at_3_count += 1
			if index >= 0 and index < 5:
				hits_at_5_count += 1
			if index >= 0 and index < 10:
				hits_at_10_count += 1
			if index >= 0 and index < 15:
				hits_at_15_count += 1
			if index >= 0 and index < 20:
				hits_at_20_count += 1
			if index >= 0 and index < 25:
				hits_at_25_count += 1

		# recall is the rate of the applied jobs that are ranked among the Top-N recommended items, 
		# when mixed with a set of non-relevant items
		recall_at_3 = hits_at_3_count / float(applied_jobs_count_testset)
		recall_at_5 = hits_at_5_count / float(applied_jobs_count_testset)
		recall_at_10 = hits_at_10_count / float(applied_jobs_count_testset)
		recall_at_15 = hits_at_15_count / float(applied_jobs_count_testset)
		recall_at_20 = hits_at_20_count / float(applied_jobs_count_testset)
		recall_at_25 = hits_at_25_count / float(applied_jobs_count_testset)

		user_metrics = {
			'hits@3_count':hits_at_3_count,
			'hits@5_count':hits_at_5_count, 
			'hits@10_count':hits_at_10_count, 
			'hits@15_count':hits_at_15_count,
			'hits@20_count':hits_at_20_count,
			'hits@25_count':hits_at_25_count,
			'applied_count': applied_jobs_count_testset, 
			'recall@3': recall_at_3, 
			'recall@5': recall_at_5, 
			'recall@10': recall_at_10,
			'recall@15': recall_at_15,
			'recall@20': recall_at_20,
			'recall@25': recall_at_25}
		print(user_metrics)
		return user_metrics

	def evaluate_model(self, model):
		users_metrics = []
		#for idx, user_id in enumerate(final_test_users_df['UserID']): 
		for idx, user_id in enumerate(users_in_apps_test):
		# at first we used final_test_users_df, but it turned out even some users made all applications 
		# before the training time is over were actually categorized wrongly to the type of 'test users', 
		# thus we decide to use the users in the 'test_app_project.tsv' as the test users
			user_metrics = self.evaluate_model_for_user(model, user_id)
			user_metrics['_user_id'] = user_id
			users_metrics.append(user_metrics)

		print('%d users processed' % (idx+1))
		detailed_results_df = pd.DataFrame(users_metrics).sort_values('applied_count', ascending=False)
		global_recall_at_3 = detailed_results_df['hits@3_count'].sum() / float(detailed_results_df['applied_count'].sum())
		global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['applied_count'].sum())
		global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['applied_count'].sum())
		global_recall_at_15 = detailed_results_df['hits@15_count'].sum() / float(detailed_results_df['applied_count'].sum())
		global_recall_at_20 = detailed_results_df['hits@20_count'].sum() / float(detailed_results_df['applied_count'].sum())
		global_recall_at_25 = detailed_results_df['hits@25_count'].sum() / float(detailed_results_df['applied_count'].sum())

		global_metrics = {'modelName': model.get_model_name(), 'recall@3': global_recall_at_3, 'recall@5': global_recall_at_5, 'recall@10': global_recall_at_10, 'recall@15': global_recall_at_15, 'recall@20': global_recall_at_20, 'recall@25': global_recall_at_25}
		return global_metrics, detailed_results_df

model_evaluator = ModelEvaluator()

content_based_recommender_model = ContentBasedRecommender(job_ids, jobs1_df)

print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.to_csv('cb_detailed_results.tsv', sep='\t')