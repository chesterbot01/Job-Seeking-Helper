import pandas as pd
from datetime import datetime

train_end_time = datetime(2012, 4, 10, 0, 0, 0)

target_jobs_df = app_df = pd.read_csv('jobs_2app_project.tsv', delimiter="\t")

app_df = pd.read_csv('apps.tsv', delimiter="\t")
app_df = app_df[app_df['WindowID'] == 1]
app_df = app_df.drop('WindowID', axis=1)

app_df = app_df[app_df['JobID'].isin(target_jobs_df['JobID'])]

app_df['ApplicationDate'] = pd.to_datetime(app_df['ApplicationDate'])

#app_df.to_csv('jobs2app_app_project.tsv', sep='\t')


# in the training time, select the test users who have applied to more than 5 high quality jobs.
target_users_df = pd.read_csv('target_users_project.tsv', delimiter="\t")
app_df = app_df[app_df['ApplicationDate'] < train_end_time]
app_df = app_df[app_df['UserID'].isin(target_users_df['UserID'])]
#print(app_df.shape)
users_app_count_df = app_df.groupby(['UserID', 'JobID']).size().groupby('UserID').size()
final_test_users_df = users_app_count_df[users_app_count_df >= 5].reset_index()
#final_test_users_df.to_csv('final_test_users.tsv', sep='\t')

#now let us start to select the test dataset
app_df = pd.read_csv('jobs2app_app_project.tsv', delimiter="\t")
app_df['ApplicationDate'] = pd.to_datetime(app_df['ApplicationDate'])
app_df = app_df[app_df['ApplicationDate'] > train_end_time]
app_df = app_df[app_df['UserID'].isin(final_test_users_df['UserID'])]
#app_df.to_csv('test_app_project.tsv', sep='\t')
print(app_df.shape)

#now let us start to select the train dataset to build user profiles
app_df = pd.read_csv('jobs2app_app_project.tsv', delimiter="\t")
app_df['ApplicationDate'] = pd.to_datetime(app_df['ApplicationDate'])
app_df = app_df[app_df['ApplicationDate'] < train_end_time]
app_df = app_df[app_df['UserID'].isin(final_test_users_df['UserID'])]
#app_df.to_csv('train_app_project.tsv', sep='\t')
print(app_df.shape)