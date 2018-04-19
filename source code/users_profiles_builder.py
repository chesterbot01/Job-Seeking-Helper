from user_profile import build_user_profile
import pandas as pd
import pickle

train_app_df = pd.read_csv('train_app_project.tsv', delimiter="\t")
target_users_df = pd.read_csv('final_test_users.tsv', delimiter="\t")
users_list = target_users_df['UserID'].tolist() # 1419 users in total

def build_users_profiles():
	users_profiles = {} # creates an empty dict
	for user_id in users_list:
		users_profiles[user_id] = build_user_profile(user_id, train_app_df)
	return users_profiles

users_profiles = build_users_profiles()
with open("users_profiles.data", "wb") as myFile:
	pickle.dump(users_profiles, myFile)