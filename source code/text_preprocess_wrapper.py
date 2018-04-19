from text_preprocess import feature_generate
import pandas as pd

jobs1_df = pd.read_csv('splitjobs/jobs1.tsv', delimiter="\t", na_filter=False, error_bad_lines=False)
#export_feature_dic_df = jobs1_df['JobID'].copy()

result_df = pd.DataFrame(columns=['JobID', 'Feature'])
result_df['JobID'] = jobs1_df.loc[:, 'JobID']

result_df['Feature'] = jobs1_df.apply(lambda x: feature_generate(
	x['Title'], 
	x['Description'], 
	x['Requirements'], 
	x['City'],
    x['State']), axis=1)

result_df.to_csv('feature_dic_project.tsv', sep='\t')