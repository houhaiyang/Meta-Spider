


import os
import pandas as pd

datadir = 'data_spider'

print(datadir)

# Read all sample names.
allset_labels = pd.read_csv(os.path.join(datadir, '_allset_labels.csv'), index_col=None, header=0)
all_samplename = allset_labels['rownames'].tolist()

# Extract valid data.
list_allsample = []
for samplename in all_samplename:
    try:
        # print(samplename)
        path = os.path.join(datadir, f'{samplename}.csv')
        df_sample = pd.read_csv(path, index_col=None, header=0)
        # Filter rows with' pvalue' column value less than 0.05.
        df_sample = df_sample[df_sample['pvalue'] < 0.05]
        df_sample = df_sample.drop(columns='pvalue')
        df_sample = df_sample.rename(columns={'spider_value': f'{samplename}'})
        df_sample = df_sample.rename(columns={'gene_x': 'feature_x', 'gene_y': 'feature_y'})
        list_allsample.append(df_sample)
    except:
        print(f"sample {samplename} error !")

# Merge samples data frame.
merged_df = pd.merge(list_allsample[0], list_allsample[1], on=['feature_x', 'feature_y'], how='outer')
for i in range(2, len(list_allsample)):
    merged_df = pd.merge(merged_df, list_allsample[i], on=['feature_x', 'feature_y'], how='outer')

# Sort data.
df_2 = merged_df.iloc[:, 2:]
sorted_df_2 = df_2.count().sort_values(ascending=False)
df_2_sorted = df_2[sorted_df_2.index]
sorted_df = pd.concat([merged_df.iloc[:, :2], df_2_sorted], axis=1)

# Calculate the number of non-NaN values in each row.
row_nan_counts = sorted_df.count(axis=1)
# Sorts rows according to the number of non-NaN values.
sorted_df = sorted_df.iloc[row_nan_counts.sort_values(ascending=False).index]
df_net_allsample = sorted_df.reset_index(drop=True)
# Save the merged data.
df_net_allsample.to_csv(os.path.join(datadir, '_metaspider_allsample.csv'), index=False)
