


import os
import numpy as np
import pandas as pd
from metaspider.metaspider import metaspider

# Select species with an occurrence rate greater than 5%
def filt_species(data_project, occ_th=0.05):
    count_gt_zero = (data_project > 0).sum()
    species_occurrence = count_gt_zero / len(data_project)
    filtered_species = species_occurrence[species_occurrence > occ_th]
    df_filtered = data_project[filtered_species.index]

    return df_filtered


set_dir = 'dataRaw/'
out_dir = 'data'
outdir_spider = 'data_spider'

# Read data.
data_abd = pd.read_csv(os.path.join(set_dir, 'data_abundance.csv'), index_col=0, header=0, sep=',')
data_phe = pd.read_csv(os.path.join(set_dir, 'data_phenotype.csv'), index_col=0, header=0, sep=',')

# Replace the 0 value in the data frame with NaN.
data_abd.replace(0, np.nan, inplace=True)
th_occ_feature = 0.05
df_filtered = filt_species(data_abd, occ_th=th_occ_feature)
# Handling infinity and missing values.
df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan)
df_filtered = df_filtered.fillna(0)

# Create a new data frame using intersection.
data_all = df_filtered

# Calculate the value rate of each column.
non_zero_rate = (data_all != 0).sum(axis=1) / data_all.shape[1]
# Filter rows with value ratio greater than 0.01.
where = non_zero_rate > 0.01
data_all = data_all.loc[where]
data_phe = data_phe.loc[where]

# Convert to relative abundance.
# Calculate the sum of each line.
sums = data_all.sum(axis=1)
# Divide each element by the sum of the corresponding rows.
data_all = data_all.div(sums, axis=0)
# sums = data_all.sum(axis=1)

# Save data.
data_all.to_csv(os.path.join(out_dir, 'data_all.csv'), index=True)
data_phe.to_csv(os.path.join(out_dir, 'data_phe.csv'), index=True)

# Run MetaSpider - For the baseline sample, select control  --------------
data_phe_control = data_phe[data_phe['group'] == 0]
where = data_all.index.isin(data_phe_control.index)
data_control = data_all.iloc[where]
data_control.to_csv(os.path.join(out_dir, 'data_control.csv'), index=True)
print(f'num_all:{data_all.shape[0]}, '
      f'num_ctrl:{data_control.shape[0]}, '
      f'num_crc:{data_all.shape[0] - data_control.shape[0]}')

metaspider(data_all=data_all,
           data_base=data_control,
           outdir=outdir_spider,
           N_cluster=20,
           th_occ_feature=th_occ_feature,
           th_occ_sample=0.05,
           corr_th=0.2,
           top=5,
           n_jobs=8,
           quantile=-1)
