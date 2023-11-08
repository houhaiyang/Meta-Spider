<div align=center><img width="684" height="504.5" src="docs/MSlogo2.png"/></div>

# Meta-Spider
- A powerful method for constructing sample-individual networks.
- Estimating sample-specific networks through outlier-aware clustering, exclusion of indirect interactions, and application of linear interpolation.

#### Current functions

- By using outlier-aware clustering, a core sample set is constructed and each sample is assigned a category label.
- Indirect interactions between “features” are excluded to retain only direct interactions.
- Linear interpolation is applied to estimate the sample-specific network.

-------------

## Description

This project is based on Python 3.7+ and developed using PyCharm on Windows 10+ (11).

- Author: Haiyang Hou
- Date: 2023-11-07
- Version: v0.9.3
- If you want to use this package, please indicate the source and tell me in "lssues". Free use.

-------------

## Installation
Requirements: python>=3.7, numpy, pandas, sklearn, joblib, seaborn, matplotlib

Install through local:
```commandline
pip install dist/metaspider-0.9.3-py3-none-any.whl
```
-------------

## Usage
```python
metaspider(data=exp_df, outdir='result', N_cluster=50, th_occ_feature=0.5, th_occ_sample=0.50, fill_min=0.00001, ptcc_corr_th=0.35, top=5, n_jobs=16, quantile=0.25)
```

#### Parameter interpretation

- `outdir`: The directory for outputting sample-specific networks.

- `N_cluster`: The number of clusters to be created.

- `th_occ_feature`: The threshold for feature filtering.

- `th_occ_sample`: The threshold for sample filtering.

- `fill_min`: The minimum value for data filling.

- `ptcc_corr_th`: The correlation threshold used in the second step of network construction.

- `top`: The number of reference genes with strong correlation.

- `n_jobs`: The number of parallel processing threads.

- `quantile`: The quantile used for partitioning discrete samples.


#### Input data
- exp_df:

  |       | node_1 | node_2 | ... | node_n |
  | :---: | :----: | :-------: | :-------: | :---: |
  |  sample_1   | 0.01  |    0.009    |    ...     | 0.02 |
  |  sample_2   | 0.0002  |   0.0003    |    ...      | 0.0004 |
  |  ...   | ...   |    ...     |    ...      | ...  |
  |  sample_m   | 0.003  |   0.004    |    ...      | 0.005 |


#### Example

Please refer to '_Script order of a single project.txt' for an example.

