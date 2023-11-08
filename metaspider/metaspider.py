import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import datetime
# from tqdm import tqdm
from metaspider.clustering_fn import Outlier_Aware_Clustering
from metaspider.corr_fn import corr_fn, exp_fn
from metaspider.difnet_fn import sample_weight,difnet_fn
from metaspider.clr_trans import clr_df

seed = 1234  # 设置种子值
np.random.seed(seed)  # 声明随机数种子


# 数据质控 -------------------------------------------
def quality_control(exp_df, th_occ_feature=0.5):
    exp_df_copy = exp_df.copy()
    # 删除出现率小于 th_occ_feature=50% 的基因
    # th_occ_feature = 0.5
    exp_df_copy = exp_df_copy.T
    exp_df_copy['gene_occurrence'] = (exp_df_copy != 0).mean(axis=1)
    exp_df_copy = exp_df_copy[exp_df_copy['gene_occurrence'] >= th_occ_feature]
    exp_df_copy = exp_df_copy.drop('gene_occurrence', axis=1)
    exp_df_copy = exp_df_copy.T
    exp_df = exp_df_copy.copy()
    if exp_df.shape[0] < 10:
        raise ValueError("After th_occ_feature, N_sample should be greater than or equal to 10.")
    else:
        return exp_df



# 主函数，串联三部分
def metaspider(data_all, data_base, outdir='result', N_cluster=50,
               th_occ_feature=-1, th_occ_sample=-1,
               corr_th=0.35, top=5, n_jobs=8, quantile=-1,
               method='pearson', fill_min=0.00001):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if (th_occ_feature != -1):
        print("Quality control of features !")
        data_all = quality_control(data_all, th_occ_feature=th_occ_feature)
        column_names = data_all.columns.tolist()
        # 根据列名从exp_df中提取列
        exp_df = data_base[column_names]
    else:
        # data_all = data_all
        exp_df = data_base

    metaspider_features = exp_df.columns.tolist()
    with open(os.path.join(outdir, '_metaspider_features.txt'), 'w') as file:
        for feature in metaspider_features:
            file.write(feature + '\n')
    # print(exp_df.shape)


    # 第一步，离散感知聚类

    # 填充 0 值 --------------
    data_all = data_all.apply(lambda x: np.where(x < fill_min, fill_min, x))
    exp_df = exp_df.apply(lambda x: np.where(x < fill_min, fill_min, x))


    coreset_names, allset_labels, N_core = Outlier_Aware_Clustering(data_all = data_all,
                                                                    df_in=exp_df,
                                                                    N_cluster=N_cluster,
                                                                    th_occ_sample=th_occ_sample,
                                                                    quantile=quantile,
                                                                    fill_min = 0.00001)

    allset_labels.to_csv(os.path.join(outdir, '_allset_labels.csv'), index=False)

    # 中心对数比(Centered Log-ratio, CLR)转化
    data_all = clr_df(data_all)
    exp_df = clr_df(exp_df)

    coreset = exp_df.loc[exp_df.index.isin(coreset_names)]
    coreset_labels = allset_labels.loc[allset_labels['rownames'].isin(coreset_names)]
    coreset_labels = coreset_labels.reset_index(drop=True)
    coreset_labels.to_csv(os.path.join(outdir, '_coreset_labels.csv'), index=False)

    N_all = data_all.shape[0]
    N_feature = coreset.shape[1]
    print(f'N_all,N_core,N_feature = {N_all},{N_core},{N_feature}')
    now = datetime.datetime.now()
    created_time = now.strftime('%Y-%m-%d %H:%M:%S')
    df_log = pd.DataFrame({'Variable': ['outdir', 'th_occ_feature', 'th_occ_sample',
                                        'corr_th', 'top', 'n_jobs', 'quantile',
                                        'N_all', 'N_feature', 'N_core', 'created_time'],
                           'Value': [outdir, th_occ_feature, th_occ_sample,
                                     corr_th, top, n_jobs, quantile, N_all, N_feature, N_core, created_time]})
    df_log.to_csv(os.path.join(outdir, '_log.csv'), index=False)


    # 第二步 遍历所有样本，依次构建 PTCC_a、PTCC_b
    def process(i, allset_labels, coreset_labels, coreset, corr_fn, data_all, method):
        try:
            label_i = allset_labels['labels'][i]
            # 去除 coreset 中的相同 label 的样本
            where_bool = coreset_labels['labels'] != label_i
            df_a = coreset[where_bool.tolist()]
            PTCC_a = corr_fn(df_a, corr_th=corr_th, top=top, method=method)

            # 所处理的单个样本
            row_data = data_all.iloc[i]
            df_b = df_a.append(row_data, ignore_index=True)
            PTCC_b = corr_fn(df_b, corr_th=corr_th, top=top, method=method)

            # 第三步 生成 spider net
            N_scale = sample_weight(df_b,bp=1.0)
            df_spider = difnet_fn(PTCC_a, PTCC_b, N_scale=N_scale)
            samplename = allset_labels['rownames'][i]
            outpath = os.path.join(outdir, f'{samplename}.csv')
            df_spider.to_csv(outpath, index=False)

        except:
            sample_i = allset_labels['rownames'][i]
            print(f"sample {i}, {sample_i} error !")

    print('Start multithreading')
    Parallel(n_jobs=n_jobs)(delayed(process)(i,
                                             allset_labels,
                                             coreset_labels,
                                             coreset,
                                             corr_fn,
                                             data_all,
                                             method) for i in range(N_all))
    print('End multithreading')


# 调用主函数
if __name__ == "__main__":
    # os.chdir("D:/BGI/25.Meta-Spider/Meta-Spider")

    # 生成虚拟数据 ----------------------------

    # 所有样本 ------
    # 1000 个样本，200 个特征（基因、物种），高相关 30%
    exp = exp_fn(row=1000, col=200, rate=0.3)
    # 用50%的0值填充,重新归一化
    num_zeros = int(0.5 * exp.size)
    exp.ravel()[np.random.choice(exp.size, num_zeros, replace=False)] = 0
    row_sums = exp.sum(axis=1)
    exp = exp / row_sums[:, np.newaxis]
    rownames = [f"sample_{i + 1}" for i in range(exp.shape[0])]
    colnames = [f"gene_{i + 1}" for i in range(exp.shape[1])]
    exp_all = pd.DataFrame(exp, columns=colnames, index=rownames)

    # 基线样本 ------
    # 300 个样本，200 个特征（基因、物种），高相关 30%
    exp = exp_fn(row=300, col=200, rate=0.3)
    # 用50%的0值填充,重新归一化
    num_zeros = int(0.5 * exp.size)
    exp.ravel()[np.random.choice(exp.size, num_zeros, replace=False)] = 0
    row_sums = exp.sum(axis=1)
    exp = exp / row_sums[:, np.newaxis]
    rownames = [f"sample_{i + 1}" for i in range(exp.shape[0])]
    colnames = [f"gene_{i + 1}" for i in range(exp.shape[1])]
    exp_base = pd.DataFrame(exp, columns=colnames, index=rownames)


    # os.chdir("D:/BGI/25.Meta-Spider/Meta-Spider")
    metaspider(data_all=exp_all, data_base=exp_base, outdir='result',
               N_cluster=50, th_occ_feature=0.4, th_occ_sample=0.50,
               corr_th=0.1, top=5, n_jobs=8,
               quantile=25, method='pearson')

    # data_all = exp_all
    # data_base = exp_base
    # outdir = 'result'
    # N_cluster = 50
    # th_occ_feature = 0.4
    # th_occ_sample = 0.50
    # corr_th = 0.1
    # top = 5
    # n_jobs = 8
    # quantile = 25
    # method = 'pearson'

