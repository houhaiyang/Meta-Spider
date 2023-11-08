

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from metaspider.ptcc_fn import exp_fn
seed = 1234  # 设置种子值
np.random.seed(seed)  # 声明随机数种子

from scipy.spatial import distance
def remove_outliers_euclidean(data, quantile=25):
    # 计算每个样本与其他样本之间的欧式距离
    distances = distance.cdist(data, data, 'euclidean')
    # 计算每个样本与其他样本的平均距离
    mean_distances = np.mean(distances, axis=1)
    # 计算第三四分位数（Q3）
    q3 = np.percentile(mean_distances, 100-quantile)
    # 过滤保留位于下25%分位数和上75%分位数之间的样本
    filtered_df = data[mean_distances <= q3]

    return filtered_df

def Outlier_Aware_Clustering(data_all, df_in, N_cluster = 50, th_occ_sample = -1, quantile=-1, fill_min = 0.00001):
    # data = exp_df
    # 数据质控 -------------------------------------------
    exp_df_copy = df_in.copy()

    # 删除出现率小于 th_occ_sample=50% 的样本
    # th_occ_sample = 0.70
    if (th_occ_sample != -1):
        print("Quality control of samples !")
        exp_df_copy['sample_occurrence'] = (exp_df_copy != 0).mean(axis=1)
        exp_df_copy = exp_df_copy[exp_df_copy['sample_occurrence'] >= th_occ_sample]
        exp_df_copy = exp_df_copy.drop('sample_occurrence', axis=1)

        exp_df_copy_normalized = exp_df_copy.apply(lambda x: x / np.sum(x), axis=1)

        exp_df_copy = df_in.copy()  # 原始备份
        exp_df = exp_df_copy_normalized.copy()
    else:
        exp_df_copy = df_in.copy()
        exp_df = df_in.copy()

    # 删除离群样本 行为样本----------------------------
    # 将小于0.0001的值替换为0.0001
    # fill_min = 0.00001
    exp_df = exp_df.apply(lambda x: np.where(x < fill_min, fill_min, x))
    if (quantile != -1):
        print("Dealing with outlier samples !")
        exp_df = remove_outliers_euclidean(data=exp_df, quantile=quantile)
    # else:
    #     exp_df = exp_df

    # 聚成50类 行为样本---------------------
    # 创建KMeans对象，将n_clusters参数设置为50
    # N_cluster = 50
    N_middle = exp_df.shape[0]  # 不离散的样本数
    N_core = int(min(N_cluster, N_middle))
    if N_core < 10:
        raise ValueError("N_core should be greater than or equal to 10.")

    kmeans = KMeans(n_clusters=N_core)
    # 对exp_df进行聚类
    kmeans.fit(exp_df)
    # 获取每个样本的类别标签
    labels = kmeans.labels_

    # 初始化核心样本集
    core_samples = []
    # 遍历每个聚类，选择与聚类中心最近的样本作为核心样本
    for i in range(N_core):
        # 获取属于当前聚类的样本索引
        indices = [j for j, label in enumerate(labels) if label == i]
        # 计算当前聚类的样本到聚类中心的距离
        distances = kmeans.transform(exp_df.iloc[indices])
        # 找到距离聚类中心最近的样本的索引
        closest_index = np.argmin(np.sum(distances, axis=1))
        # 将最近的样本添加到核心样本集中
        core_samples.append(indices[closest_index])

    # 打印核心样本集
    # print(core_samples)

    rownames = exp_df.index
    # colnames = exp_df.columns
    coreset_names = rownames[core_samples]
    rownames_df = pd.DataFrame({'rownames': rownames})
    labels_df = pd.DataFrame({'labels': labels})
    sample_label = pd.concat([rownames_df, labels_df], axis=1)

    data = exp_df_copy.copy()
    coreset_names = coreset_names.tolist()

    # coreset_names长度50，是list，代表50个核心样本名
    # sample_label有314个样本，共两列：'rownames'和'labels'，一共50种label
    # 数据框 data 包含所有1000个样本，请基于核心样本对没有分类的样本（1000-259）进行分类，类别为距离最近的一个核心样本的label

    # 初始化所有样本的标签
    all_sample_label = pd.DataFrame({'rownames': data_all.index, 'labels': -1})
    # 使用 merge 函数将 sample_label 中的已知值添加到 all_sample_label 中，并覆盖原来的值
    all_sample_label = all_sample_label.merge(sample_label, on='rownames', how='left', suffixes=('', '_y'))
    all_sample_label['labels'] = all_sample_label['labels_y']
    all_sample_label = all_sample_label.drop(['labels_y'], axis=1)
    # 将 'labels' 列中的 NaN 值替换为 -1
    all_sample_label['labels'] = all_sample_label['labels'].fillna(-1)

    # 将核心样本名转换为索引
    core_indices = [data.index.get_loc(name) for name in coreset_names]
    # 获取核心样本的特征向量
    core_samples = data.iloc[core_indices]
    # 将之前聚类样本名转换为索引
    finish_indices = [data.index.get_loc(name) for name in sample_label["rownames"]]

    # 计算当前样本到所有核心样本的距离
    distances = cdist(data_all, core_samples)

    # 遍历每个未分类样本，为其分配最近的核心样本的标签
    for i in range(len(data_all)):
        if (all_sample_label['labels'][i] == -1):  # 只处理未分类样本
            # print(i)
            # 找到距离最近的核心样本的索引
            closest_index = np.argmin(distances[i])

            # 分配距离最近的核心样本的标签给当前样本
            l = all_sample_label['labels'][core_indices[closest_index]]
            all_sample_label.loc[i, 'labels'] = l

    allset_labels = all_sample_label.copy()

    return coreset_names, allset_labels, N_core


# 调用主函数
if __name__ == "__main__":
    # import os
    # os.chdir("D:/BGI/25.Meta-Spider/Meta-Spider")

    # 生成虚拟数据
    # 1000 个样本，300 个特征（基因、物种），高相关 30%
    exp = exp_fn(row=1000, col=300, rate=0.3)
    # 用50%的0值填充,重新归一化
    num_zeros = int(0.5 * exp.size)
    exp.ravel()[np.random.choice(exp.size, num_zeros, replace=False)] = 0
    row_sums = exp.sum(axis=1)
    exp = exp / row_sums[:, np.newaxis]

    rownames = [f"sample_{i + 1}" for i in range(exp.shape[0])]
    colnames = [f"gene_{i + 1}" for i in range(exp.shape[1])]
    exp_df = pd.DataFrame(exp, columns=colnames, index=rownames)

    # 数据质控 -------------------------------------------
    exp_df_copy = exp_df.copy()
    # 删除出现率小于 th_occ_feature=50% 的基因
    th_occ_feature = 0.5
    exp_df_copy = exp_df_copy.T
    exp_df_copy['gene_occurrence'] = (exp_df_copy != 0).mean(axis=1)
    exp_df_copy = exp_df_copy[exp_df_copy['gene_occurrence'] >= th_occ_feature]
    exp_df_copy = exp_df_copy.drop('gene_occurrence', axis=1)
    exp_df_copy = exp_df_copy.T
    exp_df = exp_df_copy.copy()

    coreset_names,allset_labels = Outlier_Aware_Clustering(df_in=exp_df,
                                                           N_cluster = 50,
                                                           th_occ_sample = 0.50,
                                                           quantile=25)

    # 打印核心样本名称
    # print(coreset_names)
    allset_labels.to_csv('allset_labels.csv', index=False)
    with open('coreset_names.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(coreset_names))



