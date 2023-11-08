
import numpy as np
import pandas as pd
import scipy.stats as stat
seed = 1234  # 设置种子值
np.random.seed(seed)  # 声明随机数种子

# 输入：exp_df
# 参数：corr_th
# 输出：df_PTCC

def exp_fn(row=50, col=100, rate=0.2):
    # 使某些行的相关性较高
    exp = np.random.rand(row, col)
    high_num = round(row * rate)
    high_corr_rows = exp[:high_num]
    noise = np.random.normal(0, 0.1, (high_num, col))  # 添加高斯分布的随机噪声
    high_corr_rows = high_corr_rows + noise
    modified_exp = np.vstack((high_corr_rows, exp[high_num:]))
    row_sums = modified_exp.sum(axis=1)
    normalized_exp = modified_exp / row_sums[:, np.newaxis]  # 使每一行的和为1

    return normalized_exp

def ptcc_fn(exp_df, corr_th=0.7, top=5):
    # corr_th = 0.35

    # rownames = exp_df.index
    colnames = exp_df.columns

    correlation_matrix = exp_df.corr(method='pearson')
    upper_triangle = np.triu(correlation_matrix.values, k=1)
    indices = np.where(np.abs(upper_triangle) > corr_th)

    genes = colnames
    corr_abs = np.abs(correlation_matrix).values
    len_genes = indices[0].shape[0]
    genes_top = []
    for i, j in zip(indices[0], indices[1]):
        # print(f"Position ({i}, {j}) has correlation > 0.2")
        gene_add = np.add(corr_abs[i, 0:], corr_abs[j, 0:])  # i,j 行相加
        gene_add = gene_add.tolist()
        gene_add = list(zip(gene_add, genes))  # zip 返回元组列表

        # 同时删除 gene_add 列表中的第 i 和 j 个元组
        gene_add = [item for index, item in enumerate(gene_add) if index != i and index != j]

        gene_add_sorted = sorted(gene_add, key=lambda d: d[0], reverse=True)  # 排序，reverse = True 降序
        gene_top = [gene_add_sorted[k][1] for k in range(top)]  # 前5个基因
        genes_top.append(gene_top)

    # 判断长度是否对应
    if len(genes_top) != len_genes:
        raise ValueError("len(genes_top) != len_genes, error!")

    PTCC = []
    for n in range(len_genes):  # 遍历高相关边（基因对）
        # print(n)
        # indices[0][n]
        # indices[1][n]
        gene_x = correlation_matrix.index[indices[0][n]]
        gene_y = correlation_matrix.columns[indices[1][n]]
        # print(f"Position ({n + 1}): ({gene_x}, {gene_y})")
        ptcc_top = []  # 存储去除 top=5 个高度相关基因间接影响 的 偏相关值
        pvalue_top = []
        for gene_z in genes_top[n]:  # 遍历对应的 top5 基因
            # print(f'x,y,z: {gene_x},{gene_y},{gene_z}')
            xx = exp_df[gene_x].values  # 提取某列，并转换为数组
            yy = exp_df[gene_y].values
            zz = exp_df[gene_z].values
            # 线性回归函数：  https://blog.csdn.net/xiaoyw71/article/details/123755643
            xz = stat.linregress(zz, xx)  # 线性回归函数
            yz = stat.linregress(zz, yy)

            rxx = xx - (xz.slope * zz + xz.intercept)  # xz.slope 回归线斜率，xz.intercept 回归线截距
            ryy = yy - (yz.slope * zz + yz.intercept)  #
            ptcc_z, pvalue_z = stat.pearsonr(rxx, ryy)
            ptcc_top.append(ptcc_z)
            pvalue_top.append(pvalue_z)
        # print(ptcc_top)
        # print(pvalue_top)

        ptcc = sum(ptcc_top) / len(ptcc_top)
        pvalue = max(pvalue_top)
        record = {
            "gene_x": gene_x,
            "gene_y": gene_y,
            "ptcc": ptcc,
            "pvalue": pvalue
        }

        PTCC.append(record)

    # 结果为 df_PTCC
    df_PTCC = pd.DataFrame(PTCC, columns=["gene_x", "gene_y", "ptcc", "pvalue"])
    # df_PTCC.to_csv('df_PTCC.csv', index=False)

    return df_PTCC


# 调用主函数
if __name__ == "__main__":
    # import os
    # os.chdir("D:/BGI/25.Meta-Spider/Meta-Spider")

    # 生成虚拟数据
    exp = exp_fn(50, 100, 0.2)
    rownames = [f"sample_{i+1}" for i in range(exp.shape[0])]
    colnames = [f"gene_{i+1}" for i in range(exp.shape[1])]
    exp_df = pd.DataFrame(exp, columns=colnames, index=rownames)
    df_PTCC = ptcc_fn(exp_df, corr_th=0.35, top=5)
    df_PTCC.to_csv('PTCC_test.csv', index=False)

