
import numpy as np
import pandas as pd
import pingouin as pg
# 设置警告过滤器，将RuntimeWarning过滤掉
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


seed = 1234  # 设置种子值
np.random.seed(seed)  # 声明随机数种子

# 输入：exp_df
# 参数：corr_th
# 输出：df_CC

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

def corr_fn(exp_df, corr_th=0.7, top=5, method='pearson'): # method='spearman'
    # corr_th = 0.35
    # rownames = exp_df.index
    colnames = exp_df.columns

    # 计算相关性矩阵
    correlation_matrix = exp_df.corr(method=method)

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

    CC = []
    for n in range(len_genes):  # 遍历高相关边（基因对）
        # print(n)
        # indices[0][n]
        # indices[1][n]
        gene_x = correlation_matrix.index[indices[0][n]]
        gene_y = correlation_matrix.columns[indices[1][n]]
        # print(f"Position ({n + 1}): ({gene_x}, {gene_y})")

        # 在pg.partial_corr函数中
        # method='spearman' 用于计算偏相关系数
        # method='pearson' 用于计算相关系数,假设数据 符合正态分布假设

        # 使用errstate函数来禁用警告
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = pg.partial_corr(exp_df, x=gene_x, y=gene_y, covar=genes_top[n], method=method)
        # 恢复警告设置
        warnings.resetwarnings()

        r_float = corr['r'][0] #.astype(float)
        # print(r_float)
        p_val = corr['p-val'][0]
        # print(p_val)

        record = {
            "gene_x": gene_x,
            "gene_y": gene_y,
            "corr": r_float,
            "p_val": p_val
        }

        CC.append(record)

    # 结果为 df_CC
    df_CC = pd.DataFrame(CC, columns=["gene_x", "gene_y", "corr", "p_val"])
    # df_CC.to_csv('df_CC.csv', index=False)

    return df_CC


# 调用主函数
if __name__ == "__main__":
    # import os
    # os.chdir("D:/BGI/25.Meta-Spider/Meta-Spider")

    # 生成虚拟数据
    exp = exp_fn(50, 100, 0.2)
    rownames = [f"sample_{i+1}" for i in range(exp.shape[0])]
    colnames = [f"gene_{i+1}" for i in range(exp.shape[1])]
    exp_df = pd.DataFrame(exp, columns=colnames, index=rownames)
    df_CC = corr_fn(exp_df, corr_th=0.35, top=5, method='pearson')
    df_CC.to_csv('CC_test.csv', index=False)

