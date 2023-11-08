
import numpy as np
import pandas as pd
from metaspider.corr_fn import corr_fn, exp_fn
seed = 1234  # 设置种子值
np.random.seed(seed)  # 声明随机数种子

def sample_weight(exp_df, bp=1.0, antizero=0.00000001):
    N_sample = exp_df.shape[0]
    value = np.corrcoef(exp_df) # sample corr
    # 每个sample 与其他50个sample相关性的均值
    value = (np.sum(value, axis=1) - 1) / (N_sample - 1)
    rmax, rmin = np.max(value), np.min(value)
    dif = np.maximum(rmax - rmin, antizero)
    value = np.maximum(value - rmin, antizero) / dif

    value = value * bp * N_sample  # patlen = 80，bp=0.1 平衡参数
    # s_weight = {name: val for name, val in zip(exp_df.index, value)}
    last_value = value[-1]

    return last_value

def difnet_fn(PTCC_a, PTCC_b, N_scale=1):
    PTCC_a = PTCC_a.rename(columns={'corr': 'corr_a', 'p_val': 'pvalue_a'})
    PTCC_b = PTCC_b.rename(columns={'corr': 'corr_b', 'p_val': 'pvalue_b'})

    # PTCC_merged = PTCC_a.merge(PTCC_b, on=['gene_x', 'gene_y'], how='outer').fillna(0)
    PTCC_merged = PTCC_a.merge(PTCC_b, on=['gene_x', 'gene_y'], how='inner').fillna(0) # 取交集，避免缺失的边最后计算时过大、失真

    # print(PTCC_merged.columns)
    # 生成一个新列 pvalue，其中的值由 pvalue_a 和 pvalue_b 的最大值生成
    PTCC_merged['pvalue'] = PTCC_merged.apply(lambda row: max(row['pvalue_a'], row['pvalue_b']), axis=1)
    # 生成一个新列 corr_dif，其中的值由 corr_b 减去 corr_a 生成
    PTCC_merged['corr_dif'] = PTCC_merged['corr_b'] - PTCC_merged['corr_a']

    PTCC_merged['spider_value'] = N_scale * PTCC_merged['corr_dif'] + PTCC_merged['corr_a']

    df_spider = PTCC_merged[['gene_x', 'gene_y', 'spider_value', 'pvalue']].copy()

    return df_spider


# 调用主函数
if __name__ == "__main__":
    # import os
    # os.chdir("D:/BGI/25.Meta-Spider/Meta-Spider")

    # 生成虚拟数据 PTCC_a PTCC_b
    # exp = np.random.rand(50, 100)
    exp = exp_fn(50, 100, 0.2)
    rownames = [f"sample_{i + 1}" for i in range(exp.shape[0])]
    colnames = [f"gene_{i + 1}" for i in range(exp.shape[1])]
    exp_df = pd.DataFrame(exp, columns=colnames, index=rownames)
    PTCC_a = corr_fn(exp_df, corr_th=0.35, top=5)

    row = np.random.rand(1, 100)  # 生成一行随机数
    row = row / np.sum(row)  # 将一行随机数归一化为和为1
    exp = np.vstack([exp, row])  # 添加和为1的行
    rownames = [f"sample_{i + 1}" for i in range(exp.shape[0])]
    colnames = [f"gene_{i + 1}" for i in range(exp.shape[1])]
    exp_df = pd.DataFrame(exp, columns=colnames, index=rownames)
    PTCC_b = corr_fn(exp_df, corr_th=0.35, top=5)

    N_scale = sample_weight(exp_df)

    df_spider = difnet_fn(PTCC_a, PTCC_b, N_scale=N_scale)
    df_spider.to_csv('spider_test.csv', index=False)



