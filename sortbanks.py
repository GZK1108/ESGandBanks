import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def classify():
    # 读取ESG数据
    df = pd.read_excel('bankdata/ESGdata1.xlsx', sheet_name=0, header=0)
    # 读取银行列表
    banks = pd.read_excel('bankdata/banks.xlsx', sheet_name=0, header=0)
    # 分类列表
    daxing = banks['大型银行']
    gufenzhi = banks['股份制']
    chengnong = banks['城农商行']
    # 按类别提取数据
    daxing_data = df[df['股票代码'].isin(daxing)].reset_index(drop=True)
    gufenzhi_data = df[df['股票代码'].isin(gufenzhi)].reset_index(drop=True)
    chengnong_data = df[df['股票代码'].isin(chengnong)].reset_index(drop=True)
    # 循环处理，取均值
    daxing_data = daxing_data.groupby('日期').mean().reset_index(drop=True)
    gufenzhi_data = gufenzhi_data.groupby('日期').mean().reset_index(drop=True)
    chengnong_data = chengnong_data.groupby('日期').mean().reset_index(drop=True)
    # 将三类数据的股票代码列分别置为1、2、3
    daxing_data['股票代码'] = 1
    gufenzhi_data['股票代码'] = 2
    chengnong_data['股票代码'] = 3
    # 将三类数据合并
    df = pd.concat([daxing_data, gufenzhi_data, chengnong_data], axis=0).reset_index(drop=True)
    # 保存数据
    df.to_excel('bankdata/sortESGdata1.xlsx', index=False)

# 排序
def sortdate():
    df = pd.read_excel('bankdata/sortESGdata1.xlsx', sheet_name=0, header=0, dtype=object)
    # df['日期'] = pd.to_datetime(df['日期'])
    # 按股票代码分组，并将每组按日期倒序
    df = df.sort_values(by=['股票代码', '日期'], ascending=[True, False])
    # 保存数据
    df.to_excel('bankdata/sortESGdata11.xlsx', index=False)

sortdate()