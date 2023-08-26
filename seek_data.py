import gc
import akshare as ak
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
"""
整个代码主要收集收益率数据，处理ESG和将收益率数据与ESG数据合并到datacreate.py中
"""
# 设置路径
filepace = r'C:\Users\11453\OneDrive - MSFT\Ab课程学业\大3文档\商业银行风险管理\课程论文\ESG数据'


"""该函数用于获取所有股票信息，设置开始与截止时间，主要选择复权方式，adjust表示复权方式，可选qfq:前复权，hfq:后复权，为空表示不复权"""
def hist_data(star_date, end_date, adjust):
    # 获取中国股市所有股票代码，返回列表
    stock_symbol = pd.read_excel('bankdata/banks_all.xlsx', dtype=object, sheet_name=0)['股票代码'].values  # 读取银行股票代码

    # 创建一个dataframe存储data，文件里面的日期和设置的start_date和end_date提前对应好
    df = pd.read_excel('date.xlsx',sheet_name=0)
    df['日期'] = pd.to_datetime(df['日期']).dt.to_period('D')
    df['日期'] = df['日期'].apply(lambda x: x.strftime('%Y-%m-%d'))  # datetime格式转成str
    # 创建空dataframe
    data = pd.DataFrame()


    for i in stock_symbol:
        # 返回股票代码为i的股票的月度数据，返回数据包括
        # 日期	object	交易日
        # 开盘	float64	开盘价
        # 收盘	float64	收盘价
        # 最高	float64	最高价
        # 最低	float64	最低价
        # 成交量	int64	注意单位: 手
        # 成交额	float64	注意单位: 元
        # 振幅	float64	注意单位: %
        # 涨跌幅	float64	注意单位: %
        # 涨跌额	float64	注意单位: 元
        # 换手率	float64	注意单位: %

        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=i, period="monthly", start_date=star_date, end_date=end_date,
                                                adjust=adjust)  # 注意为后复权，与平常看到的价格差异较大
        # 用来判断返回值是否为空，主要是去除在历史区间不存在的股票
        if stock_zh_a_hist_df.empty:
            continue
        stock_monthly_data = stock_zh_a_hist_df[['日期', '收盘']].copy()  # 这里加copy是使stock_monthly_data拥有数据，原本切片只是索引，用于重命名
        del stock_zh_a_hist_df
        gc.collect()  # 垃圾回收
        stock_monthly_data.rename(columns={'收盘': i}, inplace=True)
        # 将stock与data拼接
        data = pd.concat([data, stock_monthly_data],axis=1)  # 直接拼接

        # df = pd.merge(df, stock_monthly_data, how='left')  # 将每个数据拼接，采用左外连接，保证数据长度

    data.to_excel('data.xlsx', sheet_name='Sheet1', index=False)


# 计算收益率
def cal_return():
    df = pd.read_excel('stocksdata.xlsx', sheet_name=0, header=0)
    date = df['日期'].copy()
    df = df.drop(['日期'], axis=1)
    col = df.columns

    for i in col:
        df[i] = np.log(df[i]) - np.log(df[i].shift(-1))  # 对数收益率，由于数据为从新到旧，因此是shift(-1)：向上移一行

    df_tosave = pd.concat([date, df], axis=1)
    df_tosave.fillna(df_tosave.median(), inplace=True)
    df_tosave.to_excel('yield.xlsx', sheet_name='Sheet1', index=False)


# 计算风险溢价
def cal_riskpre():
    df_return = pd.read_excel('yield.xlsx', sheet_name=0, header=0)
    df_riskfree = pd.read_excel('yield.xlsx', sheet_name=1, header=0)   # 一年期国债
    # 提取日期列，并从df_return中删除
    date = df_return['日期'].copy()
    df_return = df_return.drop(['日期'], axis=1)
    # 循环df_return每一列，减去df_riskfree的值
    col = df_return.columns
    for i in col:
        df_return[i] = df_return[i] - df_riskfree['一年期国债']/12
    df_tosave = pd.concat([date, df_return], axis=1)
    # 保存数据
    df_tosave.to_excel('riskpre.xlsx', sheet_name='Sheet1', index=False)


# hist_data('20161201', '20170101', 'hfq')
# cal_return()
cal_riskpre()