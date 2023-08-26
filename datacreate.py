import numpy as np
import pandas as pd
import warnings
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# 设置路径
filepace = r'C:\Users\11453\OneDrive - MSFT\Ab课程学业\大3文档\商业银行风险管理\课程论文\ESG数据'

"""def firststep():  # 把2018-04到2023-04的数据合并到一起
    # 读取银行名称文件
    banks = pd.read_excel(filepace + '\\' + 'banks.xlsx', sheet_name=0, header=0)
    # 读取数据
    for year in range(2018, 2024):
        df = pd.read_excel(filepace + '\\' + str(year) + '.xlsx', sheet_name=0, index_col=0, header=0)
        # 将df拼接到banks后面，按照万得代码拼接
        banks = pd.merge(banks, df, how='left', on='万得代码')
        del df
    # 保存数据
    banks.to_excel(filepace + '\\' + 'all.xlsx', sheet_name='Sheet1', index=False, header=True)


def secondstep():  # 读出不同类型的数据
    df = pd.read_excel(filepace + '\\' + 'all.xlsx', sheet_name=0, header=0)
    # 提取df前两列
    ESG = df.iloc[:, 0:2].copy()
    # 删除df前两列
    df.drop(df.columns[0:2], axis=1, inplace=True)
    # 读取ESG数据，每隔8列读取两列
    for i in range(6):
        n = 8*i
        temp = df.iloc[:, n:n+2]
        ESG = pd.concat([ESG, temp], axis=1)
    # 保存数据
    ESG.to_excel(filepace + '\\' + 'ESG.xlsx', sheet_name='Sheet1', index=False, header=True)"""


def otherdata():
    # 使用另外一份数据集提取想要的数据
    # 提取全部ESG数据中选择的银行部分，保存到新文件
    data = pd.read_excel(filepace + '\\' + '秩鼎ESG评分.xlsx', sheet_name=0, header=0,dtype='object')

    list_1 = ['股票代码', '公司全称', '评价日期', 'ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分']
    list_2 = ['股票代码', '公司全称', '评价日期', 'ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '环境管理评价得分', '节能政策评价得分', '环境排放评价得分',
              '气候变化评价得分', '人力资本评价得分', '健康与安全评价得分', '产品责任评价得分', '业务创新评价得分', '社会资本评价得分', '治理结构评价得分', '股东评价得分', '合规评价得分',
              '审计评价得分', '信息披露评价得分']
    # 取出得分列
    df = data.loc[:, list_1].copy()
    del data
    banks_file = pd.read_excel('bankdata/banks_all.xlsx', sheet_name=0, header=0,dtype='object')
    banks = banks_file['股票代码'].tolist()
    # 在df中提取想要的数据
    df = df[df['股票代码'].isin(banks)]

    # 填充df中空值
    df.fillna(0, inplace=True)
    # 修改列名，评价日期改为日期
    df.rename(columns={'评价日期': '日期'}, inplace=True)
    # 股票代码列格式设置object
    df['股票代码'] = df['股票代码'].astype('object')
    # 保存数据
    df.to_excel('秩鼎ESG评分all_1.xlsx', sheet_name='Sheet1', index=False, header=True)


# 以下函数一直到main里面运行，不单独运行
def banksdata(df):
    # 生成用于回归的数据，包括解释变量和被解释变量

    df['日期'] = pd.to_datetime(df['日期']).dt.to_period('D')
    df['日期'] = df['日期'].apply(lambda x: x.strftime('%Y-%m-%d'))  # datetime格式转成str
    # 读取风险溢价
    yiled = pd.read_excel('riskpre.xlsx', sheet_name=0, header=0)

    # 提取df中“股票代码”列，并转换为List
    banks = df['股票代码']
    # 清除重复项
    banks = banks.drop_duplicates()
    # 生成空dataframe保存数据
    df_save = pd.DataFrame()
    # 循环banks，提取每个银行的数据
    for bank in banks:
        # 提取每个银行的数据
        bank_data = df[df['股票代码'] == bank]
        # 提取yiled中的日期列和bank列
        yiled_data = yiled.loc[:, ['日期', bank]]
        # 将yiled_data的bank列重命名为风险溢价
        yiled_data.rename(columns={bank: '风险溢价'}, inplace=True)
        # 将收益率与市值与银行和时间匹配
        bank_data = pd.merge(bank_data, yiled_data, how='left', on=['日期'])
        # 将bank_data拼接到df_save后面
        df_save = pd.concat([df_save, bank_data], axis=0)
        # 删除bank_data
        del bank_data
    # 保存数据
    # df_save.to_excel('ESGdata2.xlsx', sheet_name='Sheet1', index=False, header=True)
    return df_save


# 将市值数据添加进数据集
def marketvalue(df):
    # 读取ESG数据
    # df = pd.read_excel('ESGdata1.xlsx', sheet_name=0, header=0, dtype=object)
    # 读取银行列表
    banks = pd.read_excel('banks.xlsx', sheet_name=0, header=0, dtype=object)['股票代码']
    # 生成空dataframe保存数据
    df_save = pd.DataFrame()
    # 循环
    for bank in banks:
        # 提取每个银行的数据
        bank_data = df[df['股票代码'] == bank]
        # 提取每个银行的市值
        marketvalue = pd.read_excel(f'marketvalues/{bank}.xlsx', sheet_name=0, header=0, dtype=object)
        # 处理marketvalue的日期格式
        marketvalue['date'] = pd.to_datetime(marketvalue['date']).dt.to_period('D')
        marketvalue['date'] = marketvalue['date'].apply(lambda x: x.strftime('%Y-%m-%d'))  # datetime格式转成str
        # 重命名marketvalue的date列名
        marketvalue.rename(columns={'date': '日期'}, inplace=True)
        # 将marketvalue与bank_data的日期匹配
        bank_data = pd.merge(bank_data, marketvalue, how='left', on=['日期'])
        # 将bank_data拼接到df后面
        df_save = pd.concat([df_save, bank_data], axis=0)
        # 删除bank_data
        del bank_data
    # 重命名df_save的values列名
    df_save.rename(columns={'value': '总市值'}, inplace=True)
    # 保存数据
    # df_save.to_excel('ESGdata1.1.xlsx', sheet_name='Sheet1', index=False, header=True)
    return df_save


# 将银行年龄添加进数据集
def bankage(df):
    # 读取ESG数据
    # df = pd.read_excel('ESGdata1.1.xlsx', sheet_name=0, header=0, dtype=object)
    # 读取年龄文件
    age = pd.read_excel('银行成立日期.xlsx', sheet_name=0, header=0, dtype=object)
    # 读取银行列表
    banks = pd.read_excel('banks.xlsx', sheet_name=0, header=0, dtype=object)['股票代码']
    # 生成空dataframe保存数据
    df_save = pd.DataFrame()
    # 循环
    for bank in banks:
        # 提取每个银行的ESG数据，重置索引
        bank_data = df[df['股票代码'] == bank].reset_index(drop=True)
        read_data = age[age['股票代码'] == bank]['成立日期']
        # 将read_data转换为和bank_data相同行数，列数为1的dataframe
        date_data = pd.DataFrame(np.tile(read_data, (bank_data.shape[0], 1)), columns=['成立日期'])
        # 将bank_data的日期和date_data的日期转换为datetime格式
        bank_data['日期'] = pd.to_datetime(bank_data['日期'])
        date_data['成立日期'] = pd.to_datetime(date_data['成立日期'])

        # 计算bank_data的日期与date_data的日期的差值
        bank_data['年龄'] = bank_data['日期'] - date_data['成立日期']
        # 将年龄转换为年，保留2位小数
        bank_data['年龄'] = bank_data['年龄'].map(lambda x: x.days / 365).round(2)
        # 将bank_data拼接到df_save后面
        df_save = pd.concat([df_save, bank_data], axis=0)
    df_save['日期'] = df_save['日期'].apply(lambda x: x.strftime('%Y-%m-%d'))  # datetime格式转成str
    # 保存数据
    # df_save.to_excel('ESGdata1.2.xlsx', sheet_name='Sheet1', index=False, header=True)
    return df_save


def main():
    # 读取ESG数据
    df = pd.read_excel('秩鼎ESG评分2.xlsx', sheet_name=0, header=0, dtype=object)
    esg = banksdata(df)
    esgm = marketvalue(esg)
    esgma = bankage(esgm)
    # 保存
    esgma.to_excel('ESGdata2.xlsx', sheet_name='Sheet1', index=False, header=True)




# otherdata()
main()

