import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
from sklearn import preprocessing

warnings.filterwarnings("ignore")
import statsmodels.stats.api as sms
import econometrics.statistics_models.bootstrap as bts
from sklearn.utils import resample
import random
from sklearn.decomposition import PCA


def olsreg(x, y, p):
    # 滞后p阶的ols回归
    Y = y.shift(p)
    X = sm.add_constant(x)
    # 删除X、y的前p个值
    X = X[p:]  # 这里删除空值以及无用的滞后项
    Y = Y[p:]
    model = sm.OLS(Y, X).fit(cov_type='HAC', use_t=True,cov_kwds={'maxlags':1})
    # model = sm.OLS(Y, X).fit()
    return model


def logitreg(x, y, p):
    y = y.apply(lambda x: 1 if x > 0 else 0)
    # 滞后p阶的ols回归
    Y = y.shift(p)
    X = sm.add_constant(x)
    # 删除X、y的前p个值
    X = X[p:]  # 这里删除空值以及无用的滞后项
    Y = Y[p:]
    model = sm.Logit(Y, X).fit()
    return model


def fwls(x, y, p):
    # 滞后p阶的ols回归
    Y = y.shift(p)
    X = sm.add_constant(x)
    # 删除X、y的前p个值
    X = X[p:]  # 这里删除空值以及无用的滞后项
    Y = Y[p:]
    # ols回归
    model_ols = sm.OLS(Y, X).fit()
    # log(e^2)对自变量回归
    log_e2 = np.log(model_ols.resid ** 2)
    model_ols2 = sm.OLS(log_e2, X).fit()
    # fwls回归
    wls_weight = list(1 / np.exp(model_ols2.fittedvalues))
    model = sm.WLS(Y, X, weights=wls_weight).fit()
    return model


# 单变量分组回归
def sigreg(df, banks, p):
    # 循环银行
    for bank in banks:
        print('银行：', bank, '回归结果：')
        # 将df中"股票代码"列为bank的提取出来，并且重置index
        bank_data = df[df['股票代码'] == bank].reset_index(drop=True)
        # 提取x和y
        x = bank_data[['ESG综合评价得分','环境评价得分','社会责任评价得分','治理评价得分','总市值','年龄']]
        y = bank_data['风险溢价']
        # 将x中总市值转换为对数
        x['总市值'] = np.log(x['总市值'])

        # 回归
        model = olsreg(x.loc[:, ['ESG综合评价得分','总市值','年龄']], y, p)
        # 计算异方差是否显著，返回值为[拉格朗日乘数值，拉值p值，f值，f_p值]
        # returns = sms.het_white(model.resid, model.model.exog)
        # print(returns)
        print(model.summary())
        print('=====================')


# 分组回归，不同银行回归
def classreg(df, banks, p):
    # 循环银行
    for bank in banks:
        # 将df中"股票代码"列为bank的提取出来，并且重置index
        bank_data = df[df['股票代码'] == bank].reset_index(drop=True)
        # 提取x和y
        x = bank_data[['ESG综合评价得分','环境评价得分','社会责任评价得分','治理评价得分','年龄']]
        y = bank_data['风险溢价']

        # 构造主成分
        # 降维到95%的方差，正好2个主成分
        pca = PCA(n_components=1)
        pca.fit(x)
        #k1_spss = pca.components_  # 成分得分系数矩阵
        #k1=pd.DataFrame(k1_spss.T)
        #k1.to_excel(f'k{bank}.xlsx',sheet_name='Sheet1')

        # 降维后的数据
        x_new = pca.transform(x)
        x = pd.DataFrame(x_new, columns=['F1'])

        # 将x中总市值转换为对数
        x['总市值'] = np.log(bank_data['总市值'])

        # bootstrap
        # t_down1, t_up1= bts.wild_bootstrap(x,y,5000,0.05)
        # print('bootstrap结果：', t_down1, t_up1)

        # 回归
        model = olsreg(x, y, p)
        print(model.summary())


# 面板数据向量自回归，控制个体效应
def panel_ar_reg(df, banks, p):
    # 将df中按照"股票代码"列分组
    mean = pd.DataFrame(df.groupby(['股票代码'])['ESG综合评价得分','环境评价得分','社会责任评价得分','治理评价得分','总市值','年龄'].mean())
    mean.columns = ['Mean_ESG', 'Mean_E', 'Mean_S', 'Mean_G', 'Mean_MV', 'Mean_Age']
    mean['股票代码'] = mean.index
    # 重置索引
    mean.reset_index(drop=True, inplace=True)
    data = pd.merge(df, mean, on='股票代码', how='left')
    # 通过减均值控制个体效应
    data['ESG综合评价得分'] = data['ESG综合评价得分'] - data['Mean_ESG']
    data['环境评价得分'] = data['环境评价得分'] - data['Mean_E']
    data['社会责任评价得分'] = data['社会责任评价得分'] - data['Mean_S']
    data['治理评价得分'] = data['治理评价得分'] - data['Mean_G']
    # data['总市值'] = data['总市值'] - data['Mean_MV']
    data['年龄'] = data['年龄'] - data['Mean_Age']
    # 取x和y
    x = data[['ESG综合评价得分','环境评价得分','社会责任评价得分','治理评价得分','总市值','年龄']]
    y = data['风险溢价']
    # 将x中总市值转换为对数
    x['总市值'] = np.log(x['总市值'])
    # 回归
    model = olsreg(x, y, p)
    print(model.summary())


# 面板数据向量自回归，控制个体效应
def panelreg(df, banks, p):
    # 取x和y
    x = df[['ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分','年龄']]
    y = df['风险溢价']

    # 构造主成分
    # 降维到95%的方差，正好2个主成分
    pca = PCA(n_components=0.95)
    pca.fit(x)
    # 输出特征值
    # print(pca.explained_variance_ratio_)
    # k1_spss = pca.components_  # 成分得分系数矩阵，主成分为原始数据减均值再矩阵乘得分系数矩阵
    # print(k1_spss)

    # 降维后的数据
    x_new = pca.transform(x)
    x = pd.DataFrame(x_new, columns=['F1', 'F2'])
    # 将df中的总市值添加到x
    x['总市值'] = np.log(df['总市值'])

    # 下面代码实现面板数据回归
    # 构造一个one-hot编码，区别不同银行
    dummies = pd.get_dummies(df['股票代码'])
    # 将one-hot编码与x合并
    x = pd.concat([x, dummies], axis=1)
    # 回归
    model = olsreg(x, y, p)
    print(model.summary())
    # 计算异方差是否显著，返回值为[拉格朗日乘数值，拉值p值，f值，f_p值]
    """returns = sms.het_white(model.resid, model.model.exog)
    print(returns)"""

    # 以下代码实现滞后，因为F滞后而总市值不滞后，因此不能直接使用变量p
    """for l in range(1,13):
        print('滞后期数：', l)
        x['股票代码'] = df['股票代码']
        newx = x.groupby(['股票代码'])[['F1', 'F2']].shift(-l)
        newx['总市值'] = x['总市值']
        newx = pd.concat([newx, dummies], axis=1)
        newx['risk_premium'] = y
        newx.dropna(inplace=True)
        model = olsreg(newx.iloc[:, :-1], newx.iloc[:, -1], p)
        print(model.summary())"""

    # 以下代码实现分类回归
    """x['股票代码'] = df['股票代码']
    x['风险溢价'] = y
    for bank in banks:
        # 将df中"股票代码"列为bank的提取出来，并且重置index
        bank_data = x[x['股票代码'] == bank].reset_index(drop=True)
        bank_data.drop(['股票代码'], axis=1, inplace=True)
        # 回归
        model = olsreg(bank_data.iloc[:, :-1], bank_data.iloc[:, -1], p)
        print(model.summary())
        returns = sms.het_white(model.resid, model.model.exog)
        print(returns)"""


# 单变量面板
def sigpanel(df, banks, p):
    # 取x和y
    x = df[['ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '总市值', '年龄']]
    y = df['风险溢价']
    # 将x中总市值转换为对数
    x['总市值'] = np.log(x['总市值'])
    x = x[['ESG综合评价得分','总市值', '年龄']]

    # 构造一个one-hot编码，区别不同银行
    dummies = pd.get_dummies(df['股票代码'])
    # 将one-hot编码与x合并
    x = pd.concat([x, dummies], axis=1)

    # 回归
    model = fwls(x, y, p)
    # 计算异方差是否显著，返回值为[拉格朗日乘数值，拉值p值，f值，f_p值]
    returns = sms.het_white(model.resid, model.model.exog)
    print(returns)
    print(model.summary())


if __name__ == '__main__':
    # 读取数据，分类数据
    df = pd.read_excel('bankdata/sortESGdata1.xlsx', sheet_name=0, header=0)
    # 不分类数据
    # df = pd.read_excel('bankdata/ESGdata1.xlsx', sheet_name=0, header=0)
    # 读出银行列表
    banks = df['股票代码'].drop_duplicates()
    # 单变量回归
    # sigreg(df, banks, 0)
    # 分组回归
    classreg(df, banks, 0)
    # 面板数据回归
    # panelreg(df, banks, 0)
    # 单变量面板
    # sigpanel(df, banks, 0)



