import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
import scipy as sp

# 计算均方误差
def MSE(r_real, r_hat):
    denominator = len(r_real)
    numerator_res = (r_real - r_hat) ** 2
    numerator = np.sum(numerator_res)
    mse = numerator / denominator
    return mse


def R2OOS(r_real, r_hat, r_bar):
    denominator_res = (r_real - r_bar) ** 2
    denominator = np.sum(denominator_res)
    numerator_res = (r_real - r_hat) ** 2
    numerator = np.sum(numerator_res)
    r2oos = 1 - numerator / denominator
    return r2oos


def newx(x_data, x_lookback):
    """
    :param x_data: 转换的数量，格式要求为series
    :param x_lookback:
    :return:
    """
    # 该函数用于根据look-back区间调整x
    x_max = np.max(x_lookback)
    x_min = np.min(x_lookback)
    # 设置x的新区间，若x中数大于x_max或小于x_min,则x=x，否则x=0
    new_x = x_data.apply(lambda x: x if x > x_max or x < x_min else 0)
    return new_x


def fwls(x, y):
    # ols回归
    model_ols = sm.OLS(y, x).fit(cov_type='HAC', use_t=True, cov_kwds={'maxlags': 1})
    # log(e^2)对自变量回归
    log_e2 = np.log(model_ols.resid ** 2)
    model_ols2 = sm.OLS(log_e2, x).fit(cov_type='HAC', use_t=True, cov_kwds={'maxlags': 1})
    # fwls回归
    wls_weight = list(1 / np.exp(model_ols2.fittedvalues))
    model = sm.WLS(y, x, weights=wls_weight).fit(cov_type='HAC', use_t=True, cov_kwds={'maxlags': 1})
    return model


# 混淆矩阵
def confusion_matrix(y_test, y_pred):
    # 将y_test和y_pred转换成array
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    # 将y_test和y_pred转换为0和1
    y_test = np.where(y_test > 0, 1, -1)
    y_pred = np.where(y_pred > 0, 1, -1)  # logit的阈值为0.5，ols阈值为0
    # 计算混淆矩阵
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    return cm



if __name__ == "__main__":
    df = pd.read_excel('bankdata/sortESGdata1.xlsx', sheet_name=0, header=0)
    banks = df['股票代码'].drop_duplicates()

    Table9_ar = pd.DataFrame(
        columns=['ESG_return', 'TSH_return', 'ESG_SR', 'TSH_SR', 'Return_Difference', 'SR_Difference'], index=np.arange(1, 4))

    for bank in banks:
        bank_data = df[df['股票代码'] == bank].reset_index(drop=True)
        # 提取x和y
        x = bank_data[['ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '总市值', '年龄']]
        x['总市值'] = np.log(x['总市值'])
        y = bank_data['风险溢价']

        # 找到日期为2019-08-31的数据位置,2020-03-31
        index_dot = bank_data[bank_data['日期'] == '2019-08-31'].index.tolist()[0]

        # 读取测试集
        y_test = y[0:index_dot]

        y_pred = []
        y_bar = []

        # 递归
        for k in range(len(y_test)):
            # 提取训练集
            x_train_temp = x.iloc[index_dot - k:, :].reset_index(drop=True)
            y_train = y[index_dot - k:].reset_index(drop=True)
            # 均值预测
            y_mean = np.mean(y_train)

            # 根据训练集生成主成分
            pca = PCA(n_components=1)
            pca.fit(x_train_temp[['ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '年龄']])

            # 降维后的数据
            x_new = pca.transform(x_train_temp[['ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '年龄']])
            x_train = pd.DataFrame(x_new, columns=['F1'])

            # 添加总市值
            x_train['总市值'] = x_train_temp['总市值']
            # 添加常数项
            x_train['const'] = 1
            # 计算F1的标准差
            # x_train['F1'] = x_train['F1'] / np.std(x_train['F1'])

            # 把y_train变为0或1
            # y_train = y_train / y_sig
            # y_train = np.where(y_train >= 0, 1, 0)

            # 回归
            # model = sm.OLS(y_train, x_train).fit(cov_type='HAC', use_t=True, cov_kwds={'maxlags': 1})
            model = fwls(x_train, y_train)

            # 对x_train进行标准化
            # x_train['F1'] = (x_train['F1'] - np.mean(x_train['F1'])) / np.std(x_train['F1'])
            # x_train['总市值'] = (x_train['总市值'] - np.mean(x_train['总市值'])) / np.std(x_train['总市值'])

            # model = sm.Logit(y_train, x_train).fit(cov_type='HAC', use_t=True, cov_kwds={'maxlags': 1})

            # 回归预测
            y_temp = model.predict(x_train.iloc[0, :])  # 提取最新一条x

            # 将y_temp插入到y_pred，这里是向后插入，因此后面y_test需要反转
            y_pred.append(y_temp[0])
            y_bar.append(y_mean)

        # 反转y_test，和y_pred和y_bar匹配，注意此时时间已经变成了从旧到新
        y_test_re = y_test[::-1].reset_index(drop=True)
        # 计算R2OS
        R2OS = R2OOS(y_test_re, y_pred, y_bar)
        print('R2OS:', R2OS)

        cm = confusion_matrix(y_test_re, y_bar)
        print(cm)



        # 新建一个dataframe，用于存储数据
        ret_sub = pd.DataFrame({'y_test': y_test_re, 'y_pred': y_pred, 'y_bar': y_bar})
        # 符号预测
        ret_sub['y_pred'] = np.where(ret_sub['y_pred'] >= 0, 1, np.where(ret_sub['y_pred'] < 0, -1, np.nan))
        ret_sub['y_bar'] = np.where(ret_sub['y_bar'] >= 0, 1, np.where(ret_sub['y_bar'] < 0, -1, np.nan))

        # 计算ESG和TSH
        ret_sub['ESG'] = ret_sub['y_pred'] * ret_sub['y_test']
        ret_sub['TSH'] = ret_sub['y_bar'] * ret_sub['y_test']
        tsm_tsh = ret_sub[['ESG', 'TSH']].dropna()


        """# 计算均值
        mu_ar = tsm_tsh['ESG'].mean()
        mu_tsh = tsm_tsh['TSH'].mean()
        # 计算标准差
        std_ar = tsm_tsh['ESG'].std()
        std_tsh = tsm_tsh['TSH'].std()
        # 计算协方差
        covar_ar = np.cov(tsm_tsh['ESG'], tsm_tsh['TSH'])[0, 1]
        t = len(tsm_tsh)
        v_ar_tsh = 1 / t * (
                    2 * std_ar ** 2 * std_tsh ** 2 - 2 * std_ar * std_tsh * covar_ar + 0.5 * mu_ar ** 2 * std_tsh ** 2 + 0.5 * mu_tsh ** 2 * std_ar ** 2 - mu_ar * mu_tsh * covar_ar ** 2 / (
                        std_ar * std_tsh))

        # 填充ar表格
        Table9_ar.loc[[bank], ['ESG_return']] = mu_ar * 100
        Table9_ar.loc[[bank], ['TSH_return']] = mu_tsh * 100
        Table9_ar.loc[[bank], ['ESG_SR']] = mu_ar / std_ar
        Table9_ar.loc[[bank], ['TSH_SR']] = mu_tsh / std_tsh
        Table9_ar.loc[[bank], ['Return_Difference']] = (mu_ar - mu_tsh) * 100
        Table9_ar.loc[[bank], ['SR_Difference']] = mu_ar / std_ar - mu_tsh / std_tsh


    # 保存表格
    Table9_ar.to_excel('Table1.xlsx',sheet_name='Sheet1')
"""