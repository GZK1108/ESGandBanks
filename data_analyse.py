import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import statsmodels.api as sm

df = pd.read_excel('bankdata/sortESGdata1.xlsx', sheet_name=0, header=0)
df['总市值'] = np.log(df['总市值'])
# 按股票代码分组，进行变量统计性描述
# stat = df.groupby(['股票代码'])['风险溢价','ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '总市值', '年龄'].describe()
# stat = df[['风险溢价','ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '总市值', '年龄']].describe()
# stat.to_excel('stat.xlsx', sheet_name='Sheet1', index=True)
# 变量相关性分析
# corr = df.groupby(['股票代码'])['ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '总市值', '年龄'].corr()
# corr = df.groupby(['股票代码'])['ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '年龄'].corr()
#corr = df[['风险溢价','ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '总市值','年龄']].corr()
#corr.to_excel('corr.xlsx', sheet_name='Sheet1', index=True)

# 对三个分组进行kmo检验与bartlett球形检验
"""for i in range(1, 4):
    # 查找股票代码为i的数据
    df1 = df[df['股票代码'] == i][['ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '总市值', '年龄']]
    kmoall, kmomodel = calculate_kmo(df1)
    b, p = calculate_bartlett_sphericity(df1)
    print(kmoall, p)
"""

# 计算变量之间的VIF，从结果来看，多重共线性严重
"""vif = pd.DataFrame()
# 添加常数项
df['const'] = 1
vif["VIF Factor"] = [variance_inflation_factor(df[['const', 'ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '总市值', '年龄']].values, i) for i in range(df[['const','ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '总市值', '年龄']].shape[1])]
vif["features"] = df[['const','ESG综合评价得分', '环境评价得分', '社会责任评价得分', '治理评价得分', '总市值', '年龄']].columns
print(vif)"""


