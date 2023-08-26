import gc
import akshare as ak
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 读取银行列表
banks = pd.read_excel('banks.xlsx', sheet_name=0, header=0,dtype = object)['股票代码']

"""for bank in banks:
    data = ak.stock_zh_valuation_baidu(symbol=bank, indicator="总市值", period="近十年")
    # 保存data，命名为bank
    data.to_excel('marketvalues'+'\\'+bank+'.xlsx', sheet_name='Sheet1', index=False)"""


