from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
class default_transfomer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.columns=['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 
       'USAGE_1', 'USAGE_2', 'USAGE_3', 'USAGE_4', 'USAGE_5',
       'DIFF_0', 'DIFF_1', 'DIFF_2', 'DIFF_3',
       'DIFF_4', 'log_LIMIT_BAL', 'log_BILL_AMT1', 'log_BILL_AMT2',
       'log_BILL_AMT3', 'log_BILL_AMT4', 'log_BILL_AMT5', 'log_BILL_AMT6',
       'log_PAY_AMT1', 'log_PAY_AMT2', 'log_PAY_AMT3', 'log_PAY_AMT4',
       'log_PAY_AMT5', 'log_PAY_AMT6', 'log_USAGE_1', 'log_USAGE_2',
       'log_USAGE_3', 'log_USAGE_4', 'log_USAGE_5', 'log_DIFF_0',
       'log_DIFF_1', 'log_DIFF_2', 'log_DIFF_3',
       'log_DIFF_4']
        self.log_columns=['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','log_LIMIT_BAL',
       'log_BILL_AMT1','log_BILL_AMT2', 'log_BILL_AMT3', 'log_BILL_AMT4', 'log_BILL_AMT5',
       'log_BILL_AMT6', 'log_PAY_AMT1', 'log_PAY_AMT2', 'log_PAY_AMT3',
       'log_PAY_AMT4', 'log_PAY_AMT5', 'log_PAY_AMT6', 'log_USAGE_1',
       'log_USAGE_2', 'log_USAGE_3', 'log_USAGE_4', 'log_USAGE_5',
       'log_DIFF_0', 'log_DIFF_1', 'log_DIFF_2', 'log_DIFF_3', 'log_DIFF_4']
    
    def fit(self):
        pass
    def log_pre (self,x):
        return np.log(1+abs(x))*np.sign(x)
    def log_pre_col (self,column):
        return column.map(lambda x:self.log_pre(x))
    
    def transform(self,df,y=None):
        #Usage
        df_usage = pd.DataFrame()
        df_usage['ID']=df.ID
        BILL = [column for column in df.columns if 'BILL' in column]
        PAY = [column for column in df.columns if 'PAY_AMT' in column]
        for i in range(5):
            df_usage[f"USAGE_{i+1}"] = df[BILL[i]]-(df[BILL[i+1]]-df[PAY[i]])
        #difference
        df_difference =pd.DataFrame()
        df_difference['ID'] = df.ID
        for i in range(5):
            df_difference[f"DIFF_{i}"] = df[BILL[i+1]]-df[PAY[i]]
        USAGE = [column for column in df_usage.columns if column !='ID']
        DIFF = [column for column in df_difference.columns if column !='ID']
        LIM =['LIMIT_BAL']
        ## Logarithmic Scaling attribute 
        df = pd.concat([df,df_usage.iloc[:,1:],df_difference.iloc[:,1:]],axis=1)
        df_log =pd.concat([df.ID,df[LIM+BILL+PAY+USAGE+DIFF].apply(self.log_pre_col,axis=0)],axis=1) #apply(function,axis=) map of those who use index/column as a index
        rename_dict = {}
        for i in df_log.iloc[:,1:].columns:
            rename_dict[i] = 'log_'+i
        df_log =df_log.rename(rename_dict,axis=1)
        self.BILL = BILL
        self.PAY =PAY
        self.DIFF = DIFF
        self.USAGE =USAGE
        return pd.concat([df,df_log[[column for column in df_log.columns if column !='ID']]],axis=1)