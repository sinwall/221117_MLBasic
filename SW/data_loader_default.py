import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class load_data_default ():
    def __init__(self,random_state=42):
        self.random_state = random_state
        self.split = StratifiedShuffleSplit(n_splits=1, test_size = 0.25, random_state=random_state)
        df_original = pd.read_excel("./input/default of credit card clients.xls",header=1) #rename mal-nomers
        df_original =df_original.rename({
                                        'PAY_0' : 'PAY_1',
                                        'default payment next month':'default'},axis=1)
        BILL = [column for column in df_original.columns if 'BILL' in column]
        PAY = [column for column in df_original.columns if 'PAY_AMT' in column]                            
        rename_dict = {}
        for column in BILL+PAY:
            rename_dict[column]=column[:-1]+'_'+column[-1]
        df_original = df_original.rename(rename_dict, axis=1)
        BILL = [column for column in df_original.columns if 'BILL' in column] #columns with BILL_AMT_n
        PAY = [column for column in df_original.columns if 'PAY_AMT' in column]                           
        self.BILL =BILL
        self.PAY = PAY
        df = df_original.copy()
        N_labels = [column for column in df_original.columns if column !='default']
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        for train_index , test_index in split.split(df,df.iloc[:,[2,-1]]):
            self.df_train, self.label_train= df[N_labels].loc[train_index],df.default[train_index]
            self.df_test, self.label_test= df[N_labels].loc[test_index], df.default[test_index]
    
    def load_train_data(self):
        return self.df_train,self.label_train

    def load_splitted_data(self):
        for train_index , test_index in self.split.split(self.df_train,pd.concat([self.df_train.SEX,self.label_train],axis=1)):
                X_train, y_train= self.df_train.iloc[train_index],self.label_train.iloc[train_index]
                X_val, y_val= self.df_train.iloc[test_index], self.label_train.iloc[test_index]       
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        return X_train,y_train,X_val,y_val
    
    D_PAY =['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'] 
    L_PAY=['log_PAY_AMT_1', 'log_PAY_AMT_2', 'log_PAY_AMT_3', 'log_PAY_AMT_4', 'log_PAY_AMT_5', 'log_PAY_AMT_6'] 
    L_BILL=['log_BILL_AMT_1', 'log_BILL_AMT_2', 'log_BILL_AMT_3', 'log_BILL_AMT_4', 'log_BILL_AMT_5', 'log_BILL_AMT_6'] 
    L_USAGE=['log_USAGE_1', 'log_USAGE_2', 'log_USAGE_3', 'log_USAGE_4', 'log_USAGE_5'] 
    L_DIFF =['log_DIFF_1', 'log_DIFF_2', 'log_DIFF_3', 'log_DIFF_4', 'log_DIFF_5']
    USAGE = [f'USAGE_{i}' for i in range(1,6)]
    DIFF = [f'DIFF_{i}' for i in range(1,6)]
