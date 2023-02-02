import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from mods_defaults import BasicTransformer
my_basic_transformer = BasicTransformer('')

class load_data_default ():
    '''
    self.df_train : non test data
    self.StratifiedKFold : StratifiedKFold with fixed data and parameter
    self.split : split method of StratifiedKFold with fixed data and parameter thus don't need parameters
    self.yield yields X_train,y_train,X_val,y_val
    '''
    def __init__(self,path="./input/default of credit card clients.xls", random_state=42,exclude=False):
        """
        path = path of input data eg: ./input/ default of credit card clients.xls
        (path goes into pd.read_excel(path,header=1)
        random state should be fixed for collab. purpose

        Note
        load_data_default goes through renaming process
        1.indexing of discrete PAY
        2.BILL_AMTn to BILL_AMT_n
        3.same for PAY_AMT
        """
        self.random_state = random_state
        self.exclude = exclude
        df_original = pd.read_excel(path,header=1) #rename mal-nomers
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
        df_t = my_basic_transformer.transform(df)
        N_labels = [column for column in df_original.columns if column !='default']
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        for train_index , test_index in split.split(df,df.iloc[:,[2,-1]]):
            self.df_train, self.label_train= df[N_labels].loc[train_index],df.default[train_index]
            self.df_test, self.label_test= df[N_labels].loc[test_index], df.default[test_index]
        if self.exclude == True:
            df= pd.concat([self.df_train,self.label_train] ,axis=1)
            df = df.loc[
                        ((df.BILL_AMT_1>1001) | (df.default!=1)) &
                        ((df.PAY_1!=-2) | (df.default!=1)) &
                        ((df_t.DIFF_1>1501)|(df.default!=1)) ]
            self.df_train = df[[column for column in df.columns if column != 'default']]
            self.label_train = df.default
        self.StratifiedKFold =StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)
    
    def load_train_data(self):
        '''
        return non test(not black box) data
        '''
        return self.df_train,self.label_train

    def split(self):
        '''
        Yields StratifiedKFold's split with fixed data, parameter
        X= non test data frame i.e. self.df_train
        y= SEX and default of non test data
        n_split = 5
        shuffle True
        random state =42
        '''
        return self.StratifiedKFold.split(self.df_train,self.label_train)
    def yield_data(self):
        '''
        Yield tuple (X_train, y_train, X_val, y_val) of StratifiedKFold.split with fixed data and parameter
        '''
        for train_index , test_index in self.split():
            X_train, y_train= self.df_train.iloc[train_index],self.label_train.iloc[train_index]
            X_val, y_val= self.df_train.iloc[test_index], self.label_train.iloc[test_index]       
            yield X_train,y_train,X_val,y_val
    
    D_PAY =['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'] 
    '''list of column names of associated category'''
    L_PAY=['log_PAY_AMT_1', 'log_PAY_AMT_2', 'log_PAY_AMT_3', 'log_PAY_AMT_4', 'log_PAY_AMT_5', 'log_PAY_AMT_6'] 
    '''list of column names of associated category'''
    L_BILL=['log_BILL_AMT_1', 'log_BILL_AMT_2', 'log_BILL_AMT_3', 'log_BILL_AMT_4', 'log_BILL_AMT_5', 'log_BILL_AMT_6'] 
    '''list of column names of associated category'''
    L_USAGE=['log_USAGE_1', 'log_USAGE_2', 'log_USAGE_3', 'log_USAGE_4', 'log_USAGE_5'] 
    '''list of column names of associated category'''
    L_DIFF =['log_DIFF_1', 'log_DIFF_2', 'log_DIFF_3', 'log_DIFF_4', 'log_DIFF_5']
    '''list of column names of associated category'''
    USAGE = [f'USAGE_{i}' for i in range(1,6)]
    '''list of column names of associated category'''
    DIFF = [f'DIFF_{i}' for i in range(1,6)]
    '''list of column names of associated category'''
