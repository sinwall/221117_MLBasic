import pandas as pd
import numpy as np
import os

from sklearn.base import TransformerMixin, BaseEstimator
from scipy.stats import kurtosis
from tsfresh.feature_extraction.extraction import extract_features

class ElementaryExtractor(BaseEstimator, TransformerMixin):
    version = 2 #moved self. features to use to transform
    stat_list = [
        'mean_',
        'med_val_',
        'min_abs_val_',
        'max_val_',
        'min_val_',
        'med_abs_diff_',
        'max_abs_diff_',
        'sum_abs_diff_',
        'l2_sum_',
        'l2_sum_diff_',
        'l2_sum_diff2_',
        'std_',
        'iqr_',
        'kurt_',
        'std_diff_',
        'iqr_diff_',
        'kurt_diff_',
        ]
    def __init__(self,channel_list=None):
        self.channel_list = channel_list
    def fit(self, X,y=None):
        return self
    
    def transform(self, X, y=None):
        self.features_to_use = []
        for i,channel in enumerate(self.channel_list):
            c_name = channel[0][:channel[0].rfind('_')]
            self.features_to_use +=[f_name+c_name for f_name in self.stat_list] 
        if self.channel_list is not None:
            sample_size,_ =X.shape
            _,channel_length =X[self.channel_list[0]].shape
            new_X = np.zeros((sample_size,channel_length,len(self.channel_list)))
            for i,channel in enumerate(self.channel_list):
                new_X[:,:,i] = X[channel]
        else:
            x = X.loc[:, 'sensor_00':'sensor_12'].values.reshape(-1, 60, 13)
        features = dict()
        for i,columns in enumerate(self.channel_list):
            c_name = columns[0][:columns[0].rfind('_')]
            channel = new_X[:, :, i]
            # mean
            features[f'mean_{c_name}'] = np.mean(channel, axis=1)
            # median of values
            features[f'med_val_{c_name}'] = np.median(channel, axis=1)
            # minimum of absolute values
            features[f'min_abs_val_{c_name}'] = np.min(np.abs(channel), axis=1)
            # maximum of values
            features[f'max_val_{c_name}'] = np.max(channel, axis=1)
            # minimum of value
            features[f'min_val_{c_name}'] = np.min(channel, axis=1)
            #median of absolute diff
            features[f'med_abs_diff_{c_name}'] = np.median(np.abs(np.diff(channel, axis=1)), axis=1)
            # maximum of absolute diff
            features[f'max_abs_diff_{c_name}'] = np.max(np.abs(np.diff(channel, axis=1)), axis=1)
            # absolute sum of difference
            features[f'sum_abs_diff_{c_name}'] = np.sum(np.abs(np.diff(channel, axis=1)), axis=1)
            # square sum
            features[f'l2_sum_{c_name}'] = np.linalg.norm(channel, axis=1)
            # square sum of difference
            features[f'l2_sum_diff_{c_name}'] = np.linalg.norm(np.diff(channel, axis=1), axis=1)
            # square sum of 2-diff
            features[f'l2_sum_diff2_{c_name}'] = np.linalg.norm(np.diff(np.diff(channel, axis=1), axis=1), axis=1)
            # standard deviation
            features[f'std_{c_name}'] = np.std(channel, axis=1)
            features[f'iqr_{c_name}'] = np.quantile(channel, 0.75, axis=1) - np.quantile(channel, 0.25, axis=1)
            features[f'kurt_{c_name}'] = kurtosis(channel, axis=1)

            features[f'std_diff_{c_name}'] = np.std(np.diff(channel, axis=1), axis=1)
            features[f'iqr_diff_{c_name}'] = np.quantile(np.diff(channel, axis=1), 0.75, axis=1) - np.quantile(np.diff(channel, axis=1), 0.25, axis=1)
            features[f'kurt_diff_{c_name}'] = kurtosis(np.diff(channel, axis=1), axis=1)

        # features[f'up_count_02'] = np.sum(np.diff(sensor_02, axis=1) >= 0, axis=1)
        # features[f'up_sum_02'] = np.sum(np.clip(np.diff(sensor_02, axis=1), 0, None), axis=1)
        # features[f'up_max_02'] = np.max(np.clip(np.diff(sensor_02, axis=1), 0, None), axis=1)
        # features[f'up_mean_02'] = np.nan_to_num(features[f'up_max_02'] / features[f'up_count_02'], posinf=40)

        # features[f'down_count_02'] = np.sum(np.diff(sensor_02, axis=1) < 0, axis=1)
        # features[f'down_sum_02'] = np.sum(np.clip(np.diff(sensor_02, axis=1), None, 0), axis=1)
        # features[f'down_min_02'] = np.sum(np.clip(np.diff(sensor_02, axis=1), None, 0), axis=1)
        # features[f'down_mean_02'] = np.nan_to_num(features[f'down_min_02'] / features[f'down_count_02'], neginf=-40)
        
        return pd.DataFrame(features)[self.features_to_use]

from sklearn.base import TransformerMixin, BaseEstimator
class BasicTransformer(BaseEstimator,TransformerMixin):
    version= 2
    columns=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT_1', 'BILL_AMT_2',
       'BILL_AMT_3', 'BILL_AMT_4', 'BILL_AMT_5', 'BILL_AMT_6', 'PAY_AMT_1',
       'PAY_AMT_2', 'PAY_AMT_3', 'PAY_AMT_4', 'PAY_AMT_5', 'PAY_AMT_6', 
       'USAGE_1', 'USAGE_2', 'USAGE_3', 'USAGE_4', 'USAGE_5',
       'DIFF_0', 'DIFF_1', 'DIFF_2', 'DIFF_3',
       'DIFF_4', 'log_LIMIT_BAL', 'log_BILL_AMT_1', 'log_BILL_AMT_2',
       'log_BILL_AMT_3', 'log_BILL_AMT_4', 'log_BILL_AMT_5', 'log_BILL_AMT_6',
       'log_PAY_AMT_1', 'log_PAY_AMT_2', 'log_PAY_AMT_3', 'log_PAY_AMT_4',
       'log_PAY_AMT_5', 'log_PAY_AMT_6', 'log_USAGE_1', 'log_USAGE_2',
       'log_USAGE_3', 'log_USAGE_4', 'log_USAGE_5', 'log_DIFF_0',
       'log_DIFF_1', 'log_DIFF_2', 'log_DIFF_3',
       'log_DIFF_4']
    log_columns=['log_LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1',
       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
       'log_BILL_AMT_1','log_BILL_AMT_2', 'log_BILL_AMT_3', 'log_BILL_AMT_4', 'log_BILL_AMT_5',
       'log_BILL_AMT_6', 'log_PAY_AMT_1', 'log_PAY_AMT_2', 'log_PAY_AMT_3',
       'log_PAY_AMT_4', 'log_PAY_AMT_5', 'log_PAY_AMT_6', 'log_USAGE_1',
       'log_USAGE_2', 'log_USAGE_3', 'log_USAGE_4', 'log_USAGE_5',
       'log_DIFF_0', 'log_DIFF_1', 'log_DIFF_2', 'log_DIFF_3', 'log_DIFF_4']
    def __init__(self,scale = 'log'):            
        self.scale = scale 
    def fit(self,x=None , y = None):
        return self
    def log_pre (self,x):
        return np.log(1+abs(2*x))*np.sign(x)
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
        df = pd.concat([df,df_log[[column for column in df_log.columns if column !='ID']]],axis=1)
        if self.scale == 'log':
            return df[self.log_columns]
        #    return df[self.log_columns].to_numpy()
        return df[self.columns]
        # return df[self.columns].to_numpy()
    
from pyts.multivariate.transformation import MultivariateTransformer
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
class MyMulPyts(BaseEstimator,TransformerMixin):
    r''''my-adaptation for multivariate time series of pyts
     Parameters
    ----------
    estimator : estimator object or list thereof
        Transformer. If one estimator is provided, it is cloned and each clone
        transforms one feature. If a list of estimators is provided, each
        estimator transforms one feature.

    flatten : bool (default = True)
        Affect shape of transform output. If True, ``transform``
        returns an array with shape (n_samples, \*). If False, the output of
        ``transform`` from each estimator must have the same shape and
        ``transform`` returns an array with shape (n_samples, n_features, \*).
        Ignored if the transformers return sparse matrices.

    Attributes
    ----------
    estimators_ : list of estimator objects
        The collection of fitted transformers.
    '''
    version =2
    def __init__(self,estimator,flatten=True, channel_list=None):
        #time stamp of channel must be same size
        #ex: USAGE and DIFF has size 5 instead of 6
        self.estimator = estimator
        self.flatten = flatten
        self.channel_list = channel_list
    def fit(self,X, y=None):
        """fit.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            or my_dataframe if channel list provided
            Multivariate time series.

        y : None or array-like, shape = (n_samples,) (default = None)
            Class labels.

        Returns
        -------
        self : object

        """
        self.transformer = MultivariateTransformer(estimator=self.estimator,flatten=self.flatten)
        if self.channel_list != None:
            sample_size,_ =X.shape
            _,channel_length =X[self.channel_list[0]].shape
            new_X = np.zeros((sample_size,len(self.channel_list),channel_length))
            for i,channel in enumerate(self.channel_list):
                new_X[:,i,:] = X[channel]
        self.transformer.fit(new_X,y=None)
        return self
    def transform(self,X,y=None):
        r"""Apply transform to each feature.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            or my_dataframe if channel list provided
            Multivariate time series.

        Returns
        -------
        X_new : array, shape = (n_samples, *) or (n_samples, n_features, *)
            Transformed time series.

        """
        if self.channel_list != None:
            sample_size,_ =X.shape
            _,channel_length =X[self.channel_list[0]].shape
            new_X = np.zeros((sample_size,len(self.channel_list),channel_length))
            for i,channel in enumerate(self.channel_list):
                new_X[:,i,:] = X[channel]
        return self.transformer.transform(new_X)
    
class NonTsPass(BaseEstimator, TransformerMixin):
    columns = [
        'SEX',
        'EDUCATION',
        'AGE',
        'log_LIMIT_BAL'
    ]
    def fit(self,X,y=None):
        return self
    def transform(self, X, y=None):
        return X[self.columns]