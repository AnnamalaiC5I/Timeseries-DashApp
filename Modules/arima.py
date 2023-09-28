import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

import warnings



class Arima:
    def __init__(self, df: pd.DataFrame):
    
        self.df = df
        #self.date_column = date_column
        #self.target_column = target_column
        self.train = None
        self.test = None
        self.model = None
        
class Univariate(Arima):

    def fit_arima(self,train_df:pd.DataFrame,target_column: str, seasonality: bool = False) -> None:
      
        self.model = auto_arima(train_df[target_column],seasonal=seasonality, 
                                start_P=1, start_Q=1, D=0,
                                max_P=4, max_Q=4, m=12,
                                trace=True, suppress_warnings=True)

        # self.model = auto_arima(train_df[target_column], 
        #               start_p=1, start_q=1,
        #               test='adf',       # use adftest to find optimal 'd'
        #               max_p=4, max_q=4, # maximum p and q
        #               m=12,              # frequency of series
        #               d=None,           # let model determine 'd'
        #               seasonal=seasonality,   # No Seasonality
        #               start_P=1, 
        #               start_Q=1,
        #               D=0, 
        #               max_P=4, max_Q=4,
        #               trace=True,
        #               error_action='ignore',  
        #               suppress_warnings=True, 
        #               stepwise=True)
        #self.model.fit(train_df[target_column])
        return self.model

    def forecast(self,model, train:pd.DataFrame,test:pd.DataFrame) -> np.ndarray:
      
        start = len(train)
        end = len(train) + len(test) - 1
       
        pred = model.predict(n_periods=len(test), start=test.index[0], end=test.index[-1], return_conf_int=False)
        return pred
    
class Multivariate(Arima):
    def __init__(self, data: pd.DataFrame, target_columns: str, exogenous_columns: list=None, ):
        self.data = data
        self.target_columns = target_columns
        self.exogenous_columns = exogenous_columns

    def fit_arima(self,train: pd.DataFrame,seasonality: bool=False) -> None:
        self.models = {}
        
            
        model1 = auto_arima(train[self.target_columns], exogenous=train[self.exogenous_columns], 
                      start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=4, max_q=4, # maximum p and q
                      m=12,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=seasonality,   # No Seasonality
                      start_P=1, 
                      start_Q=1,
                      D=0, 
                      max_P=4, max_Q=4,
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
        self.models = model1
        return self.models


    def forecast(self,model, test: pd.DataFrame) -> np.ndarray:
                predictions = {}
                
                #model = self.models

                start_date = test.index[0]
                end_date = test.index[-1]
                
                start = test.index[0].to_pydatetime()
                end = test.index[-1].to_pydatetime()

                forecast = model.predict(n_periods=len(test),start=start, end=end, exogenous=test[self.exogenous_columns])

                predictions= forecast

                return predictions 
