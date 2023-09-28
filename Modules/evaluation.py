import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import mlflow.tensorflow as mlflow_tf
from mlflow.keras import log_model, autolog
import tensorflow as tf

class Evaluation:
       
        def __init__(self):
             pass
            
        def mse(self,actual: pd.Series,predicted: pd.Series) -> float:
                 mse = mean_squared_error(actual,predicted)
                 return mse
                
        def mae(self,actual: pd.Series, predicted: pd.Series) -> float:
                 mae = mean_absolute_error(actual,predicted)
                 return mae
                
        def rmse(self,actual: pd.Series,predicted: pd.Series) -> float:
                 mse = self.mse(actual,predicted)
                 return np.sqrt(mse)
        
        def mape(self,actual: pd.Series,predicted: pd.Series) -> float:
                 return np.mean(np.abs((actual - predicted) / actual)) * 100
            
        def log_model_metrics(self,model, actual: pd.Series,predicted: pd.Series, experiment_name: str, run_name: str) -> None:
            
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment(experiment_name)
            
            mse = self.mse(actual,predicted)
            mae = self.mae(actual,predicted)
            rmse = self.rmse(actual,predicted)
            mape = self.mape(actual,predicted)
            
            metrics = {"mean_squared_error":mse,"mean_absolute_error":mae,"RMSE":rmse,"MAPE":mape}
            
            try:
                params = model.get_params()
                flag=True
                
            except:
                flag = False
            
            date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with mlflow.start_run(run_name=f"{run_name}_{date_now}"):
                        for key, value in metrics.items():
                            mlflow.log_metric(key, value)
                        
                        if flag:
                                for key, value in params.items():
                                    mlflow.log_param(key,value)
                        
                        if run_name == 'LSTM':
                                log_model(model,"model")
                                
                        else:
                               mlflow.sklearn.log_model(model, "model")
                            
            print("logs are added to the mlflow experiment tracker")
                            
            return 0