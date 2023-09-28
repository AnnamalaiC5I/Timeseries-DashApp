import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose


class eda:
      
      def __init__(self,df: pd.DataFrame) -> None:
               self.df = df
               self.rows = df.shape[0]
               self.cols = df.shape[1]
               self.columns_names = list(df.columns)

      def decomposition(self,variable=None) -> None:
                    # Perform seasonal decomposition

                    if variable == None:
                        result = seasonal_decompose(self.df, model='additive')
                        data = self.df.copy()

                    else:
                        result = seasonal_decompose(self.df[variable], model='additive')
                        data = self.df[variable].copy()
                    
                    # Extract the components
                    trend = result.trend
                    seasonal = result.seasonal
                    residual = result.resid
                    
                    # Plot the components
                    plt.figure(figsize=(10, 8))
                    plt.subplot(4, 1, 1)
                    plt.plot(data, label='Original')
                    plt.legend(loc='best')
                    
                    plt.subplot(4, 1, 2)
                    plt.plot(trend, label='Trend')
                    plt.legend(loc='best')
                    
                    plt.subplot(4, 1, 3)
                    plt.plot(seasonal, label='Seasonality')
                    plt.legend(loc='best')
                    
                    plt.subplot(4, 1, 4)
                    plt.plot(residual, label='Residuals')
                    plt.legend(loc='best')
                    
                    plt.tight_layout()
                    plt.show()


      def check_stationarity(self,variable: str) -> None:
            
            assert variable in self.columns_names, f"{variable} not found in axis. columns: {self.columns_names}"
            result=adfuller(self.df[variable])
            #print(result)
            labels = ['Test parameters', 'p-value','#Lags Used','Dataset observations']
            for value,label in zip(result,labels):
                print(label+' : '+str(value) )
            if result[1] <= 0.05:
                print("Dataset is stationary")
            else:
                print("Dataset is non-stationary ")
