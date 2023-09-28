import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class Plotter:
    def __init__(self,df: pd.DataFrame):
           self.df = df
    
    def acf_pacf_plot(self,target: str) -> None:
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(self.df[target], ax=axs[0])
        plot_pacf(self.df[target], ax=axs[1], lags=14)
        plt.tight_layout()
        plt.show()

    def plot_predictions(self, df: pd.DataFrame, actual: str, prediction: str) -> None:

        fig = go.Figure()
        
        actual_trace = go.Scatter(x=df.index,y=df[actual],mode='lines',name=f'Actual {actual}')
        predicted_trace = go.Scatter(
                x=df.index,y=df[prediction],mode='lines',name=f'Predicted {prediction}')
        fig.add_trace(actual_trace)
        fig.add_trace(predicted_trace)
        fig.update_layout(
            title='Actual vs Predicted',xaxis=dict(title='Date'),yaxis=dict(title='Value'), showlegend=True)
        fig.show()


        



