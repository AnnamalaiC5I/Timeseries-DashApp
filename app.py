import dash
from dash import html, Dash, callback, Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from assets.footer import _footer
import pickle
import statsmodels.api as sm
from Modules import import_data, preprocess, eda, anomaly_detection, evaluation, LSTM, plot, arima
import yaml
import base64
from itertools import product
import pandas as pd
import io
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import numpy as np
import warnings
from pmdarima.utils import diff
import sweetviz as sv
from assets.fig_layout import my_figlayout, my_linelayout,train_linelayout, test_linelayout, anomaly_marker, forecast_linelayout, scatter_marker
from assets.acf_pacf_plots import acf_pacf

from assets.sarima_gridsearch import sarima_grid_search

warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

_data_airp = pd.read_csv(r'C:\Users\annamalai\Downloads\my_dashboard\data\AirPassengers.csv' ,usecols = [0,1], names=['Time','Values'], skiprows=1)
_data_airp['Time'] = pd.to_datetime(_data_airp['Time'], errors='raise')


def exog_prediction(dfe,features,n):
            new_df1 = pd.DataFrame()
            #n = 10  # Number of days to predict
            df =dfe.copy()
            # Iterate over each feature
            for feature in features:
                # Get the time series data for the current feature
                data = df[feature]

                # Create and fit the SARIMA model
                model1 = sm.tsa.SARIMAX(data, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7))
                model_fit = model1.fit()

                # Make predictions for the next n days
                forecast = model_fit.get_forecast(steps=n)

                # Extract the predicted values and confidence intervals
                predicted_values = forecast.predicted_mean
                confidence_intervals = forecast.conf_int()

                # Print the predictions for the next n days
#                 print(f"Predictions for {feature}:")
#                 print(predicted_values)
#                 print("\n")

                # Optionally, you can store the predictions in a new DataFrame if needed
                new_df1[feature] = predicted_values
                
            return new_df1


app = Dash(__name__, use_pages = True,external_stylesheets=[dbc.themes.UNITED, dbc.themes.BOOTSTRAP])

obj = evaluation.Evaluation()

app.layout = html.Div([
       dcc.Store('outlier-memory',data=dict(),storage_type='session'),
       dcc.Store('sweetviz-memory',data=dict(),storage_type='session'),
       dcc.Store('yaml-memory',storage_type='session'),
       dcc.Store('graph-memory',data=dict(),storage_type='session'),
       dcc.Store('var-memory',data=dict(),storage_type='session'),
       dcc.Store("dummy-memory",data=dict(),storage_type='session'),
       dcc.Store('model-memory',data=dict(),storage_type='session'),
       dcc.Store(id='memory',storage_type='session'),
       dcc.Store(id='browser-memo', data=dict(), storage_type='session'),
       
       dash.page_container,
      
])

def impute_nan_with_average(df):
    for column in df.columns:
        if df[column].dtype != 'object':  # Check if column is numeric
            average = df[column].mean()
            df[column].fillna(average, inplace=True)
    return df


############ EDA part ############################

@callback(Output('x-col-dropdown','options'),
          Output('y-col-dropdown','options'),
          Input('eda-upload','contents'))
def update_plot_options(contents):
       print("Hi contents.......................")
       if contents is not None:
                print('if hi contents..........')
                #print(contents)
                content_type, content_string = contents.split(',')
                print(content_type)

                decoded = base64.b64decode(content_string)

                print("decoded")
                
                # Read the decoded CSV data using pandas
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

                print(df)
                cols = df.columns
                print(cols)

                return cols,cols
       
@callback(Output('sweetviz-memory','data'),
          Input('plot-sweetviz-button','n_clicks'),
          State('eda-upload','contents'))
def show_sweetviz(n_clicks,contents):
        print("!@#$%^&*()Swwtviz !@#$%^&*()!@#$%^&*()!@#&*")
        if n_clicks is None:
              
               raise PreventUpdate
        else:
                if contents is not None:
                        
                        content_type, content_string = contents.split(',')

                        decoded = base64.b64decode(content_string)

                        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

                        sv_obj = sv.analyze(df)

                        report_html = sv_obj.show_html()

                        sv_obj.save("data_report.html")

                        return {"status":True}
                
                else:
                       return {"status":False, 'reason':'content is empty check df'}
       
       
@callback(Output('fig-output124','figure'),
          Input('plot-eda-page-button','n_clicks'),
          State('x-col-dropdown','value'),
          State('y-col-dropdown','value'),
          State('plot-col-dropdown','value'),
          State('eda-upload','contents'))
def plot_graph(n_clicks,x1,y1,ptype,contents):
        
        if n_clicks is None:
                fig = go.Figure(layout=my_figlayout)
                return fig
        
        else:
            print('else condition')
            if contents is not None:
                        print('if hi contents..........')
                        #print(contents)
                        content_type, content_string = contents.split(',')

                        decoded = base64.b64decode(content_string)

                        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

                        if ptype=='line':
                                    fig = go.Figure(layout =my_figlayout)
                                    fig.add_trace(go.Scatter(
                                                        x=df[x1],
                                                        y=df[y1],
                                                        mode='lines',
                                                        line = my_linelayout,
                                                        
                                                    ))     
                                    fig.update_layout(
                                    title='Time Series Line Plot',
                                    xaxis_title=x1,
                                    yaxis_title=y1)
                                    return fig
                        
                        elif ptype == 'scatter':
                                    print('scatter it is')
                                    fig = go.Figure(layout =my_figlayout)
                                    fig.add_trace(go.Scatter(
                                                        x=df[x1],
                                                        y=df[y1],
                                                        mode='markers',
                                                        marker = scatter_marker,
                                                        
                                                    ))     
                                    fig.update_layout(
                                    title='Time Series Scatter Plot',
                                    xaxis_title=x1,
                                    yaxis_title=y1)
                                    return fig
                        
                        elif ptype == 'area':
                               fig = go.Figure(layout =my_figlayout)
                               fig.add_trace(go.Scatter(x=df[x1], y=df[y1], fill='tozeroy',mode='lines',line =train_linelayout ,fillcolor='#3DED97'))
                               fig.update_layout(
                                    title='Time Series Area Plot',
                                    xaxis_title=x1,
                                    yaxis_title=y1)
                               return fig
            else:
                fig = go.Figure(layout=my_figlayout)
                fig.add_annotation(
                                                x=0.5,
                                                y=0.5,
                                                xref='paper',
                                                yref='paper',
                                                text="Upload your dataset",
                                                showarrow=False,
                                                font=dict(color='black', size=16)
                                            )
                return fig
                                


        
############ EDA part ############################

############ Mlflow part ############################

@callback(
         Output('dummy-memory','data'),
         Input('model-memory','data')
         )
def log_evaluation(data):

    print("log evaluation ka callback triggered...............")
    if bool(data):
        load_model = data['model_path']
        loc = load_model
        with open(loc, "rb") as file:
                  uni_model = pickle.load(file)
        print("Mlflow model loaded")
                
        qwer = pd.DataFrame(data['test_df'])
        values = qwer['Values']
        predicted = qwer['predicted']
        
        print("Mlflow values and predicted series are ready")
        run_name = data['model_path'].split('.')[0]
        
        
        obj.log_model_metrics(uni_model,values,predicted,"Evaluation_check",run_name)
        print("Mlflow logged the metrics with values")
        print("################################")
        return {'status':True}
    else:
         print("Model-memory is empty")
         raise {'status':"Model-memory is empty"}


############ Mlflow part ############################


############ Anomaly Detection ############################


@callback(
          Output('feature-title','style'),
          Output('crate-title','style'),
          Output('handle-outlier-download','style'),
          Output('handle-outlier-plot','style'),
          Output('handle-outlier','style'),
          Output('anomaly-number-input','style'),
          Output('anomaly-dropdown1','style'),
          Output('anomaly-dropdown1','options'),
          Input('anomaly-upload1','contents'))
def update_anomaly_df(contents):
        print(contents)
        if contents is not None:
                content_type, content_string = contents.split(',')

                # Decode the base64 encoded CSV data
                decoded = base64.b64decode(content_string)
                
                # Read the decoded CSV data using pandas
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

                cols = df.columns
                print(cols)

                download_style = {
                        'background-color': '#042f33',
                        'color': '#000000',
                        'width':'100px',
                        'border-radius': '5px',
                        'padding': '5px',
                        'border': 'none',
                        'display':'block',
                        'outline': 'none',
                        'box-shadow': '0 0 5px rgba(0, 0, 0, 0.3)',
                        'margin-top':'40px'
                    }

                handle_outlier_plot_style ={
                        'background-color': '#042f33',
                        'color': '#000000',
                        'width':'150px',
                        'border-radius': '5px',
                        'padding': '5px',
                        'border': 'none',
                        'display':'block',
                        'outline': 'none',
                        'box-shadow': '0 0 5px rgba(0, 0, 0, 0.3)',
                        'margin-top':'40px'
                    }

                handle_outlier_style = {'background-color': '#042f33',
                        'color': '#000000',
                        'width':'150px',
                        'border-radius': '5px',
                        'padding': '5px',
                        'border': 'none',
                        'display':'block',
                        'outline': 'none',
                        'box-shadow': '0 0 5px rgba(0, 0, 0, 0.3)',
                        'margin-top':'40px'}
                
                crate_style = {'display':'block',
                               'width':'200px',
                               'background-color':'#042f33',
                               'borderRadius': '5px',
                               'padding': '5px',
                               'border': 'none',
                               'outline': 'none'}
                
                dropdown_style = {'width':'200px','display':'block'}

                style = {'display':'block'}

                return style, style, download_style, handle_outlier_plot_style,handle_outlier_style ,crate_style,dropdown_style,cols
        
        else:
                 print("anomaly else executed.......")
                 raise PreventUpdate
        

@callback(Output('outlier-memory','data'),
          State('anomaly-dropdown1','value'),
          State('anomaly-number-input','value'),
          State('anomaly-upload1','contents'),
          Input('handle-outlier','n_clicks'))
def find_anomalies(_drop,_crate,_data,button):
       
    if button is not None: ############## button >0
            if _data is not None:
                print("The data is not none for anomalies")
                content_type, content_string = _data.split(',')
                decoded = base64.b64decode(content_string)
                data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                print("read the csv successfully")
                var = _drop
                print("starting outlier detection")
                outlier_detection = anomaly_detection.OutlierDetection()
                outlier_detection.fit_isolation_forest(data[var],contamination=_crate)
                print("fit isolation forest on the chosen column")
                anomalies = outlier_detection.detect_outliers_isolation_forest(data[var])
                print('anomalies saved in the data')

                new_data = outlier_detection.handle_outliers(data, var)
                print('Handled anomalies and saved the new_data')

                my_dict = {'raw_data':data.to_dict('records'), 
                           'anomalies':anomalies.to_dict(),
                           'variable':_drop,
                           'new_data':new_data.to_dict('records')
                           }
                
                return my_dict
            
            else:
                    my_dict = {'raw_data':None, 
                           'anomalies':None,
                           'variable':None,
                           'new_data':None
                           }
                    return my_dict
            
    else:
            print("find anomalies outer else")
            raise PreventUpdate
            
       
@callback(Output('fig-output6','figure'),
          Input('handle-outlier-plot','n_clicks'),
          State('outlier-memory','data'))
def plot_anomalies(n_clicks,data):
        
        
        
        if n_clicks is None:
                fig = go.Figure(layout=my_figlayout)
                return fig
        
        else:
                    
                    if   data['raw_data'] is not None:
                                df = pd.DataFrame(data['raw_data'])
                                var = data['variable']
                                anomalies = pd.Series(data['anomalies'])
                                fig0 = go.Figure(layout =my_figlayout)
                                fig0.add_trace(go.Scatter(
                                            x=df.index,
                                            y=df[var],
                                            mode='lines',
                                            line = my_linelayout,
                                            name=f'{var}'
                                        ))     
                                fig0.add_trace(go.Scatter(
                                        x=anomalies.index,
                                        y=list(anomalies),
                                        mode='markers',
                                        marker = anomaly_marker,
                                        name='Anomalies'
                                    ))
                                
                                return fig0
                    
                    else:
                            fig = go.Figure(layout=my_figlayout)
                            fig.add_annotation(
                                                x=0.5,
                                                y=0.5,
                                                xref='paper',
                                                yref='paper',
                                                text="Error occured in the plot. Check your data",
                                                showarrow=False,
                                                font=dict(color='black', size=16)
                                            )
                            return fig
                    

@callback(Output("download-dataframe-csv", "data"),
    [Input("handle-outlier-download", "n_clicks")],
    [State('outlier-memory','data')],
    prevent_initial_call=True)
def download_csv(n_clicks,outlier_memory):
    print("DDDDDDDDDDDDDDDDDownload button")
    if n_clicks is not None:
        
        df = pd.DataFrame(outlier_memory['new_data'])
        print("read data from the outlier memory")
        #csv_string = df.to_csv(index=False, encoding="utf-8-sig")
        
        
        return dcc.send_data_frame(df.to_csv, "new_data.csv")
    
    else:
            print("Can't download the file from memory...................")
            raise PreventUpdate

############ Anomaly Detection ############################

############ Model Download ###############################
@callback(Output("download-model-pkl", "data"),
    [Input("download-plot-button", "n_clicks")],
    [State('model-memory','data')],
    prevent_initial_call=True)
def download_csv(n_clicks,model_memory):
    print("MMMMMMMMMMMMMMMMMMMMMMMMMModel DDDDDDDDDDDDDDDDDownload button")
    if n_clicks is not None:
        
        path = model_memory['model_path']
        print("$$$$$$$$$$$$$$$$$$$$read model from the memory")

        with open(path, "rb") as file:
                loaded_model = pickle.load(file)
        
        model_bytes = pickle.dumps(loaded_model)

        print("stuff 2")
        #csv_string = df.to_csv(index=False, encoding="utf-8-sig")
        
        
        return dcc.send_bytes(model_bytes, filename="model.pkl")
    
    else:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$Can't download the model from memory...................")
            raise PreventUpdate
############ Model Download ###############################


################ LSTM Dropdown enable/disable ##########################
@callback(
    Output('multivariate-div', 'style'),
    Output('lstm-dropdown-container', 'style'),
    Output('yaml-editor-container', 'style'),
    Input('model-dropdown', 'value')
)
def update_dropdown_visibility(model):
    print('model dropdown value is {}'.format(model))
    lstm_dropdown_style = {'display': 'none'}
    yaml_editor_style = {'display': 'none'}
    multivariate_dropdown_style = {'display':'none'}

    if model == 'arimax' or model == 'sarimax':
           multivariate_dropdown_style = {'display':'block'}

    if model == 'lstm':
        lstm_dropdown_style = {'display': 'block'}
        yaml_editor_style = {'display': 'block'}

    

    return multivariate_dropdown_style, lstm_dropdown_style, yaml_editor_style


@callback(Output('var-memory','data'),
        Input('multivariate-dropdown','value'))
def update_exog(val):
       print(val)
       return {'exog':val}

################################ Check yaml-memory #################################
@callback(Output('anna','component'),
          Input('input-button','n_clicks'),
          State('yaml-memory', 'data'))
def memory_check(n_clicks, data):
   if n_clicks and data is not None:
        print(data['layer'])
        return html.Div(data)
   
   else:
        raise PreventUpdate
   
@callback(Output('multivariate-dropdown', 'options'),
          State('memory','data'),
          Input('model-dropdown','value'))
def update_multivariate_dropdown(data,model):
        print('########################### multi variable option triggered ####################################################')
        if model == 'arimax' or model == 'sarimax':
                df = pd.DataFrame(data['data'])
                columns = [{'label': col, 'value': col} for col in df.columns]
                print('returning an options to dropdown')
                return columns
        else:
                raise PreventUpdate

############# Trigger the graph ########################
@callback(Output('fig-output5','figure'),
          Input('plot-button','n_clicks'),
          State('model-memory','data'))
def plot_graph(n_clicks, model_data):
        
        if n_clicks is None:
                fig = go.Figure(layout=my_figlayout)
                
                return fig
       
        if not model_data['error']:
            
            
            if n_clicks:
                try:
                    print("it comes inside the graph-memory callback")
                    test_df = model_data['test_df']
                    test = pd.DataFrame(test_df)

                    test['date'] = pd.to_datetime(test['date'])

                    for i in test.columns:
                            if i != 'date':
                                    mean_value = test[i].mean()
                                    test[i] = test[i].fillna(mean_value)
                    
                    #pickle_file_path = model_data['model_path']
                    print(test.shape)
                    # with open(pickle_file_path, 'rb') as file:
                    #            model = pickle.load(file)   

                    #pred = model.predict(n_periods=len(test), start=test.index[0], end=test.index[-1], return_conf_int=False)

                    #print(pred)
                    print('figure')
                    fig = go.Figure(layout=my_figlayout)

                    # Add 'Actual' line
                    fig.add_trace(go.Scatter(
                        x=test['date'],y=test['Values'],mode='lines',line=my_linelayout,name='Actual'))
                    print('trace 1')
                    # Add 'Predicted' line
                    fig.add_trace(go.Scatter(
                        x=test['date'],y=test['predicted'],mode='lines',line=test_linelayout,name='Predicted'))
                    print('trace 2')
                    # Customize the layout
                    fig.update_layout(
                        title='Actual vs Predicted',xaxis_title='Date',yaxis_title='Value'
                    )
                    print('trace 3')
                    #fig.update_traces(overwrite=True, line=my_linelayout)
                    print('writing..........')
                    return fig
                
                except Exception as e:
                        print(f"The error is {e}")
                        fig = go.Figure(layout=my_figlayout)
                        fig.add_annotation(
                                    x=0.5,
                                    y=0.5,
                                    xref='paper',
                                    yref='paper',
                                    text="Error occurred in the data",
                                    showarrow=False,
                                    font=dict(color='black', size=16)
                                )
                        return fig
            
            else:
                fig = go.Figure(layout=my_figlayout)
                fig.add_annotation(
                                    x=0.5,
                                    y=0.5,
                                    xref='paper',
                                    yref='paper',
                                    text="Can't plot the graph. Error occurred in the data",
                                    showarrow=False,
                                    font=dict(color='black', size=16)
                                )
                return fig
        
        else:
                
                fig = go.Figure(layout=my_figlayout)
                fig.add_annotation(
                                    x=0.5,
                                    y=0.5,
                                    xref='paper',
                                    yref='paper',
                                    text="Can't plot the graph. Error occurred in the data",
                                    showarrow=False,
                                    font=dict(color='black', size=16)
                                )
                return fig
        
        # pickle_file_path = 'path/to/your/pickle/file.pkl'
        # with open(pickle_file_path, 'rb') as file:
        #     model = pickle.load(file)     

       

###################### save data to model memory ##################################
@callback(Output('model-memory','data'),
          Input('input-button','n_clicks'),
          Input('memory','data'),
          State('yaml-memory','data'),
          State('model-dropdown', 'value'),
          State('split-input','value'),
          State('aggregate-dropdown','value'),
          State('multivariate-dropdown','value')
          )
def model_call(n_clicks,data,params,model,split_input,agg_level,exog):
        print('model call is triggered.........')
        if split_input is None:
                 split_input = 0.5

        else:
             pass

        if model == 'sarimax':
                
                if n_clicks and data is not None:
                        print('if clause')
                        try:
                            df = pd.DataFrame(data['data'])

                        except Exception as e:
                            print(f"An error occurred: {e}")

                        print('sarimax df reDY')

                        print('preprocess util')
                        y = 'Values'
                        req_features = exog+[y]+['date']
                        print(req_features)
                        df = df[req_features]
                        preprocess_util = preprocess.preprocess(df)

                        try:
                                    df['date'] = pd.to_datetime(df['date'])
                                    print(df.head())
                                    print("sarimax level is {}".format(agg_level))
                                    df3 = preprocess_util.aggregate(df, level=agg_level)
                                    df = impute_nan_with_average(df3)
                        except Exception as e:
                                print(f"An error occurred: {e}")
                                print('sarimax skipping aggregation process')
                                pass
                        print('sarimax The dataframe shape is {}'.format(df.shape))
                        df = preprocess_util.set_datetime_index(df)
                        print('sarimax')
                        
                        
                        print(f'sarimax train_test_split ratio is {split_input} and {exog}')

                        
                        train_df, test_df = preprocess_util.train_test_split(df, split_input)
                        print('sarimax split')
                        forecast = arima.Multivariate(data=df, target_columns=y, exogenous_columns=exog)
                        print(df.columns)
                        print('object created')
                        try:
                                
                                uni_model = forecast.fit_arima(train_df, seasonality = True)

                        except Exception as e:
                                print(f"The error is {e}")
                        
                        print('fit sarimax')
                        try:
                                predictions = forecast.forecast(uni_model, test_df) 

                        except Exception as e:
                                print(f"The error is {e}")
                                return {'model_path':None, 'test_df':None,'error':True}
                        print('sarimax forecasted values')
                        test_df['predicted'] = predictions
                        print(' sarimax returning the arima model to memory')
                        new_df = test_df.reset_index()
                        model_file_path = 'sarimax_model.pkl'
                        with open(model_file_path, 'wb') as file:
                            pickle.dump(uni_model, file)

                        return {'model_path':model_file_path, 'test_df':new_df.to_dict('records'),'exog':exog,'error':False}
        
                # if n_clicks and data is not None:
                #         print('if clause')
                #         try:
                #             df = pd.DataFrame(data['data'])

                #         except Exception as e:
                #             print(f"An error occurred: {e}")

                #         print('sarmax df reDY')

                #         print('preprocess util')
                #         y = 'Values'
                #         req_features = exog+[y]+['date']
                #         print(req_features)
                #         df = df[req_features]
                #         preprocess_util = preprocess.preprocess(df)

                #         try:
                #                  df['date'] = pd.to_datetime(df['date'])
                #                  print(df.head())
                #                  print("sArimax level is {}".format(agg_level))
                #                  df = preprocess_util.aggregate(df, level=agg_level)
                #         except Exception as e:
                #                 print(f"An error occurred: {e}")
                #                 print('sArimax skipping aggregation process')
                #                 pass
                #         print('sArimax The dataframe shape is {}'.format(df.shape))
                #         df = preprocess_util.set_datetime_index(df)
                #         print('sarimax')
                        
                        
                #         print(f'sarimax train_test_split ratio is {split_input} and {exog}')

                        
                #         train_df, test_df = preprocess_util.train_test_split(df, split_input)
                #         print('sArimax split')
                #         forecast = arima.Multivariate(data=df, target_columns=y, exogenous_columns=exog)
                #         print(df.columns)
                #         print('object created')

                #         try:
                #              uni_model = forecast.fit_arima(train_df, seasonality = True)

                #         except Exception as e:
                #                 print(f"The error is {e}")
                        
                #         print('fits sArimax')
                #         #predictions = forecast.forecast(test_df) 
                #         print('sArimax forecasted values')
                #         #test_df['predicted'] = predictions
                #         print(' sArimax returning the arima model to memory')
                #         model_file_path = 'sarimax_model.pkl'
                #         with open(model_file_path, 'wb') as file:
                #             pickle.dump(uni_model, file)

                #         return {'model_path':model_file_path, 'test_df':test_df.to_dict('records')}
                
                else:
                            print("Arimax else clause")
                            raise PreventUpdate
        
        if model == 'arimax':
                if n_clicks and data is not None:
                        print('if clause')
                        try:
                            df = pd.DataFrame(data['data'])

                        except Exception as e:
                            print(f"An error occurred: {e}")

                        print('armax df reDY')

                        print('preprocess util')
                        y = 'Values'
                        req_features = exog+[y]+['date']
                        print(req_features)
                        df = df[req_features]
                        preprocess_util = preprocess.preprocess(df)

                        try:
                                 df['date'] = pd.to_datetime(df['date'])
                                 print(df.head())
                                 print("Arimax level is {}".format(agg_level))
                                 df3 = preprocess_util.aggregate(df, level=agg_level)
                                 df = impute_nan_with_average(df3)
                        except Exception as e:
                                print(f"An error occurred: {e}")
                                print('Arimax skipping aggregation process')
                                pass
                        print('Arimax The dataframe shape is {}'.format(df.shape))
                        df = preprocess_util.set_datetime_index(df)
                        print('arimax')
                        
                        
                        print(f'arimax train_test_split ratio is {split_input} and {exog}')

                        
                        train_df, test_df = preprocess_util.train_test_split(df, split_input)
                        print('Arimax split')
                        forecast = arima.Multivariate(data=df, target_columns=y, exogenous_columns=exog)
                        print(df.columns)
                        print('object created')
                        try:
                               
                              uni_model = forecast.fit_arima(train_df, seasonality = False)

                        except Exception as e:
                                print(f"The error is {e}")
                        
                        print('fit Arimax')
                        try:
                             predictions = forecast.forecast(uni_model, test_df) 

                        except Exception as e:
                                print(f"The error is {e}")
                                return {'model_path':None, 'test_df':None,'error':True}
                        print('Arimax forecasted values')
                        test_df['predicted'] = predictions
                        print(' Arimax returning the arima model to memory')
                        new_df = test_df.reset_index()
                        model_file_path = 'arimax_model.pkl'
                        with open(model_file_path, 'wb') as file:
                            pickle.dump(uni_model, file)

                        return {'model_path':model_file_path, 'test_df':new_df.to_dict('records'),'exog':exog,'error':False}
                
                else:
                            print("Arimax else clause")
                            raise PreventUpdate
        
        if model == 'lstm':
                        if n_clicks and data is not None:
                                # print('if clause')
                                # try:
                                #     df = pd.DataFrame(data['data'])

                                # except Exception as e:
                                #         print(f"An error occurred: {e}")

                                # print('df reDY')
                                # num_layers = params['layers']
                                # print(num_layers)
                                # num_units = params['units']
                                # print(num_units)
                                # dropout = params['dropout']
                                # print(dropout)

                                # print('lstm function')
                                # lstm_func = LSTM.Univariate(df)
                                # print('object created')
                                # n_steps=3
                                # model1 = lstm_func.create_model_lstm(n_steps,num_layers, num_units,dropout_rate=dropout)
                                # print('model1 ready')
                                # y = 'Values'
                                # X, Y, X_list, Y_list = lstm_func.sampling(df[y].to_list(), n_steps)

                                # #new_df = lstm_func.create_processed_df(n_steps)

                                # model1 = lstm_func.train_model(X,Y,epochs=80)

                                # print("Model has been saved in the memory")

                                # model_file_path = os.path.join(os.getcwd(), 'lstm_model.pkl')
                                # with open(model_file_path, 'wb') as file:
                                #     pickle.dump(model1, file)

                                # print("Model has been saved in the memory")
                                model_file_path='lstm_model.pkl'
                                return {'model_path':model_file_path}
                        
                        else:
                            print("else clause")
                            raise PreventUpdate
        
        elif model == 'arima':
                if n_clicks and data is not None:
                        try:
                                    df = pd.DataFrame(data['data'])

                        except Exception as e:
                                        print(f"An error occurred: {e}")
                        
                        print('preprocess util')
                        y = 'Values'
                        date = df['date']
                        df = df[['date',y]]
                        print(f"The df column is {df.columns}")
                        preprocess_util = preprocess.preprocess(df)

                        try:
                                 df['date'] = pd.to_datetime(df['date'])
                                 print(df.head())
                                 print("level is {}".format(agg_level))
                                 df3 = preprocess_util.aggregate(df, level=agg_level)
                                 df = impute_nan_with_average(df3)
                        except Exception as e:
                                print(f"An error occurred: {e}")
                                print('skipping aggregation process')
                                pass
                        print('The dataframe shape is {}'.format(df.shape))
                        df = preprocess_util.set_datetime_index(df)
                        print('arima univariate')
                        forecast = arima.Univariate(df)
                        
                        print(f'train_test_split ratio is {split_input}')
                        train_df, test_df = preprocess_util.train_test_split(df, split_input)
                        print('split')
                        try:
                             uni_model = forecast.fit_arima(train_df,y,False)
                        
                        except Exception as e:
                             print(f"the error is {e}")
                        print('fit arima')

                        try:
                              predictions = forecast.forecast(uni_model,train_df,test_df) 

                        except Exception as e:
                            print(f'The error is {e}')
                            n_df = pd.DataFrame()
                            return {'model_path':'empty.pkl', 'test_df':n_df.to_dict('records'),'error':True}

                        print('forecasted values')
                        test_df['predicted'] = predictions
                        try:
                            new_df = test_df.reset_index()
                        
                        except Exception as e:
                            print(f'The error is {e}')
                        print('returning the arima model to memory')
                        model_file_path = 'arima_model.pkl'
                        with open(model_file_path, 'wb') as file:
                            pickle.dump(uni_model, file)
                        print('')
                        return {'model_path':model_file_path, 'test_df':new_df.to_dict('records'),'error':False}

                else:
                            print("Arima else clause")
                            raise PreventUpdate
        
        elif model == 'sarima':
                # if n_clicks and data is not None:
                #         try:
                #                     df = pd.DataFrame(data['data'])

                #         except Exception as e:
                #                         print(f"An error occurred: {e}")
                        
                #         print('preprocess util')
                #         y = 'Values'
                #         date = df['date']
                #         df = df[['date',y]]
                #         print(f"The df column is {df.columns}")

                #         preprocess_util = preprocess.preprocess(df)

                #         try:
                #                 df['date'] = pd.to_datetime(df['date'])
                #                 print(df.head())
                #                 print("level is {}".format(agg_level))
                #                 df = preprocess_util.aggregate(df, level=agg_level)
                #                 print('The dataframe shape is {}'.format(df.shape))

                #         except Exception as e:
                #                 print(f"the error is {e}")

                #         df = preprocess_util.set_datetime_index(df)
                #         print('srima univariate')
                #         forecast = arima.Univariate(df)
                        
                #         print(f'train_test_split ratio is {split_input}')
                #         train_df, test_df = preprocess_util.train_test_split(df, split_input)
                #         print('split')
                #         uni_model = forecast.fit_arima(train_df,y,True)
                #         print('fit sarima')
                #         predictions = forecast.forecast(uni_model,train_df,test_df) 
                #         print('forecasted values')
                #         test_df['predicted'] = predictions
                #         try:
                #             new_df = test_df.reset_index()
                        
                #         except Exception as e:
                #             print(f'The error is {e}')

                #         print('returning the sarima model to memory')
                #         model_file_path = 'sarima_model.pkl'
                #         with open(model_file_path, 'wb') as file:
                #             pickle.dump(uni_model, file)

                #         return {'model_path':model_file_path,'test_df':test_df.to_dict('records')}

                if n_clicks and data is not None:
                        try:
                                    df = pd.DataFrame(data['data'])

                        except Exception as e:
                                        print(f"An error occurred: {e}")
                        
                        print('preprocess util')
                        y = 'Values'
                        date = df['date']
                        df = df[['date',y]]
                        print(f"The df column is {df.columns}")
                        preprocess_util = preprocess.preprocess(df)

                        try:
                                 df['date'] = pd.to_datetime(df['date'])
                                 print(df.head())
                                 print("level is {}".format(agg_level))
                                 df3 = preprocess_util.aggregate(df, level=agg_level)
                                 df = impute_nan_with_average(df3)
                        except Exception as e:
                                print(f"An error occurred: {e}")
                                print('skipping aggregation process')
                                pass
                        print('The dataframe shape is {}'.format(df.shape))
                        df = preprocess_util.set_datetime_index(df)
                        print('arima univariate')
                        forecast = arima.Univariate(df)
                        
                        print(f'train_test_split ratio is {split_input}')
                        train_df, test_df = preprocess_util.train_test_split(df, split_input)
                        print('split')
                        try:
                             uni_model = forecast.fit_arima(train_df,y,True)
                        
                        except Exception as e:
                             print(f"the error is {e}")
                        print('fit arima')

                        try:
                              predictions = forecast.forecast(uni_model,train_df,test_df) 

                        except Exception as e:
                            print(f'The error is {e}')
                            n_df = pd.DataFrame()
                            return {'model_path':'empty.pkl', 'test_df':n_df.to_dict('records'),'error':True}
                            

                        print('forecasted values')
                        test_df['predicted'] = predictions
                        try:
                            new_df = test_df.reset_index()
                        
                        except Exception as e:
                            print(f'The error is {e}')
                        print('returning the arima model to memory')
                        model_file_path = 'sarima_model.pkl'
                        with open(model_file_path, 'wb') as file:
                            pickle.dump(uni_model, file)
                        print('')
                        return {'model_path':model_file_path, 'test_df':new_df.to_dict('records'),'error':False}

                else:
                            print("Sarima else clause")
                            raise PreventUpdate
                
        else:
                print("the outer outer else executed")
                raise PreventUpdate


                
     
############################# save data from yaml editor to yaml  memory ########################
@callback(Output('yaml-memory', 'data'),
          Input('yaml-editor', 'value'))
def update_model_values(yaml_data):
    try:
        data = yaml.load(yaml_data, Loader=yaml.FullLoader)
        lstm_data = data.get('model', {}).get('lstm', {})
        units = lstm_data.get('units')
        layers = lstm_data.get('layers')
        dropout = lstm_data.get('dropout')

        yaml_editor_value = yaml_data

        print("data saved to yaml memory")

        return {'layers':layers, 'units': units, 'dropout':dropout}


    except Exception as e:
        # Handle any errors during YAML parsing
        print(f"Error parsing YAML: {e}")
        return {'layers':'none', 'units': 'none', 'dropout':'none'}
    

################################ fetch columns from csv and write a dcc components ###################
@callback(
    Output('x-col','style'),
    Output('y-col','style'),
    Output(component_id='dropdown-x-column', component_property='options'),
    Output(component_id='dropdown-y-column', component_property='options'),
    Input('upload-data','contents')
)
def update_columns(contents):
        
        x_options = []
        y_options = []
        if contents is not None:
                content_type, content_string = contents[0].split(',')

                # Decode the base64 encoded CSV data
                decoded = base64.b64decode(content_string)
                
                # Read the decoded CSV data using pandas
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

                x_options = [{'label': col, 'value': col} for col in df.columns]
                y_options = [{'label': col, 'value': col} for col in df.columns]

                return {'display':'block'},{'display':'block'},x_options, y_options
        

################################ Stationarity ###################################################

@callback(Output('memory','data'),
          Input('upload-data','contents'))
def save_data(contents):
               if contents is not None:
                        content_type, content_string = contents[0].split(',')

                        decoded = base64.b64decode(content_string)

                        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                        print(df)
                        print('data saved to memory')

                        return {'data':df.to_dict('records')}

# @callback(
#         Output('adf','style'),
#         Output(component_id='stationarity-test', component_property='children'),
#         Input('dropdown-x-column','value'),
#         Input('dropdown-y-column','value'),
#         Input('upload-data','contents'),
#         Input('upload-data','filename')
# )
# def stationarity_check(x,y, contents, filename):
        
#             if contents is not None and x is not None and y is not None:
#                 content_type, content_string = contents[0].split(',')

#                 # Decode the base64 encoded CSV data
#                 decoded = base64.b64decode(content_string)
                
#                 # Read the decoded CSV data using pandas
#                 df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#                 stat_test = adfuller(df[y])
#                 pv = stat_test[1]
#                 if pv <= 0.05: # p-value
                
#                         return {'display':'none'}, dbc.Alert(children=['Test p-value: {:.4f}'.format(pv),html.Br(),'The data is ',html.B(['stationary'], className='alert-bold')], color='success')
#                 else:
#                         return {'display':'none'}, dbc.Alert(children=['Test p-value: {:.4f}'.format(pv),html.Br(),'The data is ',html.B(['not stationary'], className='alert-bold')], color='danger')
#             else:
#                      raise PreventUpdate
            
@callback(
    Output(component_id='d1-dropdown', component_property='disabled'),
    Output(component_id='d1-dropdown', component_property='options'),
    Input('upload-data','contents'),
    Input(component_id='d1-check', component_property='value'),
    Input('dropdown-x-column','value'),
    Input('dropdown-y-column','value'),
)
def dropdown_activation(contents, _check,x,y):
    
    if contents is not None and x is not None and y is not None:
                
                content_type, content_string = contents[0].split(',')

                # Decode the base64 encoded CSV data
                decoded = base64.b64decode(content_string)
                
                # Read the decoded CSV data using pandas
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                _data = df.copy()
                print("hiiiii")
                # Calculate disabled
                if _check:
                    print('The d1 check disabled if triggered')
                    _disabled = False
                else:
                    print('The d1 check enabled else triggered')
                    _disabled = True    
                # Calculate options
                _opts = range(1,int(len(_data[y])/2),1)
                print("returning false and options")
                return _disabled, list(_opts)
    else:
         print("outer else executed")
         raise PreventUpdate
            
@callback(
    Output(component_id='d1-check', component_property='style'),
    Output(component_id='d1-dropdown', component_property='style'),
    Output(component_id='log-check', component_property='style'),
    Output(component_id='fig-transformed', component_property='style'),
    Output(component_id='fig-boxcox', component_property='style'),
    Output(component_id='fig-acf', component_property='style'),
    Output(component_id='fig-pacf', component_property='style'),
    Output(component_id='fig-transformed', component_property='figure'),
    Output(component_id='fig-boxcox', component_property='figure'),
    Output(component_id='fig-acf', component_property='figure'),
    Output(component_id='fig-pacf', component_property='figure'),
    Input(component_id='log-check', component_property='value'),
    Input(component_id='d1-check', component_property='value'),
    Input(component_id='d1-dropdown', component_property='value'),
    Input('upload-data','contents'),
    Input('dropdown-x-column','value'),
    Input('dropdown-y-column','value'),
)
def data_transform(_logtr, _d1check,_d1v, contents, x, y ):

    if contents is not None and x is not None and y is not None:
                content_type, content_string = contents[0].split(',')

                # Decode the base64 encoded CSV data
                decoded = base64.b64decode(content_string)
                
                # Read the decoded CSV data using pandas
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                _data = df.copy()

                try:

                        if _logtr:
                            _min_value = min(_data[y]) # correct for 0 or negative values
                            if _min_value == 0:
                                _data[y] = _data[y] + 0.5
                            elif _min_value < 0:
                                _data[y] = _data[y] + np.abs(_min_value) + 0.5
                            _data[y] = list(np.log(_data[y])) # apply log transformation
                        if _d1check and _d1v:
                            _dvalues = diff(list(_data[y]), lag=_d1v, differences=1)
                            _data = _data.iloc[_d1v:]
                            _data[y] = _dvalues
                
                except:
                       raise PreventUpdate
    
                stat_test = adfuller(df[y])
                pv = stat_test[1]
                if pv <= 0.05: # p-value
                        print(f"stationary data: {pv}")
                        comp =  dbc.Alert(children=['Test p-value: {:.4f}'.format(pv),html.Br(),'The data is ',html.B(['stationary'], className='alert-bold')], color='success')
                else:
                        comp =  dbc.Alert(children=['Test p-value: {:.4f}'.format(pv),html.Br(),'The data is ',html.B(['not stationary'], className='alert-bold')], color='danger')
                        print(f"not a stationary data: {pv}")

                fig_1 = go.Figure(layout=my_figlayout)
                fig_1.add_trace(go.Scatter(x=_data[x], y=_data[y], line=dict()))
                fig_1.update_layout(title='Transformed Data Linechart', xaxis_title='Time', yaxis_title='Values')
                fig_1.update_traces(overwrite=True, line=my_linelayout)


                # Box-Cox plot
                v_ = np.array(_data[y])
                rolling_avg = []; rolling_std = []
                for i in range(0,len(v_),1):
                    rolling_avg.append(v_[:i+1].mean())
                    rolling_std.append(v_[:i+1].std())
                _data['rolling_avg'] = rolling_avg
                _data['rolling_std'] = rolling_std
                fig_2 = go.Figure(layout=my_figlayout)
                _hovertemplate = (
                    "<i>Rolling Avg</i>: %{x:.2f}<br>"+
                    "<i>Rolling Std</i>: %{y:.2f}"+
                    "<extra></extra>")
                fig_2.add_trace(go.Scatter(x=_data['rolling_avg'], y=_data['rolling_std'], mode='markers', marker_size=4, marker_color='#3DED97', hovertemplate=_hovertemplate))
                fig_2.update_layout(title='Box-Cox Plot', xaxis_title='Rolling Average', yaxis_title='Rolling Standard Deviation')

                # ACF, PACF
                fig_3, fig_4 = acf_pacf(_data, y)

                return {'display':'block'}, {'display':'block'}, {'display':'block'}, {'display':'block'}, {'display':'block'}, {'display':'block'}, {'display':'block'},fig_1, fig_2, fig_3, fig_4
    
    elif contents is None:
            _data = pd.read_csv(r'C:\Users\annamalai\Downloads\my_dashboard\data\AirPassengers.csv')
    
    
            fig_1 = go.Figure(layout=my_figlayout)
            fig_1.add_trace(go.Scatter(x=_data['date'], y=_data['Values'], line=dict()))
            fig_1.update_layout(title='Transformed Data Linechart', xaxis_title='Time', yaxis_title='Values')
            fig_1.update_traces(overwrite=True, line=my_linelayout)

            # Box-Cox plot
            v_ = np.array(_data['Values'])
            rolling_avg = []; rolling_std = []
            for i in range(0,len(v_),1):
                rolling_avg.append(v_[:i+1].mean())
                rolling_std.append(v_[:i+1].std())
            _data['rolling_avg'] = rolling_avg
            _data['rolling_std'] = rolling_std
            fig_2 = go.Figure(layout=my_figlayout)
            _hovertemplate = (
                "<i>Rolling Avg</i>: %{x:.2f}<br>"+
                "<i>Rolling Std</i>: %{y:.2f}"+
                "<extra></extra>")
            fig_2.add_trace(go.Scatter(x=_data['rolling_avg'], y=_data['rolling_std'], mode='markers', marker_size=4, marker_color='#3DED97', hovertemplate=_hovertemplate))
            fig_2.update_layout(title='Box-Cox Plot', xaxis_title='Rolling Average', yaxis_title='Rolling Standard Deviation')

            # ACF, PACF
            fig_3, fig_4 = acf_pacf(_data, 'Values')

            return {'display':'none'}, {'display':'none'}, {'display':'none'},{'display':'block'}, {'display':'block'}, {'display':'block'}, {'display':'block'},fig_1, fig_2, fig_3, fig_4
    
    elif contents is not None and (x is None or y is None):
           
           fig_1 = go.Figure(layout=my_figlayout)
           fig_2 = go.Figure(layout=my_figlayout)
           fig_3 = go.Figure(layout=my_figlayout)
           fig_4 = go.Figure(layout=my_figlayout)
           return {'display':'none'}, {'display':'none'}, {'display':'none'},{'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'none'},fig_1, fig_2, fig_3, fig_4

################################# Hyperparameter Tuning #########################################

@callback(
    Output(component_id='p-from', component_property='options'),
    Output(component_id='d-from', component_property='options'),
    Output(component_id='q-from', component_property='options'),
    Output(component_id='sp-from', component_property='options'),
    Output(component_id='sd-from', component_property='options'),
    Output(component_id='sq-from', component_property='options'),
    Output(component_id='sm-from', component_property='options'),
    Input(component_id='train-slider', component_property='value'),
    State('memory','data')
)
def dropdown_opt_from(_trainp,data):

    if bool(data) == False:
              _data = _data_airp.copy()

    else:
             _data = pd.DataFrame(data['data'])

    if _trainp:
        idx_split = round(len(_data['Values']) * int(_trainp)/100) # Split train-test
        _data = _data.iloc[:idx_split+1]
    _opts = range(0,int(len(_data['Values'])/2),1)
    _opts = list(_opts)
    return _opts, _opts, _opts, _opts, _opts, _opts, _opts

@callback(
    Output(component_id='p-to', component_property='options'),
    Input(component_id='p-from', component_property='value'),
    State('memory','data')
)
def dropdown_opt_to(_from, data):

    if bool(data) == False:
              _data = _data_airp.copy()

    else:
             _data = pd.DataFrame(data['data'])

    _data = _data_airp.copy()
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='d-to', component_property='options'),
    Input(component_id='d-from', component_property='value'),
    State('memory','data')
)
def dropdown_opt_to(_from, data):
    if bool(data) == False:
              _data = _data_airp.copy()

    else:
             _data = pd.DataFrame(data['data'])

    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='q-to', component_property='options'),
    Input(component_id='q-from', component_property='value'),
    State('memory','data')
)
def dropdown_opt_to(_from,data):
    if bool(data) == False:
              _data = _data_airp.copy()

    else:
             _data = pd.DataFrame(data['data'])
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='sp-to', component_property='options'),
    Input(component_id='sp-from', component_property='value'),
    State('memory','data')
)
def dropdown_opt_to(_from, data):
    if bool(data) == False:
              _data = _data_airp.copy()

    else:
             _data = pd.DataFrame(data['data'])
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='sd-to', component_property='options'),
    Input(component_id='sd-from', component_property='value'),
    State('memory','data')
)
def dropdown_opt_to(_from,data):
    if bool(data) == False:
              _data = _data_airp.copy()

    else:
             _data = pd.DataFrame(data['data'])
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='sq-to', component_property='options'),
    Input(component_id='sq-from', component_property='value'),
    State('memory','data')
)
def dropdown_opt_to(_from,data):
    if bool(data) == False:
              _data = _data_airp.copy()

    else:
             _data = pd.DataFrame(data['data'])
    
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

@callback(
    Output(component_id='sm-to', component_property='options'),
    Input(component_id='sm-from', component_property='value'),
    State('memory','data')
)
def dropdown_opt_to(_from,data):
    if bool(data) == False:
              _data = _data_airp.copy()

    else:
             _data = pd.DataFrame(data['data'])
    if not _from:
        _from = 0
    _opts = range(int(_from),int(len(_data['Values'])/2),1)
    return list(_opts)

# Grid Search & Show combinations
@callback(
    Output(component_id='comb-nr', component_property='children'),
    Output(component_id='gs-results', component_property='children'),
    Output(component_id='browser-memo', component_property='data'),
    Input(component_id='train-slider', component_property='value'),
    Input(component_id='start-gs', component_property='n_clicks'),
    Input(component_id='p-from', component_property='value'),
    Input(component_id='p-to', component_property='value'),
    Input(component_id='d-from', component_property='value'),
    Input(component_id='d-to', component_property='value'),
    Input(component_id='q-from', component_property='value'),
    Input(component_id='q-to', component_property='value'),
    Input(component_id='sp-from', component_property='value'),
    Input(component_id='sp-to', component_property='value'),
    Input(component_id='sd-from', component_property='value'),
    Input(component_id='sd-to', component_property='value'),
    Input(component_id='sq-from', component_property='value'),
    Input(component_id='sq-to', component_property='value'),
    Input(component_id='sm-from', component_property='value'),
    Input(component_id='sm-to', component_property='value'),
    State(component_id='browser-memo', component_property='data'),
    State('memory','data'),
    prevent_initial_call='initial_duplicate'
)
def grid_search_results(_trainp, _nclicks, p_from,p_to,d_from,d_to,q_from,q_to,sp_from,sp_to,sd_from,sd_to,sq_from,sq_to,sm_from,sm_to,_memo,mydata):
    # Calculate combinations
    _p = list(range(int(p_from), int(p_to)+1, 1))
    _d = list(range(int(d_from), int(d_to)+1, 1))
    _q = list(range(int(q_from), int(q_to)+1, 1))
    _P = list(range(int(sp_from), int(sp_to)+1, 1))
    _D = list(range(int(sd_from), int(sd_to)+1, 1))
    _Q = list(range(int(sq_from), int(sq_to)+1, 1))
    _m = list(range(int(sm_from), int(sm_to)+1, 1))
    _combs = list(product(_p, _d, _q, _P, _D, _Q, _m))
    # Split data
    _datatrain = None; _datatest = None
    if _trainp:
        if bool(mydata) == False:
                   _data = _data_airp.copy()

        else:
                  print("@@@@@@@@@@@@@@@@@@@Data has been loaded from the memory for hyper parameter tuning@@@@@@@@@@@@@@@@@@@")
                  _data = pd.DataFrame(mydata['data'])
        _data['Values'] = list(np.log(_data['Values']))
        idx_split = round(len(_data['Values']) * int(_trainp)/100) # Split train-test
        _datatrain = _data.iloc[:idx_split+1]
        _datatest = _data.iloc[idx_split+1:]
    # Grid search
    if int(_nclicks) > 0 and _datatrain is not None and _datatest is not None:
        _gs_res = sarima_grid_search(_data, _combs)
        _gs_res_tbl = _gs_res.iloc[:10]
        _gs_res_tbl.columns = ['Parameters (p,d,q)(P,D,Q)m', 'AIC Score']
        _gs_res_tbl['AIC Score'] = round(_gs_res_tbl['AIC Score'], 3)
        if 'grid_search_results' in _memo.keys():
            _memo.pop('grid_search_results')
        _memo['grid_search_results'] = _gs_res_tbl.to_dict('records')
    if 'grid_search_results' in _memo.keys():
        _gs_res_tbl = pd.DataFrame(_memo['grid_search_results'])
        tbl_ = dbc.Table.from_dataframe(_gs_res_tbl, index=False, striped=False, bordered=True, hover=True, size='sm')
        title_ = html.P([html.B(['Top-10 models by AIC score'])], className='par')
        _res = [html.Hr([], className = 'hr-footer'), title_, tbl_]
    else:
        _res = None
    return len(_combs), _res, _memo


################################# Hyperparameter Tuning #########################################


################################# Forecast ######################################################

@callback(Output('fig-output8','figure'),
          Input('submit-button-01','n_clicks'),
          State('input-forecasts','value'),
          State('model-memory','data'))
def forecast(n_clicks,prd,data):
        if n_clicks:
            if bool(data):
                path = data['model_path']
                with open(path, "rb") as file:
                      loaded_model = pickle.load(file)

                df = pd.DataFrame(data['test_df'])

                df['date'] = pd.to_datetime(df['date'])

                last_date_2 = df['date'].iloc[-2]
                last_date = df['date'].iloc[-1]

                freq = last_date - last_date_2

                duration = freq

                print(f"The duration.days is {duration.days}")

                if duration.days >= 7 and duration.days % 7 == 0 and duration.days<=27:
                    frequency = 'W'
                elif duration.days >= 28 or duration.days in (28, 30, 31):
                    frequency = 'M'
                else:
                    frequency = 'D'

                print(f"The frequency is {frequency}")
                next_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=prd, freq=frequency)

                if 'exog' not in data:
                                ans = loaded_model.predict(n_periods=prd)

                                forecast_df = pd.DataFrame({
                                                    'date': next_dates,
                                                    'predicted': ans
                                                })
                                print('forecast done')

                                fig = go.Figure(layout=my_figlayout)
                                # fig.add_annotation(
                                #                 x=0.5,
                                #                 y=0.5,
                                #                 xref='paper',
                                #                 yref='paper',
                                #                 text="You are succeeded Beta",
                                #                 showarrow=False,
                                #                 font=dict(color='black', size=16)
                                #             )
                                # return fig
                                print('fig created')
                                fig.add_trace(go.Scatter(
                                      x=df['date'],y=df['Values'],mode='lines',line=my_linelayout,name='Test'))
                                print('trace 1')
                                fig.add_trace(go.Scatter(
                                    x=df['date'],y=df['predicted'],mode='lines',line=test_linelayout,name='Test Predicted'))
                                print('trace 2')
                                fig.add_trace(go.Scatter(
                                    x=forecast_df['date'],y=forecast_df['predicted'],mode='lines',line=forecast_linelayout,name='Forecast'))
                                print('trace 3')
                                fig.update_layout(
                                    title='Actual vs Predicted',xaxis_title='Date',yaxis_title='Value'
                                )
                                print('update graph')
                                return fig
                                
                else:
                        print("!@#$%^&*()(*&^%$#@!@#$%^&*())))(*&^%$#@!@#$%^&*())")
                        exog = list(data['exog'])
                        exog_df = df[exog]
                        
                        nwdf = pd.DataFrame()
                        start = len(exog_df)
                        ans_df = exog_prediction(exog_df,exog,prd)
                        ans_df.reset_index(drop=True, inplace=True)

                        for j in range(0,prd):
                            new_df1 = ans_df.iloc[j,:].to_frame().T
                            ans = loaded_model.predict(n_periods=1,X=np.array(new_df1[-1:]))
                            
                            new_row = new_df1.copy()
                            nwdf = nwdf.append(new_row)
                            
                            nwdf.loc[j, 'Predicted'] = ans[0]

                        new_index = range(start, start+prd)
                        nwdf.set_index(pd.Index(new_index), inplace=True)

                        nwdf['date'] = next_dates

                        print(nwdf.head())


                                
                        fig = go.Figure(layout=my_figlayout)
                        # fig.add_annotation(
                        #                         x=0.5,
                        #                         y=0.5,
                        #                         xref='paper',
                        #                         yref='paper',
                        #                         text="exog is there",
                        #                         showarrow=False,
                        #                         font=dict(color='black', size=16)
                        #                     )
                        print('hiii')
                        fig.add_trace(go.Scatter(
                                      x=df['date'],y=df['Values'],mode='lines',line=my_linelayout,name='Test'))
                        print('hello')
                        fig.add_trace(go.Scatter(
                                    x=df['date'],y=df['predicted'],mode='lines',line=test_linelayout,name='Test Predicted'))
                        print('namaste')
                        fig.add_trace(go.Scatter(
                                    x=nwdf['date'],y=nwdf['Predicted'],mode='lines',line=forecast_linelayout,name='Forecast'))
                        return fig
                       
                


            else:
                    print('data is empty')
                    fig = go.Figure(layout=my_figlayout)
                    return fig

        else:
            print("click not happened yet")    
            fig = go.Figure(layout=my_figlayout)
            return fig
            
        
      

        

################################# Forecast ######################################################

if __name__ == "__main__":
       app.run_server(debug=True, port=4211)


