import dash
from dash import html
import dash_bootstrap_components as dbc
from pages.components import navbar


dash.register_page(__name__, path='/')


# layout = html.Div([
#     navbar,    
#     html.H1('This is our Home Page')

# ]) 

layout = html.Div([
    navbar,
dbc.Container([
    # title
    dbc.Row([
        dbc.Col([],width=5),
        dbc.Col([
            html.H3(['Welcome Course5i !']),
            html.P([html.B(['App Overview'])], className='par', style={
                #'margin-left':'65px'
            })
        ], width=3, className='row-titles'),
        dbc.Col([],width=4)
    ]),
    # Guidelines
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            html.P([html.B('Introduction'),html.Br(),
                    'This app allows you to upload your own Timeseries data and perform forecasting on top of it. You can also perform stationarity check, Exploratory Data Analysis, Anomaly Detection on the data.'], className='guide'),

            html.P([html.B('1) Upload your Dataset'),html.Br(),
                    'In this step you need to upload your csv and then you will be able to visualize the timeseries data distribution. You can select the X and Y column in your dataset and the distribution will be plotted. More over it also generates a sample high level report of the dataset about its missing values etc.'], className='guide'),

            html.P([html.B('2) Stationarity Check'),html.Br(),
                    'The tools available on the page are: log and differencing, the Box-Cox plot and the A. Dickey Fuller test.',html.Br(),
                    'Once the data is stationary, check the ACF and PACF plots for suitable model parameters.'], className='guide'),

            html.P([html.B('3) Exploratory Data Analysis'),html.Br(),
                    'This Page contains a basic Exploratory Data Analysis on the Timeseries Data that was uploaded by the user. This may also include a DeepEye Output for the same. '], className='guide'),
            
            html.P([html.B('4) Anomaly Detection'),html.Br(),
                    'This step includes Normalization, Handling Missing Values, Outlier Detection, Outlier Imputation and Visualization of these Outliers. '], className='guide'),

            html.P([html.B('5) Select Your Model'),html.Br(),
                    'Select your model from the choices - LSTM, Arima, Sarima, Arimax, Sarimax'
                    'Choose the train-test split and provide from-to ranges for any parameter.'
                    'The seasonality component of the model can be excluded by leaving all right-hand parameters to 0.',html.Br(),
                    'The 10 top-performer models (according to the AIC score), are shown.'], className='guide')
        ], width = 8),
        dbc.Col([], width = 2)
    ])
])
])