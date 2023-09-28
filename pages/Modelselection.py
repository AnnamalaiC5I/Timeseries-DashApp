import dash
from dash import html, Input, Output, State, callback
from pages.components import navbar
import dash_bootstrap_components as dbc
from dash import dcc
from dash.exceptions import PreventUpdate
from Modules import LSTM
import yaml

dash.register_page(__name__, path='/models')
dropdown_options = [
    {'label': 'ARIMA', 'value': 'arima'},
    {'label': 'SARIMA', 'value': 'sarima'},
    {'label': 'LSTM', 'value': 'lstm'},
    {'label': 'ARIMAX', 'value': 'arimax'},
    {'label': 'SARIMAX', 'value': 'sarimax'},
   
]

yaml_template = """
# YAML Template
model:
      lstm:
         layers: 4
         units: [50,25,12,1]
         dropout: 0.2
"""

layout = html.Div([
    
    navbar,
    #html.H1('Model Selection'),
    html.Div(
        style={'display': 'flex', 'flex-direction': 'row'},
        children=[
        html.Div(
                style={'margin-right': '20px'},
                children=[
                    html.H4('Model'),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=dropdown_options,
                        value='sarima',
                        style={'width': '200px'},
                        clearable=False
                    )
                ]
            ),
        html.Div(
                id='lstm-dropdown-container',
                style={'display': 'none','margin-right':'20px'},
                children=[
                    html.H4('Type'),
                    dcc.Dropdown(
                        id='lstm-type-dropdown',
                        options=[
                            {'label': 'Univariate', 'value': 'univariate'},
                            {'label': 'Multivariate', 'value': 'multivariate'}
                        ],
                        value='univariate',
                        style={'width': '200px'}
                    )
                ]
            ),
        html.Div(
                    id='arima-dropdown-container',
                    style={'display': 'block', 'margin-left': '20px'},
                    children=[
                        html.H4('Train Split Ratio'),
                        dcc.Input(
                            id='split-input',
                            type='number',
                            value=0.7,
                            min=0,
                            max=1,
                            step=0.1,
                            style={
        'width': '200px',
        'background-color': '#042f33',
        'color': 'black',
        'border-radius': '5px',
        'padding': '5px',
        'border': 'none',
        'outline': 'none',
        'box-shadow': '0 0 5px rgba(0, 0, 0, 0.3)'
    }
                        )
                    ]
                ),

        html.Div(
                style={'margin-left': '20px'},
                children=[
                    html.H4('Data Aggregation'),
                    dcc.Dropdown(
                        id='aggregate-dropdown',
                        options=[
                            {'label':'None', 'value':'None'},
                            #{'label': 'Daily', 'value': 'daily'},
                            {'label': 'Weekly', 'value': 'weekly'},
                            {'label': 'Monthly', 'value': 'monthly'},
                            {'label': 'Yearly', 'value': 'yearly'}
                        ],
                        value='monthly',
                        style={'width': '200px'},
                        clearable=False
                    )
                ]
            ),
        html.Div(
                id ='multivariate-div',
                style={'margin-left': '40px', 'display':'none'},
                children=[
                    html.H4('Exog features', style={'margin-left':'20px'}),
                    dcc.Dropdown(
                        id='multivariate-dropdown',
                        options=[],
                        placeholder= 'Select the exog features',
                        style={'width': '400px'},
                        clearable=True,
                        multi=True
                    )
                ],
            ),

        html.Div(
            style={'margin-left':'20px'},
            children=[
            html.Button('Train',style={
                        'background-color': '#042f33',
                        'color': '#000000',
                        'width':'100px',
                        'border-radius': '5px',
                        'padding': '5px',
                        'border': 'none',
                        'outline': 'none',
                        'box-shadow': '0 0 5px rgba(0, 0, 0, 0.3)',
                        'margin-top':'40px'
                    }, id='input-button')
            ]
        )

        ]),
         html.Div(
        id='yaml-editor-container',
        style={'width': '400px', 'margin-top': '5px', 'display': 'none'},
        children=[
            html.H3('YAML Code'),
            dcc.Textarea(
                id='yaml-editor',
                value=yaml_template,
                style={
                    'width': '400px',
                    'height': '200px',
                    'background-color': 'black',
                    'color': 'white'
                }
            )
        ], ),

    html.Div([
          html.Div(
            style={'margin-left':'20px'},
            children=[
            html.Button('Plot',style={
                        'background-color': '#042f33',
                        'color': '#000000',
                        'width':'100px',
                        'border-radius': '5px',
                        'padding': '5px',
                        'border': 'none',
                        'outline': 'none',
                        'box-shadow': '0 0 5px rgba(0, 0, 0, 0.3)',
                        'margin-top':'40px',
                        
                    }, id='plot-button'),
                    dcc.Download(id="download-model-pkl"),
                    html.Button('Download model',style={
                        'background-color': '#042f33',
                        'color': '#000000',
                        'width':'200px',
                        'margin-left':'30px',
                        'border-radius': '5px',
                        'padding': '5px',
                        'border': 'none',
                        'outline': 'none',
                        'box-shadow': '0 0 5px rgba(0, 0, 0, 0.3)',
                        'margin-top':'40px',
                        
                    }, id='download-plot-button')
            ]
        ),
        
    ])   , 

        html.Div([
            dbc.Row([ 
        dbc.Col([
            dcc.Loading(id='p2-21-loading', type='circle', children=dcc.Graph(id='fig-output5',style={'display':'block'}, className='my-graph'))
        ], className='multi-graph'),
    ]),
        ],id='model-selection-graph'),

        html.Div(id='anna')
    
])




