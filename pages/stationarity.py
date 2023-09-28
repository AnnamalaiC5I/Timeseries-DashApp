import dash
from dash import html
import dash_bootstrap_components as dbc
from pages.components import navbar    
from dash import dcc
from statsmodels.tsa.stattools import adfuller
from dash.exceptions import PreventUpdate
from dash import Input, Output, callback
import base64
import pandas as pd
import io
import plotly.graph_objects as go
import numpy as np
import warnings
from pmdarima.utils import diff

from assets.fig_layout import my_figlayout, my_linelayout
from assets.acf_pacf_plots import acf_pacf

warnings.filterwarnings("ignore")



dash.register_page(__name__, path='/stationarity')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

layout = html.Div([
     navbar,
     
# Log and differencing
dbc.Row([
        dbc.Col([],width=4),
        dbc.Col([html.H3(['Transform dataset to make it Stationary'])], className='row-titles',width=6),
        dbc.Col([],width=2),
    ]),


dbc.Row([
    dbc.Col([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '350px',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    )
    ], width=5),

#     dbc.Col([
#     html.Div(id='output-data-upload', style={'margin-top':'30px'}),
# ], width=2),

     dbc.Col([
             dbc.Row([
                 dbc.Col([html.P(['Select x column:'], className='par')]) ,
                 dbc.Col([dcc.Dropdown(id='dropdown-x-column')])   
             ],style={'margin-top':'30px'}),
     ], width=3, id='x-col', style={'display':'none','margin-right':'100px'}),

     dbc.Col([
             dbc.Row([
                 dbc.Col([html.P(['Select y column:'], className='par')]) ,
                 dbc.Col([dcc.Dropdown(id='dropdown-y-column')])   
             ],style={'margin-top':'30px'}),
     ], width=3, id='y-col', style={'display':'none'})
]),

    # # Transformations
    dbc.Row([
        dbc.Col([], width = 4),
        dbc.Col([
            dcc.Checklist(['1) Apply log'], persistence=True, persistence_type='session', id='log-check',style={'display':'none'})
        ], width = 2),
        dbc.Col([
            dcc.Checklist(['2) Apply difference'], persistence=True, persistence_type='session', 
                          id='d1-check', style={'display':'none'}),
            dcc.Dropdown(options=[], value='', clearable=True, disabled=True, searchable=True, placeholder='Choose lag', persistence=True, persistence_type='session', id='d1-dropdown', style={'display':'none'})
        ], width = 2),
        dbc.Col([], width = 4),
        
    ], className='row-content'),
    # Augmented Dickey-Fuller test
    # dbc.Row([
    #     dbc.Col([], width = 3),
    #     dbc.Col([html.P(['Augmented Dickey-Fuller test: '], id='adf',className='par', style={'display':'none'})], width = 2),
    #     dbc.Col([
    #         dcc.Loading(id='p2-1-loading', type='circle', children=html.Div([], id = 'stationarity-test'))
    #     ], width = 4),
    #     dbc.Col([], width = 3)
    # ], style={
    #         'margin-top':'60px'
    # }),

    dbc.Row([ 
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-transformed',style={'display':'none'}, className='my-graph'))
        ], width=6, className='multi-graph'),
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-acf',style={'display':'none'}, className='my-graph'))
        ], width=6, className='multi-graph')
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-boxcox',style={'display':'none'}, className='my-graph'))
        ], width = 6, className='multi-graph'),
        dbc.Col([
            dcc.Loading(id='p2-2-loading', type='circle', children=dcc.Graph(id='fig-pacf',style={'display':'none'}, className='my-graph'))
        ], width=6, className='multi-graph')
    ])
])

# @callback(
    
#     Output('output-data-upload','children'),
#     Input('upload-data','filename')
# )
# def upload_file(filename):
#          return filename

           

    
    

