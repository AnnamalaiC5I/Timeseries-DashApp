import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
from itertools import product
from pages.components import navbar
import pandas as pd

dash.register_page(__name__, path='/Exploratiry-data-analysis')

 # Generate the HTML report
layout = html.Div([
    navbar,
    html.Div([
        #html.H3('EDA Page'),

        html.Div(
            style={'display':'flex', 'flex-direction':'row'},
            children=[
  
        html.Div(
            style={'margin-top':'40px'},
            children=[
                    dcc.Upload(
                    id='eda-upload',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '300px',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False),
            ]
        ),
        html.Div(
                id = 'eda-dropdown-div',
                style={'margin-right': '30px', 'margin-top':'30px'},
                children=[
                    html.H4(id='xfeature-title',style={'display':'block'},children=['X Axis']),
                    dcc.Dropdown(
                        id='x-col-dropdown',
                        style={'width':'200px','display':'block'},
                        clearable=False
                    )
                ]
        ),
        html.Div(
                id = 'eda-dropdown-div1',
                style={'margin-right': '20px', 'margin-top':'30px'},
                children=[
                    html.H4(id='yfeature-title',style={'display':'block'},children=['Y Axis']),
                    dcc.Dropdown(
                        id='y-col-dropdown',
                        style={'width':'200px','display':'block'},
                        clearable=False
                    )
                ]
        ),
        html.Div(
                id = 'eda-dropdown-div2',
                style={'margin-right': '20px', 'margin-top':'30px'},
                children=[
                    html.H4(id='plot-type-title',style={'display':'block'},children=['Plot Type']),
                    dcc.Dropdown(
                        id='plot-col-dropdown',
                        options=[
                            {'label':'Scatter Plot', 'value':'scatter'},
                            {'label':'Area Plot', 'value':'area'},
                            {'label': 'Line Plot', 'value': 'line'},
                            #{'label': 'Bar Plot', 'value': 'bar'},
                            #{'label': 'Yearly', 'value': 'yearly'}
                        ],
                        value='scatter',
                        style={'width':'200px','display':'block'},
                        clearable=False
                    )
                ]
        ),
        html.Div(
            style={'margin-left':'10px', 'margin-top':'30px'}, #-10px
            children=[
            html.Button('Plot',style={
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
                    }, id='plot-eda-page-button')
            ]
        ),
        html.Div(
            style={'margin-left':'10px', 'margin-top':'30px'}, #-10px
            children=[
            html.Button('Sweetviz report',style={
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
                    }, id='plot-sweetviz-button')
            ]
        ),

        ])
    ]),
    html.Div([
            dbc.Row([ 
        dbc.Col([
            dcc.Loading(id='project-k-loading', type='circle', children=dcc.Graph(id='fig-output124',style={'display':'block'}, className='my-graph'))
        ], className='multi-graph'),
    ]),
        ],id='eda-graph')
])


