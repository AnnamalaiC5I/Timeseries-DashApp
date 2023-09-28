import dash
from dash import html
import dash_bootstrap_components as dbc
from pages.components import navbar    
from dash import dcc
from statsmodels.tsa.stattools import adfuller
from dash.exceptions import PreventUpdate
from dash import Input, Output, callback, State
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

disclaimer = html.Div(
    [
        html.P(
            "NOTE: This is done for SARIMA, if required this can be extended to other Models as well.",
            style={"color": "#999999", "font-size": "15px"},
        )
    ], style={'margin-top':'30px'}
)

dash.register_page(__name__, path='/hyper-parameter-tuning')


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



main_container = dbc.Container([
    # title
    
    dbc.Row([
        dbc.Col([
            html.H3(['Hyperparameter Tuning']),
            html.P([html.B(['SARIMA(p,d,q; P,D,Q,m) grid search'])], className='par')
        ], width=12, className='row-titles')
    ]),
    # train-test split
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([html.P(['Select the ',html.B(['train']),' percentage: '], className='par')], width = 4),
        dbc.Col([
            html.Div([
                dcc.Slider(50, 95, 5, value=80, marks=None, tooltip={"placement": "bottom", "always_visible": True}, id='train-slider', persistence=True, persistence_type='session')
            ], className = 'slider-div')
        ], width = 3),
        dbc.Col([], width = 3),
    ]),
    # model params
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Row([
                    dbc.Col([], width = 1),
                    dbc.Col([html.P(['Set ',html.B(['p, d, q']),' parameters range (from, to)'], className='par')], width = 10),
                    dbc.Col([], width = 1),
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['p']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='p-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='p-to')], width=4),
                    dbc.Col([], width=1)
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['d']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='d-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='d-to')], width=4),
                    dbc.Col([], width=1)
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['q']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='q-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='q-to')], width=4),
                    dbc.Col([], width=1)
                ])                
            ], className = 'div-hyperpar')
        ], width = 6, className = 'col-hyperpar'),
        dbc.Col([
            html.Div([
                dbc.Row([
                    dbc.Col([], width = 1),
                    dbc.Col([html.P(['Set ',html.B(['P, D, Q, m']),' seasonal parameters range (from, to)'], className='par')], width = 10),
                    dbc.Col([], width = 1),   
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['P']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sp-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sp-to')], width=4),
                    dbc.Col([], width=1)
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['D']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sd-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sd-to')], width=4),
                    dbc.Col([], width=1)
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['Q']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sq-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sq-to')], width=4),
                    dbc.Col([], width=1)
                ]),
                dbc.Row([
                    dbc.Col([html.P([html.B(['m']),':'], className='par')], width=2),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='from', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sm-from')], width=4),
                    dbc.Col([], width=1),
                    dbc.Col([dcc.Dropdown(options=[], value='0', placeholder='to', clearable=False, searchable=True, persistence=True, persistence_type='memory', id='sm-to')], width=4),
                    dbc.Col([], width=1)
                ])
            ], className = 'div-hyperpar')
        ], width = 6, className = 'col-hyperpar')
    ], style={'margin':'20px 0px 0px 0px'}),
    dbc.Row([
        dbc.Col([], width=3),
        dbc.Col([
            html.P(['Grid Search combinations: ', html.B([], id='comb-nr')], className='par')
        ], width=3),
        dbc.Col([
            html.Button('Start Grid Search', id='start-gs', n_clicks=0, title='The grid search may take several minutes', className='my-button')
        ], width=3, style={'text-align':'left', 'margin':'5px 1px 1px 1px'}),
        dbc.Col([], width=3)
    ]),
    # grid Search results
    dbc.Row([
        dbc.Col([], width = 4),
        dbc.Col([
            dcc.Loading(id='gs-loading', type='circle', children=html.Div(id='gs-results'))
        ], width = 4),
        dbc.Col([], width = 4),
    ]),
    html.Div(
    [
        disclaimer,
    ]
)
])


layout = html.Div([
    navbar,
    main_container
])

