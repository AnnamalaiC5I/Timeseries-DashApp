import dash
from dash import html, Input, Output, State, callback
from pages.components import navbar
import dash_bootstrap_components as dbc
from dash import dcc
from dash.exceptions import PreventUpdate


dash.register_page(__name__, path='/forecast')


layout = html.Div([
    navbar,
    html.Div(
        style={'display': 'flex', 'flex-direction': 'row'},
        children = [

            html.Div(
                style={'margin-right': '20px'},
                children=[
                    html.H4('Number of Forecasts'),
                    dcc.Input(id='input-forecasts',style={'width':'200px','background-color':'#042f33','borderRadius': '5px',
                               'padding': '5px',
                               'border': 'none',
                               'outline': 'none',
                               'display':'block'}, type='number', value=5, min=1)
                ]
            ),

            html.Div(
                style={'margin-left':'20px'},
                children = [
                   html.Button('Submit', id='submit-button-01',style={
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
                    }, n_clicks=0),
            ])

        ]
        ),
    
  
    html.Div([
            dbc.Row([ 
        dbc.Col([
            dcc.Loading(id='p2-290-loading', type='circle', children=dcc.Graph(id='fig-output8',style={'display':'block'}, className='my-graph'))
        ], className='multi-graph'),
    ]),
        ],id='forecast-graph')
])