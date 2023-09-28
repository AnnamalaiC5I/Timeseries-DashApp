import dash
from dash import html, dcc, Input, Output, callback, State
from pages.components import navbar
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/anomaly')

layout = html.Div([
    navbar,

    html.Div(
        style={'display': 'flex', 'flex-direction': 'row'},
        children = [

        html.Div(
            style={'margin-top':'40px'},
            children=[
                    dcc.Upload(
                    id='anomaly-upload1',
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
                id = 'anomaly-dropdown-div1',
                style={'margin-right': '20px', 'margin-top':'30px'},
                children=[
                    html.H4(id='feature-title',style={'display':'none'},children=['Feature']),
                    dcc.Dropdown(
                        id='anomaly-dropdown1',
                        style={'width':'200px','display':'none'},
                        clearable=False
                    )
                ]
        ),
        html.Div(
            id = 'anomaly-cont-rate',
            style={'margin-right':'20px','margin-top':'30px'},
           children= [
                    html.H4(id='crate-title',style={'display':'none'},children=['Contamination rate']),
                    dcc.Input(
                        style={'width':'200px','background-color':'#042f33','borderRadius': '5px',
                               'padding': '5px',
                               'border': 'none',
                               'outline': 'none',
                               'display':'none'},
                        id='anomaly-number-input',
                        type='number',
                        min=0,
                        max=1,
                        step=0.1,
                        value=0.5
                    ),
                    
                ]
                ),

        html.Div(
            style={'margin-left':'20px', 'margin-top':'30px'}, #-10px
            children=[
            html.Button('Handle Outlier',style={
                        'background-color': '#042f33',
                        'color': '#000000',
                        'width':'150px',
                        'border-radius': '5px',
                        'padding': '5px',
                        'border': 'none',
                        'display':'none',
                        'outline': 'none',
                        'box-shadow': '0 0 5px rgba(0, 0, 0, 0.3)',
                        'margin-top':'40px'
                    }, id='handle-outlier')
            ]
        ),

        html.Div(
            style={'margin-left':'20px','margin-top':'30px'}, #-10px
            children=[
            html.Button('Plot',style={
                        'background-color': '#042f33',
                        'color': '#000000',
                        'width':'100px',
                        'border-radius': '5px',
                        'padding': '5px',
                        'border': 'none',
                        'display':'none',
                        'outline': 'none',
                        'box-shadow': '0 0 5px rgba(0, 0, 0, 0.3)',
                        'margin-top':'40px'
                    }, id='handle-outlier-plot')
            ]
        ),
        html.Div(
            style={'margin-left':'20px','margin-top':'30px'}, #-10px
            children=[
            dcc.Download(id="download-dataframe-csv"),
            html.Button('Download',style={
                        'background-color': '#042f33',
                        'color': '#000000',
                        'width':'100px',
                        'border-radius': '5px',
                        'padding': '5px',
                        'border': 'none',
                        'display':'none',
                        'outline': 'none',
                        'box-shadow': '0 0 5px rgba(0, 0, 0, 0.3)',
                        'margin-top':'40px'
                    }, id='handle-outlier-download')
            ]
        )

        ]
    ),

    html.Div([
            dbc.Row([ 
        dbc.Col([
            dcc.Loading(id='p2-25-loading', type='circle', children=dcc.Graph(id='fig-output6',style={'display':'block'}, className='my-graph'))
        ], className='multi-graph'),
    ]),
        ],id='anomaly-detection-graph')
   
    
    
])

