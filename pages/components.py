import dash_bootstrap_components as dbc

file_path = 'C:/Users/annamalai/Downloads/my_dashboard/SWEETVIZ_REPORT.html'

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/",style={
            'color':'white',
        })),
        dbc.NavItem(dbc.NavLink("EDA", href="/Exploratiry-data-analysis",style={
            'color':'white'
        })),
        dbc.NavItem(dbc.NavLink("Outlier Detection", href="/anomaly",style={
            'color':'white'   
        })),
        dbc.NavItem(dbc.NavLink("Stationarity Check", href="/stationarity",style={
            'color':'white'
        })),
        dbc.NavItem(dbc.NavLink("Parameter Tuning", href="/hyper-parameter-tuning",style={
            'color':'white'
        })),
        dbc.NavItem(dbc.NavLink("Models", href="/models",style={
            'color':'white'
        })),
        dbc.NavItem(dbc.NavLink("Model Tracking", target="_blank", href="http://127.0.0.1:5000/",style={
            'color':'white'
        })),
        dbc.NavItem(dbc.NavLink("Forecast", href="/forecast",style={
            'color':'white'
        })),
        dbc.NavItem(dbc.NavLink("Q/A", href="/chatbot",style={
            'color':'white'
        })),
    ],
    brand="Course5 Tuner",
    
    color="black",
    dark=True,
    
    style={
        'background-color':'black',
        'color':'white',
        
    }
)

