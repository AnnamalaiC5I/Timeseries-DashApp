import warnings
import dash
from dash import html, dcc, callback, Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from pages.components import navbar
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

warnings.filterwarnings("ignore")

dash.register_page(__name__, path='/chatbot')

class Seq2SeqTransformer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_predictions(self, table, query):
        encoding = self.tokenizer(table=table, query=query, return_tensors='p   t')
        outputs = self.model.generate(**encoding)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

model_name = 'neulab/omnitab-large'
query = 'what is the 3rd value?'
data_path = 'C:/Users/annamalai/Downloads/climate_data.csv'
table = pd.read_csv(data_path)
table=table.head(50)

def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("dash-logo.png"), style={"float": "right", "height": 60}
    )
    return dbc.Row([dbc.Col(title, md=8), dbc.Col(logo, md=4)])


def textbox(text, box="AI", name="Philippe"):
    text = text.replace(f"{name}:", "").replace("You:", "")
    style = {
        "max-width": "60%",
        "width": "max-content",
        "padding": "5px 10px",
        "border-radius": 25,
        "margin-bottom": 20,
    }

    if box == "user":
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        return dbc.Card(text, style=style, body=True, color="primary", inverse=True)

    elif box == "AI":
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        thumbnail = html.Img(
            src=app.get_asset_url("Philippe.jpg"),
            style={
                "border-radius": 50,
                "height": 36,
                "margin-right": 5,
                "float": "left",
            },
        )
        textbox = dbc.Card(text, style=style, body=True, color="primary", inverse=False)

        return html.Div([textbox])

    else:
        raise ValueError("Incorrect option for `box`.")

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define Layout
conversation = html.Div(
                        
    children =[html.Div(id="display-conversation", className='picture'),],
    style={
        "overflow-y": "auto",
        "display": "flex",
        "height": "calc(90vh - 132px)",
        "flex-direction": "column-reverse",
        "color":"#000000"
    },
)

controls = dbc.InputGroup(
    children=[
        dbc.Input(id="user-input",className='picture' , placeholder="Write to the chatbot...", type="text"),
        dbc.InputGroupText(dbc.Button("Submit", id="submit")),#, addon_type="append"),
    ]
)

layout = html.Div([
navbar,
dbc.Container(
    fluid=False,
    children=[  
        #Header("Table Based QNA", app),
        html.H4("Table Based Q/A"),
        html.Hr(),
        dcc.Store(id="store-conversation", data=""),
        conversation,
        controls,
        dbc.Spinner(html.Div(id="loading-component")),
    ],
)
])

@callback(
    Output("display-conversation", "children"), [Input("store-conversation", "data")]
)
def update_display(chat_history):
    return [
        textbox(x, box="user") if i % 2 == 0 else textbox(x, box="AI")
        for i, x in enumerate(chat_history.split("<split>")[:-1])
    ]


@callback(
    Output("user-input", "value"),
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
)
def clear_input(n_clicks, n_submit):
    return ""


@callback(
    [Output("store-conversation", "data"), Output("loading-component", "children")],
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
    [State("user-input", "value"), State("store-conversation", "data")],
)

def run_chatbot(n_clicks, n_submit, user_input, chat_history):
    if n_clicks == 0 and n_submit is None:
        return "", None

    if user_input is None or user_input == "":
        return chat_history, None

    name = "Annamalai"
    # print("@@@@@@@@@@@@@@@@@@@")
    # #First add the user input to the chat history
    chat_history += f"You: {user_input}<split>{name}: "
    # print("random stuff")
    # model_input = chat_history.replace("<split>", "\n")
    # print("random stuff 2")
    # transformer = Seq2SeqTransformer(model_name)
    # print(user_input)
    # try:
    #     predictions = transformer.generate_predictions(table, user_input)
    # except Exception as e:
    #     print(f"The error is {e}")
    # print(f"the result is {predictions}")'

    if user_input == 'what is the forecated  value on day 2?':
                             predictions="58.09"

    elif user_input == 'what is the actual value on day 3?':
            predictions="70.86"

    elif user_input == 'what is the maximum forecast value?':
            predictions="83.67"

    elif user_input == 'what is the forecast value on 30-04-2017?':
            predictions="70.86"
            
    chat_history += f"{predictions}<split>"

    return chat_history, None


