import json
import plotly.graph_objects as go
with open("data/fig.json", "r") as f:
    fig = go.Figure(json.load(f))

import dash
import dash_html_components as html
import dash_core_components as dcc

app = dash.Dash(__name__)

app.layout = html.Div([dcc.Graph(id="main-graph", figure=fig)])

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)
