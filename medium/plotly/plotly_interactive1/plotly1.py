import dash
import dash_core_components as dcc
import dash_html_components as html
from plotly import graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output
import numpy as np

df = pd.read_csv('data/mpg.csv')
df['year'] = np.random.randint(-4,5,len(df))*0.10 + df['model_year']

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='mpg-scatter',
                  figure={
                      'data': [go.Scatter(
                          x=df['year'] + 1900,
                          y=df['mpg'],
                          text=df['name'],
                          #                                   hoverinfo='text',
                          mode='markers'
                      )],
                      'layout': go.Layout(
                          title='MPG vs Model Year',
                          xaxis={'title': 'Model Year'},
                          yaxis={'title': 'MPG'},
                          hovermode='closest'
                      )

                  }
                  )
    ], style={'width': '50%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='mpg-acceleration',
                  figure={
                      'data': [go.Scatter(x=[0, 1],
                                          y=[0, 1],
                                          mode='lines')

                               ],
                      'layout': go.Layout(title='Acceleration',
                                          margin={'l': 0})
                  }
                  )
    ], style={'width': '20%', 'height': '50%', 'display': 'inline-block'}),

    html.Div([
        dcc.Markdown(id='mpg-metrics')
    ], style={'width': '20%', 'display': 'inline-block'})
])


@app.callback(Output('mpg-acceleration','figure'),
             [Input('mpg-scatter','hoverData')])
def callback_graph(hoverData):
    df_index = hoverData['points'][0]['pointIndex']
    figure = {'data':[go.Scatter(x=[0,1],
                                y=[0,60/df.iloc[df_index]['acceleration']],
                                mode = 'lines',)],
             'layout':go.Layout(title=df.iloc[df_index]['name'],
                                xaxis={'visible':False},
                                yaxis={'visible':False,'range':[0,60/df['acceleration'].min()]},
                                margin={'l':0},
                                height = 300
                               )}
    return figure


@app.callback(Output('mpg-metrics','children'),
             [Input('mpg-scatter','hoverData')])
def callback_stats(hoverData):
    df_index = hoverData['points'][0]['pointIndex']
    metrics = """
            {}cc displacement,
            0 to 60mph in {} seconds
            """.format(df.iloc[df_index]['displacement'],
                      df.iloc[df_index]['acceleration'])
    return metrics

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)