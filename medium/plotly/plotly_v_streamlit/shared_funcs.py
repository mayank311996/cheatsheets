def load_fig():
    import json
    import plotly.graph_objects as go
    with open("data/fig.json", "r") as f:
        fig = go.Figure(json.load(f))
    return fig