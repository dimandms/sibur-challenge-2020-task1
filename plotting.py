import plotly.graph_objects as go
from constants import TARGET_COLUMNS


def plot_submition(submission):
    fig = go.Figure()

    for target in TARGET_COLUMNS:
        fig.add_trace(go.Scatter(
            x=submission['timestamp'], y=submission[target], name=target))

    fig.update_layout(
        width=1200,
        height=500,
    )

    fig.show()
