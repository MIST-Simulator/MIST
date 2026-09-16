import matplotlib.pyplot as plt
from typing import List
from GenA.Request.request import Request
import plotly.graph_objects as go
from typing import List

def plot_request_queue(requests: List[Request]):
    # Using sorted() function
    sorted_requests = sorted(requests, key=lambda obj: obj.request_id)

    fig = go.Figure()

    y_labels = [f"Request {i+1}" for i in range(len(sorted_requests))]

    for i, request in enumerate(sorted_requests):
        y = i

        # Queue bar
        fig.add_trace(go.Bar(
            x=[request.metrics.duration_in_queue],
            y=[y],
            orientation='h',
            name='Queue' if i == 0 else None,
            marker_color='blue',
            opacity=0.8,
            base=request.metrics.arrival_time,
            showlegend=i == 0,
            text=[f"Queue: {request.metrics.duration_in_queue} ms"],
            hoverinfo='text'
        ))

        # Prefill bar
        fig.add_trace(go.Bar(
            x=[request.data[0].finished_time - request.metrics.first_scheduled_time],
            y=[y],
            orientation='h',
            name='Prefill' if i == 0 else None,
            marker_color='green',
            opacity=0.8,
            base=request.metrics.first_scheduled_time,
            showlegend=i == 0,
            text=[f"Prefill: {request.data[0].finished_time - request.metrics.first_scheduled_time} ms"],
            hoverinfo='text'
        ))

        # Dec bars
        for j in range(1, len(request.data)):
            fig.add_trace(go.Bar(
                x=[request.data[j].finished_time - request.data[j].scheduled_time],
                y=[y],
                orientation='h',
                name='Dec' if i == 0 and j == 1 else None,
                marker_color='red',
                opacity=0.8,
                base=request.data[j].scheduled_time,
                showlegend=i == 0 and j == 1,
                text=[f"Dec {j}: {request.data[j].finished_time - request.data[j].scheduled_time} ms"],
                hoverinfo='text',
            ))

    fig.update_layout(
        barmode='stack',
        title='Request Timeline',
        xaxis_title='Time (ms)',
        yaxis_title='Requests',
        yaxis=dict(
            tickvals=list(range(len(sorted_requests))),
            ticktext=y_labels
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400 + (len(sorted_requests) * 30),  # Adjust height based on number of requests
        width=1000
    )

    fig.show()