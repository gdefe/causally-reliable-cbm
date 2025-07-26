# plotly imports
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default="browser"    # or 'browser'
pio.templates.default="plotly_white"

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import torch
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
import plotly.express as px
from plotly.subplots import make_subplots

# colors in rgb format
# 'cem' is red, 'blackbox' is grey, 'c2bm' is green, 'cbm_linear' is light_blue, 'cbm_mlp' is darker_blue
# 'scbm' in orange
colors = {'cem': '255, 0, 0', 
          'blackbox': '128, 128, 128', 
          'cbm_linear': '64, 224, 208',
          'cbm_mlp': '0, 150, 255',
          'scbm': '255, 165, 0',
          'c2bm': '0, 128, 0'}

legend = {'blackbox': 'OpaqNN',
          'cem': 'CEM',
          'cbm_linear': 'CBM₊ₗᵢₙ',
          'cbm_mlp': 'CBM₊ₘₗₚ',
          'scbm': 'SCBM',
          'c2bm': 'C²BM'
}
markers = {'blackbox': 'circle-open',
           'cem': 'square-open',
           'cbm_linear': 'diamond-open',
           'cbm_mlp': 'cross-open',
           'scbm': 'pentagon-open',
           'c2bm': 'triangle-up-open'
    }
 
title_font_size = 80 # 80
axis_title_font_size = 65 # 65
axis_title_font_size_2 = 40 # 40
legend_font_size = 40  # 22
tick_font_size = 53   # 53

def plot_difference(y):
    df = pd.DataFrame(y)
    df = df.sort_index()
    max_per_step = df.iloc[:,:-1].max(axis=1)
    df['pdiff'] = (df.loc[:,'c2bm'] - max_per_step) #*100. / max_per_step
    df['color'] = df['pdiff'].apply(lambda v: 'green' if v > 0 else 'red')
    fig = go.Bar(x=df.index,
                 y=df['pdiff'],
                 marker_color=df['color'])
    return fig

def plot_intervention(y, y_std, title):
    # plotly histogram: concept names on x-axis, delta y accuracy on y-axis for each model
    model_names = list(y.keys())
    label_names = list(y[model_names[-1]].keys())

    # Create figure with secondary y-axis
    fig = go.Figure()
    for model_name in model_names:
        if y[model_name]:
            y_delta = list(y[model_name].values())
            y_std_data = list(y_std[model_name].values())
            fig.add_trace(go.Bar(x=list(range(len(label_names))), 
                                 y=y_delta, 
                                 error_y=dict(type='data', array=y_std_data),
                                 name=legend[model_name],
                                 marker_color='rgb('+colors[model_name]+')'
                                 )
                        )
    # show all ticks
    # fig.update_xaxes(tickmode='array', tickvals=list(label_names), ticktext=list(label_names))
    fig.update_layout(title=title,
                      title_x=0.5,
                      title_y=0.99,
                      title_font_size=title_font_size,
                      xaxis_title='Intervened concept',
                      xaxis_title_font_size=axis_title_font_size,
                      yaxis_title="Rel. improv. (%) on task accuracy",
                      yaxis_title_font_size=axis_title_font_size)
    # place legend at the bottom of the plot
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.9,
        xanchor="right",
        x=1
    ))
    # text size and tiks size
    fig.update_layout(font=dict(size=legend_font_size))
    fig.update_layout(showlegend=False)
    fig.update_layout(legend= {'itemsizing': 'constant'})
    fig.update_xaxes(tickfont=dict(size=tick_font_size))
    fig.update_yaxes(tickfont=dict(size=tick_font_size))
    # fig.show()
    return fig


def plot_level_intervention(y, y_std, y_label, title, add_diff=False):
    # plotly histogram: concept names on x-axis, delta y accuracy on y-axis for each model
    model_names = list(y.keys())
    
    if add_diff:
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True,
                            vertical_spacing=0.01,
                            row_heights=[0.2, 0.8],
                            # subplot_titles=('', title),
                            )
    else:
        fig = go.Figure()
    y_delta = {}
    for model_name in model_names:
        if y[model_name]:
            # reordering the x-axis labels
            y_delta = {i: y[model_name][i] for i in range(len(y[model_name]))}
            y_std_data = {i: y_std[model_name][i] for i in range(len(y_std[model_name]))}
            x = list(y_delta.keys())
            # fill between the upper and lower bounds
            y_upper = list({i: y_delta[i] + y_std_data[i] for i in y_delta.keys()}.values())
            y_lower = list({i: y_delta[i] - y_std_data[i] for i in y_delta.keys()}.values())
            fig.add_trace(go.Scatter(x=x+x[::-1], # x, then x reversed
                                     y=y_upper+y_lower[::-1], # upper, then lower reversed
                                     fill='toself',
                                     line=dict(color='rgba('+colors[model_name]+',0.)'),
                                     fillcolor='rgba('+colors[model_name]+'0.05)',
                                     hoverinfo="skip",
                                     opacity=0.6,
                                     showlegend=False),
                        row=2 if add_diff else None, col=1 if add_diff else None
                        )
            # line plot with dots at the points
            fig.add_trace(go.Scatter(x=x, 
                                     y=list(y_delta.values()), 
                                     mode='lines+markers', 
                                     name=legend[model_name],
                                     marker=dict(size=25,
                                                 symbol=markers[model_name],
                                                 line=dict(width=2,
                                                           color='rgb('+colors[model_name]+')')
                                                 ), 
                                     line=dict(color='rgb('+colors[model_name]+')',
                                               width=4),
                                     ),
                        row=2 if add_diff else None, col=1 if add_diff else None
                        )
    if add_diff:
        fig_diff = plot_difference(y)
        fig.add_trace(fig_diff, row=1, col=1)

    # # Update yaxis properties
    # fig.update_yaxes(title_text="yaxis 1 title", row=2, col=1)
    # fig.update_yaxes(title_text="yaxis 2 title", range=[40, 80], row=1, col=2)
    # fig.update_yaxes(title_text="yaxis 3 title", showgrid=False, row=2, col=1)
    # fig.update_yaxes(title_text="yaxis 4 title", row=2, col=2)

    # show all ticks
    # fig.update_xaxes(tickmode='array', tickvals=list(y_delta.keys()), ticktext=list(y_delta.keys()))
    # place legend at the bottom of the plot
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.9,
        xanchor="right",
        x=1
    ))
    # hide legend
    fig.update_layout(showlegend=False)
    fig.update_layout(title=title,
                    #   title_x=0.5,
                    #   title_y=0.97,
                      title_x=0.15,
                      title_y=0.7,
                      title_font_size=title_font_size)
    fig.update_layout(font=dict(size=legend_font_size))

    if add_diff:
        fig.update_xaxes(tickfont=dict(size=tick_font_size), row=1 if add_diff else None, col=1 if add_diff else None,
                        title_text="")
    fig.update_xaxes(tickfont=dict(size=tick_font_size), row=2 if add_diff else None, col=1 if add_diff else None,
                     showline=True, linewidth=2, linecolor='black', mirror=True,
                     title_font_size=axis_title_font_size,
                     title_text="Number of intervened concepts")

    
    if add_diff:
        fig.update_yaxes(tickfont=dict(size=axis_title_font_size_2), row=1 if add_diff else None, col=1 if add_diff else None,
                        title_font_size=axis_title_font_size_2,
                        title_text='Δ Improv.')
    fig.update_yaxes(tickfont=dict(size=tick_font_size), row=2 if add_diff else None, col=1 if add_diff else None,
                     showline=True, linewidth=2, linecolor='black', mirror=True,
                     title_font_size=axis_title_font_size,
                     title_text=y_label)


    fig.update_xaxes(showgrid=True, row=2 if add_diff else None, col=1 if add_diff else None)
    fig.update_yaxes(showgrid=True, row=2 if add_diff else None, col=1 if add_diff else None)

    return fig


def convert_adjMatrix_to_causallearnGraph(adj_matrix, node_labels):
    """
    Convert an adjacency matrix to a causallearn GeneralGraph.

    Args:
    adj_matrix (np.ndarray): An adjacency matrix representing a causal graph.
    node_labels (list): A list of node labels.

    Returns:
    GeneralGraph: A GeneralGraph object representing the graph.
    edges: a list of Edges object representing the edges in the graph.
    """
    # Initialize the GeneralGraph with nodes
    nodes = []
    for i in range(len(node_labels)):
        nodes.append(GraphNode(node_labels[i]))

    g_causal = GeneralGraph(nodes)
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):  # Only check the upper triangle to avoid redundant checks
            if adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                # Add undirected edge
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL)
                edge.properties.append(Edge.Property.dd)
                g_causal.add_edge(edge)
                edges.append(edge)
            elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
                # Add directed edge i → j
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
                edge.properties.append(Edge.Property.dd)
                g_causal.add_edge(edge)
                edges.append(edge)
            elif adj_matrix[i, j] == 0 and adj_matrix[j, i] == 1:
                # Add directed edge j → i
                edge = Edge(nodes[j], nodes[i], Endpoint.TAIL, Endpoint.ARROW)
                edge.properties.append(Edge.Property.dd)
                g_causal.add_edge(edge)
                edges.append(edge)
            elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == 1:
                # Add bidirected edge j <-> i
                edge = Edge(nodes[j], nodes[i], Endpoint.ARROW, Endpoint.ARROW)
                edge.properties.append(Edge.Property.dd)
                g_causal.add_edge(edge)
                edges.append(edge)

    return g_causal, edges

def maybe_plot_graph(graph, plot_name):
    """
    Plot a graph from a graph dictionary.
    Attributes:
        graph (Dataframe): A dictionary containing the graph information.
        plot_name (str): The name of the plot.

    Returns:
        None
    """
    if graph is not None:
        g = torch.Tensor(graph.values)
        labels = list(graph.index)
        g, edges_g = convert_adjMatrix_to_causallearnGraph(g, labels)
        # Convert graph to PyDot format
        pyd = GraphUtils.to_pydot(g, edges_g)
        
        # Create and read the PNG image of the graph
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        
        # Save the image as a PNG file without displaying it
        plt.imsave(f'{plot_name}.png', img)

