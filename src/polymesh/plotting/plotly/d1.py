# -*- coding: utf-8 -*-
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from ...explode import explode_mesh


__all__ = ['plot_lines_3d', 'stack_lines_3d', 'scatter_points_3d']


def scatter_lines_3d(fig, coords, topo, *args, **kwargs):
    # this only works if all the lines form a continuous path
    # in a way, that the first node of the lext line always
    # coincides with the end node of the previous line
    X, _ = explode_mesh(coords, topo)
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    scatter_cells = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(
            color='black',
            width=4
        ),
        showlegend=False
    )
    scatter_cells.update(hoverinfo='skip')
    fig.add_trace(scatter_cells)


def stack_lines_3d(fig, coords, topo, *args, **kwargs):
    X, _ = explode_mesh(coords, topo)
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    def _stack_line_3d(i):
        scatter_cells = go.Scatter3d(
            x=x[2*i: 2*(i+1)],
            y=y[2*i: 2*(i+1)],
            z=z[2*i: 2*(i+1)],
            mode='lines',
            line=dict(
                color='black',
                width=4
            ),
            showlegend=False
        )
        scatter_cells.update(hoverinfo='skip')
        fig.add_trace(scatter_cells)
    list(map(_stack_line_3d, range(topo.shape[0])))


def scatter_points_3d(coords, *args, scalars=None, markersize=1, 
                      scalar_labels=None, **kwargs):
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    
    dfdata = {'x': x, 'y': y, 'z': z,
              'size': np.full(len(x), markersize),
              'symbol': np.full(len(x), 4),
              'id': np.arange(1, len(x) + 1)}
    
    if scalars is not None and scalar_labels is None:
        nData = scalars.shape[1]
        scalar_labels = ['#{}'.format(i+1) for i in range(nData)]
    
    if scalar_labels is not None:
        assert len(scalar_labels) == scalars.shape[1]
        hover_data = scalar_labels
        sdata = {scalar_labels[i] : scalars[:, i] for i in range(len(scalar_labels))}
        dfdata.update(sdata)
    else:
        hover_data = ["x", "y", "z"]
            
    custom_data = scalar_labels if scalars is not None else None
    
    df = pd.DataFrame.from_dict(dfdata)
    
    fig = px.scatter_3d(
            df,
            x='x', y='y', z='z',
            hover_name="id",
            hover_data=hover_data,
            size='size',
            text="id",
            custom_data=custom_data,
        )
            
    if scalars is not None:
        tmpl = lambda i : '{' + 'customdata[{}]:.4e'.format(i) + '}'
        lbl = lambda i : scalar_labels[i]
        fnc = lambda i : "{label}: %{index}".format(label=lbl(i), index=tmpl(i))
        labels = [fnc(i) for i in range(len(scalar_labels))]
        fig.update_traces(hovertemplate="<br>".join(labels))
    else:
        fig.update_traces(
            hovertemplate="<br>".join([
                "x: %{x}",
                "y: %{y}",
                "z: %{z}",
            ]))
        
    return fig


def plot_lines_3d(coords, topo, *args, scalars=None, fig=None, **kwargs):

    n2 = topo[:, [0, -1]].max() + 1
    _scalars = scalars[:n2] if scalars is not None else None
    
    if fig is None:
        fig = scatter_points_3d(coords[:n2], *args, scalars=_scalars, **kwargs)
    else:
        scatter_points_3d(coords[:n2], *args, scalars=_scalars, fig=fig, **kwargs)
                
    for i, _ in enumerate(fig.data):
        #fig.data[i].marker.symbol = 'diamond'
        fig.data[i].marker.size = 5

    stack_lines_3d(fig, coords, topo)

    fig.update_layout(
        template="plotly",
        autosize=True,
        # width=720,
        # height=250,
        margin=dict(l=1, r=1, b=1, t=1, pad=0),
        scene=dict(
            aspectmode='data',
            #xaxis = dict(nticks=5, range=[xmin - delta, xmax + delta],),
            #yaxis = dict(nticks=5, range=[ymin - delta, ymax + delta],),
            #zaxis = dict(nticks=5, range=[zmin - delta, zmax + delta],),
        )
    )

    return fig
