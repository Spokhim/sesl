# This is to store functions for visualisation and plotting.
import numpy as np
import matplotlib.pyplot as plt
from lapy.plot import _get_color_levels, _map_z2color
import plotly
import plotly.graph_objs as go

def _get_colorscale(vmin, vmax):
    """Put together a colorscale map depending on the range of v-values.  Adjusted from Lapy's plot module.

    Parameters
    ----------
    vmin : float
        Minimum value.
    vmax : float
        Maximum value.

    Returns
    -------
    colorscale: array_like of shape (2,2)
        Colorscale map.
    """
    if vmin > vmax:
        raise ValueError("incorrect relation between vmin and vmax")
    # color definitions
    posstart = "rgb(253, 230, 210)" #"rgb(253, 210, 199)"
    posstop = "rgb(178, 24, 43)"
    negstart = "rgb(33, 102, 172)"
    negstop = "rgb(220, 240, 240)" # "rgb(209, 229, 240)"
    zcolor = "rgb(247, 247, 247)"

    if vmin > 0:
        # only positive values
        colorscale = [[0, posstart], [1, posstop]]
    elif vmax < 0:
        # only negative values
        colorscale = [[0, negstart], [1, negstop]]
    else:
        # both pos and negative (here extra color for values around zero)
        zz = -vmin / (vmax - vmin)
        eps = 0.000000001
        zero = 0.001
        if zz < (eps + zero):
            # only very few negative values (map to zero color)
            colorscale = [
                [0, zcolor],
                [zero, zcolor],
                [zero + eps, posstart],
                [1, posstop],
            ]
        elif zz > (1.0 - eps - zero):
            # only very few positive values (map to zero color)
            colorscale = [
                [0, negstart],
                [1 - zero - eps, negstop],
                [1 - zero, zcolor],
                [1, zcolor],
            ]
        else:
            # sufficient negative and positive values
            colorscale = [
                [0, negstart],
                [zz - zero - eps, negstop],
                [zz - zero, zcolor],
                [zz + zero, zcolor],
                [zz + zero + eps, posstart],
                [1, posstop],
            ]
    return colorscale

def plot_tria_mesh(
    tria,
    vfunc=None,
    tfunc=None,
    vcolor=None,
    tcolor=None,
    showcaxis=False,
    caxis=None,
    xrange=None,
    yrange=None,
    zrange=None,
    plot_edges=False,
    plot_levels=False,
    edge_color="rgb(50,50,50)",
    tic_color="rgb(50,200,10)",
    background_color=None,
    flatshading=False,
    width=800,
    height=800,
    camera=None,
    html_output=False,
    export_png=None,
    scale_png=1.0,
    no_display=False,
    colorscale='rdbu_r',
):
    """Adjust Lapy's plot tria mesh to adjust colour scale. 

    Parameters
    ----------
    tria : lapy.TriaMesh
        Triangle mesh to plot.
    vfunc : array_like, Default=None
        Scalar function at vertices.
    tfunc : array_like, Default=None
        3d vector function of gradient.
    vcolor : list of str, Default=None
        Sets the color of each vertex.
    tcolor : list of str, Default=None
        Sets the color of each face.
    showcaxis : bool, Default=False
        Whether a colorbar is displayed or not.
    caxis : list or tuple of shape (2, 1):
        Sets the bound of the color domain.
        caxis[0] is lower bound caxis[1] upper bound.
        Elements are int or float.
    xrange : list or tuple of shape (2, 1)
        Sets the range of the x-axis.
    yrange : list or tuple of shape (2, 1)
        Sets the range of the y-axis.
    zrange : list or tuple of shape (2, 1)
        Sets the range of the z-axis.
    plot_edges : bool, Default=False
        Whether to plot edges or not.
    plot_levels : bool, Default=False
        Whether to plot levels or not.
    edge_color : str, Default="rgb(50,50,50)"
        Color of the edges.
    tic_color : str, Default="rgb(50,200,10)"
        Color of the ticks.
    background_color : str, Default=None
        Color of background.
    flatshading : bool, Default=False
        Whether normal smoothing is applied to the meshes or not.
    width : int, Default=800
        Width of the plot (in px).
    height : int, Default=800
        Height  of the plot (in px).
    camera : dict of str, Default=None
        Camera describing center, eye and up direction.
    html_output : bool, Default=False
        Whether or not to give out as html output.
    export_png : str, Default=None
        Local file path or file object to write the image to.
    scale_png : int or float
        Scale factor of image. >1.0 increase resolution; <1.0 decrease resolution.
    no_display : bool, Default=False
        Whether to plot on display or not.
    colorscale : str or None. Default=None
        Color scale of the plot, accepting any color scale from plotly's mesh3D. https://plotly.com/python/builtin-colorscales/.  
        Otherwise if None, ensures that the midpoint is at 0 unless full positive or negative.
    """
    # interesting example codes:
    # https://plot.ly/~empet/14749/mesh3d-with-intensities-and-flatshading/#/

    if type(tria).__name__ != "TriaMesh":
        raise ValueError("plot_tria_mesh works only on TriaMesh class")

    if (vfunc is not None or tfunc is not None) and (
        vcolor is not None or tcolor is not None
    ):
        raise ValueError(
            "plot_tria_mesh can only use either vfunc/tfunc or vcolor/tcolor,"
            " but not both at the same time"
        )

    if vcolor is not None and tcolor is not None:
        raise ValueError(
            "plot_tria_mesh can only use either vcolor or tcolor,"
            " but not both at the same time"
        )

    x, y, z = zip(*tria.v)
    i, j, k = zip(*tria.t)

    vlines = []
    if vfunc is None:
        if tfunc is None:
            triangles = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                flatshading=flatshading,
                vertexcolor=vcolor,
                facecolor=tcolor,
            )
        elif tfunc.ndim == 1 or (tfunc.ndim == 2 and np.min(tfunc.shape) == 1):
            # scalar tfunc
            min_fcol = np.min(tfunc)
            max_fcol = np.max(tfunc)
            # special treatment for constant functions
            if np.abs(min_fcol - max_fcol) < 0.0001:
                if np.abs(max_fcol) > 0.0001:
                    min_fcol = -np.abs(min_fcol)
                    max_fcol = np.abs(max_fcol)
                else:  # both are zero
                    min_fcol = -1
                    max_fcol = 1
            # if min_fcol >= 0 and max_fcol <= 1:
            #    min_fcol = 0
            #    max_fcol = 1
            if colorscale is None:
                colorscale = _get_colorscale(min_fcol, max_fcol)
            facecolor = [_map_z2color(zz, colorscale, min_fcol, max_fcol) for zz in tfunc]
            # for tria colors overwrite flatshading to be true:
            triangles = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                facecolor=facecolor,
                flatshading=True,
            )
        elif tfunc.ndim == 2 and np.min(tfunc.shape) == 3:
            # vector tfunc
            s = 0.7 * tria.avg_edge_length()
            centroids = (1.0 / 3.0) * (
                tria.v[tria.t[:, 0], :]
                + tria.v[tria.t[:, 1], :]
                + tria.v[tria.t[:, 2], :]
            )
            xv = np.column_stack(
                (
                    centroids[:, 0],
                    centroids[:, 0] + s * tfunc[:, 0],
                    np.full(tria.t.shape[0], np.nan),
                )
            ).reshape(-1)
            yv = np.column_stack(
                (
                    centroids[:, 1],
                    centroids[:, 1] + s * tfunc[:, 1],
                    np.full(tria.t.shape[0], np.nan),
                )
            ).reshape(-1)
            zv = np.column_stack(
                (
                    centroids[:, 2],
                    centroids[:, 2] + s * tfunc[:, 2],
                    np.full(tria.t.shape[0], np.nan),
                )
            ).reshape(-1)
            vlines = go.Scatter3d(
                x=xv,
                y=yv,
                z=zv,
                mode="lines",
                line=dict(color=tic_color, width=2),
            )
            triangles = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, flatshading=flatshading)
        else:
            raise ValueError(
                "tfunc should be scalar (face color) or 3d for each triangle"
            )

    elif vfunc.ndim == 1 or (vfunc.ndim == 2 and np.min(vfunc.shape) == 1):
        # scalar vfunc
        if plot_levels:
            colorscale = _get_color_levels()
        elif colorscale is None:
            colorscale = _get_colorscale(min(vfunc), max(vfunc))

        triangles = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            intensity=vfunc,
            colorscale=colorscale,
            flatshading=flatshading,
        )
    elif vfunc.ndim == 2 and np.min(vfunc.shape) == 3:
        # vector vfunc
        s = 0.7 * tria.avg_edge_length()
        xv = np.column_stack(
            (
                tria.v[:, 0],
                tria.v[:, 0] + s * vfunc[:, 0],
                np.full(tria.v.shape[0], np.nan),
            )
        ).reshape(-1)
        yv = np.column_stack(
            (
                tria.v[:, 1],
                tria.v[:, 1] + s * vfunc[:, 1],
                np.full(tria.v.shape[0], np.nan),
            )
        ).reshape(-1)
        zv = np.column_stack(
            (
                tria.v[:, 2],
                tria.v[:, 2] + s * vfunc[:, 2],
                np.full(tria.v.shape[0], np.nan),
            )
        ).reshape(-1)
        vlines = go.Scatter3d(
            x=xv, y=yv, z=zv, mode="lines", line=dict(color=tic_color, width=2)
        )
        triangles = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, flatshading=flatshading)
    else:
        raise ValueError("vfunc should be scalar or 3d for each vertex")

    if plot_edges:
        # 4 points = three edges for each tria, nan to separate triangles
        # this plots every edge twice (except boundary edges)
        xe = np.column_stack(
            (
                tria.v[tria.t[:, 0], 0],
                tria.v[tria.t[:, 1], 0],
                tria.v[tria.t[:, 2], 0],
                tria.v[tria.t[:, 0], 0],
                np.full(tria.t.shape[0], np.nan),
            )
        ).reshape(-1)
        ye = np.column_stack(
            (
                tria.v[tria.t[:, 0], 1],
                tria.v[tria.t[:, 1], 1],
                tria.v[tria.t[:, 2], 1],
                tria.v[tria.t[:, 0], 1],
                np.full(tria.t.shape[0], np.nan),
            )
        ).reshape(-1)
        ze = np.column_stack(
            (
                tria.v[tria.t[:, 0], 2],
                tria.v[tria.t[:, 1], 2],
                tria.v[tria.t[:, 2], 2],
                tria.v[tria.t[:, 0], 2],
                np.full(tria.t.shape[0], np.nan),
            )
        ).reshape(-1)

        # define the lines to be plotted
        lines = go.Scatter3d(
            x=xe,
            y=ye,
            z=ze,
            mode="lines",
            line=dict(color=edge_color, width=1.5),
        )

        data = [triangles, lines]

    else:
        data = [triangles]

    if vlines:
        data.append(vlines)

    # line_marker = dict(color='#0066FF', width=2)

    noaxis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title="",
    )

    layout = go.Layout(
        width=width,
        height=height,
        scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis),
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
    )

    if camera is not None:
        layout.scene.camera.center.update(camera["center"])
        layout.scene.camera.eye.update(camera["eye"])
        layout.scene.camera.up.update(camera["up"])

    if xrange is not None:
        layout.scene.xaxis.update(range=xrange)
    if yrange is not None:
        layout.scene.yaxis.update(range=yrange)
    if zrange is not None:
        layout.scene.zaxis.update(range=zrange)

    data[0].update(showscale=showcaxis)

    if caxis is not None:
        data[0].update(cmin=caxis[0])
        data[0].update(cmax=caxis[1])

    fig = go.Figure(data=data, layout=layout)

    if no_display is False:
        if not html_output:
            plotly.offline.iplot(fig)
        else:
            plotly.offline.plot(fig)

    if export_png is not None:
        fig.write_image(export_png, scale=scale_png)

def plot_mean_and_range(data, x=None, stat_type='median', range_type='iqr', show_minmax=True, linecolor='blue', fillcolor='lightblue', ax=None):
    """
    Plots the mean and range (min to max or standard deviation) across the columns of a 2D array.

    Parameters:
    data : np.ndarray
        A 2D array where each column represents a variable.
    x : np.ndarray, optional
        The x-axis values. If None, will use the column indices of the data.
    stat_type : str, optional
        The type of statistic to plot. Can be 'mean', 'median', 'max', 'min'.
    range_type : str, optional
        The type of range to plot. Can be 'std', 'iqr', or None.
    show_minmax : bool, optional
        If True, will show the min and max range as a shaded area.
    linecolor : str, optional
        The color of the mean line.
    fillcolor : str, optional
        The color of the shaded area representing the range.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new plot is created.

    Returns:
    None (the plot is displayed)
    """

    # Check if data contains any NaN values
    if np.isnan(data).any():
        print("Warning: Data contains NaN values. They will be ignored in the visualisation.")
    
    if stat_type == 'mean':
        mean = np.nanmean(data, axis=0)
    elif stat_type == 'median':
        mean = np.nanmedian(data, axis=0)
    elif stat_type == 'max':
        mean = np.nanmax(data, axis=0)
    elif stat_type == 'min':
        mean = np.nanmin(data, axis=0)
    else:
        raise ValueError("stat_type must be either 'mean', 'median', 'max', or 'min'")
    
    if range_type == 'std':
        min_vals = mean - np.nanstd(data, axis=0)
        max_vals = mean + np.nanstd(data, axis=0)
    elif range_type == 'iqr':
        min_vals = np.nanpercentile(data, 25, axis=0)
        max_vals = np.nanpercentile(data, 75, axis=0)    

    if show_minmax == True:
        min_range = np.nanmin(data, axis=0)
        max_range = np.nanmax(data, axis=0)

    if x is None:
        # Create x-axis values
        x = np.arange(data.shape[1])

    # Use the provided axis or create a new one
    if ax is None:
        ax = plt.gca()

    # Plot mean as a line
    ax.plot(x, mean, label=stat_type.capitalize(), color=linecolor)

    # Add shaded region for range
    if range_type:
        ax.fill_between(x, min_vals, max_vals, color=fillcolor, alpha=0.7, )
    if show_minmax:
        ax.fill_between(x, min_range, max_range, color=fillcolor, alpha=0.3, )        

    # Add labels and legend
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Value')
    ax.set_title('Mean and Range Across Columns')
    ax.legend()
    ax.grid()