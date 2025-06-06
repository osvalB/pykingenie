import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from ..utils.processing import *
from ..utils.math       import median_filter

PLOT_TICKS_LENGTH_CST = 8

def config_fig(fig, name_for_download, export_format=None,
               plot_width=10, plot_height=6, scale_factor=50, autosize=True,save=False):
    """
    Configure Plotly figure size and optionally save to file.

    Parameters:
        fig (go.Figure): The Plotly figure.
        name_for_download (str): Base name for saving the figure.
        export_format (str): 'png', 'svg', 'pdf', etc. (if saving).
        plot_width (int): Width in arbitrary units.
        plot_height (int): Height in arbitrary units.
        scale_factor (int): Multiply width/height to get actual pixels.
        save (bool): Whether to save the figure as a file.

    Returns:
        go.Figure: The updated figure.
    """

    fig.update_layout(
        autosize=False,
        width=plot_width * scale_factor,
        height=plot_height * scale_factor
    )

    if save and export_format:
        filename = f"{name_for_download}.{export_format}"
        try:
            fig.write_image(filename)
        except Exception as e:
            print(f"Error saving image: {e}")

    return fig

def plot_plate_info(pyKinetics, experiment_name, font_size=18):

    py_single_exp = pyKinetics.experiments[experiment_name]

    sample_row      = py_single_exp.sample_row
    sample_column   = py_single_exp.sample_column
    sample_type     = py_single_exp.sample_type
    sample_id       = py_single_exp.sample_id

    sample_conc_labeled = py_single_exp.sample_conc_labeled

    sample_type_simple = [sample_type_to_letter(st) for st in sample_type]
    df = pd.DataFrame({'sample_row': sample_row, 'sample_column': sample_column, 'sample_type_simple': sample_type_simple})
    marker_colors = ['pink' if st == 'S' else 'rgba(0,0,0,0)' for st in df['sample_type_simple']]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['sample_column'], y=df['sample_row'], mode='markers+text',
        marker=dict(size=26, color=marker_colors, line=dict(color='black', width=1)),
        text=df['sample_type_simple'], textposition='middle center',
        textfont=dict(size=font_size-2),
        hoverinfo='text',
        hovertext=[
            f"ID: {sid}<br>Type: {stype}<br>{conc}"
            for sid, stype, conc in zip(sample_id, sample_type, sample_conc_labeled)
        ]
    ))
    fig.update_layout(
        xaxis=dict(title='', tickfont=dict(size=font_size)),
        yaxis=dict(title='', tickfont=dict(size=font_size), autorange='reversed'),
        title=dict(text=f"Sample plate layout ({experiment_name})", font=dict(size=font_size+1))
    )
    return fig

def plot_traces(xs, ys, legends, colors, show,
                font_size=18, show_grid_x=False, show_grid_y=False,
                marker_size=1, line_width=2, vertical_lines=None):

    fig = go.Figure()
    total_traces = sum(show)
    max_points_per_trace = int(4000 / total_traces) if total_traces else 4000

    for x, y, legend, color, s in zip(xs, ys, legends, colors, show):
        if not s:
            continue

        # x and y are lists of arrays, we need to concatenate them
        x = np.concatenate(x) if isinstance(x, list) else x
        y = np.concatenate(y) if isinstance(y, list) else y

        x = subset_data(x, max_points_per_trace)
        y = subset_data(y, max_points_per_trace)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines+markers', name=legend,
            marker=dict(size=marker_size, color=color),
            line=dict(width=line_width, color=color)
        ))

    return fig

def plot_traces_all(pyKinetics, legends_df,
                    plot_width=16, plot_height=14, plot_type='png',
                    font_size=18, show_grid_x=False, show_grid_y=False,
                    marker_size=1, line_width=2, vertical_lines=None):

    all_xs = pyKinetics.get_experiment_properties('xs')
    all_ys = pyKinetics.get_experiment_properties('ys')

    # Obtain the legends, colors, and show flags from the legends_df
    legends = legends_df['Legend'].tolist()
    colors = legends_df['Color'].tolist()
    show = legends_df['Show'].tolist()

    cnt = 0
    plot_list = []
    for xs, ys in zip(all_xs, all_ys):
        n = len(xs)
        legends_tmp = legends[cnt:cnt+n]
        colors_tmp = colors[cnt:cnt+n]
        show_tmp = show[cnt:cnt+n]
        cnt += n
        fig = plot_traces(xs, ys, legends_tmp, colors_tmp, show_tmp, font_size, show_grid_x, show_grid_y, marker_size, line_width, vertical_lines)
        plot_list.append(fig)

    n_plots = len(plot_list)

    # Set number of rows: 2 if less than 8 plots, else 3
    nrows = 2 if n_plots < 9 else 3
    nrows = min(nrows, n_plots)  # Do not exceed the number of plots - case n equal 1
    ncols       = int(np.ceil(n_plots / nrows))
    subplot_fig = make_subplots(rows=nrows, cols=ncols)

    col_arr = np.arange(1, ncols + 1)
    row_arr = np.arange(1, nrows + 1)
    # Row and column counters for subplotting
    row_col_info = combine_sequences(row_arr,col_arr)

    for i,fig in enumerate(plot_list):

        row = row_col_info[i][0]
        col = row_col_info[i][1]

        for trace in fig.data:
            subplot_fig.add_trace(trace, row=row, col=col)

    # Apply custom layout globally
    subplot_fig.update_layout(
        title=dict(text="Sensor traces", font=dict(size=font_size + 2)),
        legend=dict(font=dict(size=font_size * 0.8)),
        margin=dict(t=60, b=60, l=70, r=40),
        autosize=True
    )

    # Apply consistent axis layout to all x and y axes
    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):
            subplot_fig.update_xaxes(
                row=i, col=j,
                title=dict(text='Time (s)', font=dict(size=font_size)),
                tickfont=dict(size=font_size * 0.9),
                showgrid=show_grid_x, showline=True, zeroline=False, ticks="outside",
                tickwidth=2, ticklen=PLOT_TICKS_LENGTH_CST,
                tickangle=0, nticks=6
            )
            subplot_fig.update_yaxes(
                row=i, col=j,
                title=dict(text='Signal (a.u.)', font=dict(size=font_size)),
                tickfont=dict(size=font_size * 0.9),
                showgrid=show_grid_y, showline=True, zeroline=False, ticks="outside",
                tickwidth=2, ticklen=PLOT_TICKS_LENGTH_CST,
                nticks=6
            )

    subplot_fig.update_layout(
        title=dict(text="Traces", font=dict(size=font_size + 2)),
        legend=dict(font=dict(size=font_size))    # Set legend font size here

    )

    subplot_fig = config_fig(subplot_fig,
                             name_for_download='Sensor_traces_plot',
                             export_format=plot_type,
                             plot_width=plot_width,
                             plot_height=plot_height,
                             scale_factor=50,
                             save=False)

    return subplot_fig


def plot_steady_state(pyKinetics,
                      plot_width=30,
                      plot_height=20,
                      plot_type='png',
                      font_size=18,
                      show_grid_x=True,
                      show_grid_y=True,
                      marker_size=5,
                      line_width=2,
                      plot_fit=False):

    pyKinetics_fittings = pyKinetics.fittings.values()

    # Create individual figures and collect traces
    fig_lst        = []
    subplot_titles = []
    for fit in pyKinetics_fittings:
        fig = go.Figure()
        yFit  = fit.signal_ss_fit
        names = fit.names
        # Remove the string after the last underscore
        selected_name = names[0]
        selected_name = "_".join(selected_name.split("_")[:-1])
        subplot_titles.append(selected_name)
        for i, x in enumerate(fit.lig_conc_lst_per_id):
            x = np.array(x)
            y = fit.signal_ss[i]
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                     name=names[i], marker=dict(size=marker_size)))
            if yFit is not None and plot_fit:
                x_fit = np.array(x)
                y_fit = np.array(yFit[i])
                idx = np.argsort(x_fit)
                fig.add_trace(go.Scatter(x=x_fit[idx], y=y_fit[idx], mode='lines',
                                         name=names[i],
                                         line=dict(color='black', width=line_width),
                                         showlegend=False))
        fig_lst.append(fig)

    n_plots = len(fig_lst)

    # Set number of rows: 2 if less than 8 plots, else 3
    nrows = 2 if n_plots < 9 else 3
    nrows = min(nrows, n_plots)  # Do not exceed the number of plots - case n equal 1
    ncols = int(np.ceil(n_plots / nrows))

    subplot_fig = make_subplots(rows=nrows, cols=ncols,
                                subplot_titles=subplot_titles)

    col_arr = np.arange(1, ncols + 1)
    row_arr = np.arange(1, nrows + 1)
    # Row and column counters for subplotting
    row_col_info = combine_sequences(row_arr,col_arr)

    for i,fig in enumerate(fig_lst):

        row = row_col_info[i][0]
        col = row_col_info[i][1]

        for trace in fig.data:
            subplot_fig.add_trace(trace, row=row, col=col)

    # Assign axis titles and layout to each subplot
    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):

            # Set the x-axis title only for the last row
            if i == nrows:
                title_text_x = 'Ligand concentration (μM)'
            else:
                title_text_x = ''
            # Set the y-axis title only for the first column
            if j == 1:
                title_text_y = 'Signal (a.u.)'
            else:
                title_text_y = ''    

            subplot_fig.update_xaxes(
                row=i, col=j,
                title=dict(text=title_text_x, font=dict(size=font_size)),
                tickfont=dict(size=font_size),
                exponentformat='power', showgrid=show_grid_x, showline=True, zeroline=False, ticks="outside",
                tickwidth=2, ticklen=PLOT_TICKS_LENGTH_CST, type='log'
            )
            subplot_fig.update_yaxes(
                row=i, col=j,
                title=dict(text=title_text_y, font=dict(size=font_size)),
                tickfont=dict(size=font_size),
                showgrid=show_grid_y, showline=True, zeroline=False, ticks="outside",
                tickwidth=2, ticklen=PLOT_TICKS_LENGTH_CST
            )

    subplot_fig.update_layout(
        title=dict(text="Steady-state", font=dict(size=font_size + 2)),
        legend=dict(font=dict(size=font_size))    # Set legend font size here

    )

    subplot_fig = config_fig(subplot_fig,
                             name_for_download='Steady-state plot',
                             export_format=plot_type,
                             plot_width=plot_width,
                             plot_height=plot_height,
                             scale_factor=50,
                             save=False)

    return subplot_fig

def plot_association_dissociation(
    pyKinetics,
    plot_width=26,
    plot_height=20,
    plot_type='png',
    font_size=14,
    show_grid_x=False,
    show_grid_y=False,
    marker_size=4,
    line_width=2,
    split_by_smax_id=True,
    max_points_per_plot=2000,
    smooth_curves_fit=False,
    rolling_window=0.1
):
    pyKinetics_fittings = pyKinetics.fittings.values()

    lig_conc_lst = pyKinetics.get_experiment_properties('lig_conc_lst', fittings=True)

    all_lig_conc = np.concatenate(lig_conc_lst,axis=0)
    min_lig = np.min(all_lig_conc)
    max_lig = np.max(all_lig_conc)

    n_subplots = sum(len(fit.lig_conc_lst_per_id) for fit in pyKinetics_fittings) if split_by_smax_id else len(pyKinetics_fittings)

    # Set number of rows: 2 if less than 8 plots, else 3
    nrows = 2 if n_subplots < 9 else 3
    nrows = min(nrows, n_subplots)  # Do not exceed the number of plots - case n equal 1

    ncols = int(np.ceil(n_subplots / nrows))

    row_arr = np.arange(1, nrows + 1)
    col_arr = np.arange(1, ncols + 1)
    # Row and column counters for subplotting
    row_col_info = combine_sequences(row_arr, col_arr)

    # Obtain the subplot titles
    subplot_titles = []
    for fit in pyKinetics_fittings:

        names = fit.names
        # Remove the string after the last underscore

        # Add one title per fitting dataset, that shares the same thermodynamic parameters.
        if not split_by_smax_id:

            selected_name = names[0]
            selected_name = "_".join(selected_name.split("_")[:-1])
            subplot_titles.append(selected_name)
        
        # Add one title per fitting dataset, that shares both the thermodynamic parameters and the smax_id. 
        else:
            subplot_titles += names

    fig = make_subplots(
        rows=nrows, 
        cols=ncols, 
        shared_xaxes=False, 
        shared_yaxes=False, 
        vertical_spacing=0.08,
        subplot_titles=subplot_titles)

    subplot_idx = 0

    for fit in pyKinetics_fittings:

        lig_conc = fit.lig_conc_lst_per_id
        fitted_curves_assoc = fit.signal_assoc_fit
        fitted_curves_disso = fit.signal_disso_fit
        time_assoc = fit.time_assoc_lst
        time_disso = fit.time_disso_lst
        raw_curves_assoc = fit.assoc_lst
        raw_curves_disso = fit.disso_lst

        fc_counter = 0
        for i, l in enumerate(lig_conc):
            n_traces = len(l)
            max_points_per_trace = np.ceil(max_points_per_plot / n_traces).astype(int)

            row = row_col_info[subplot_idx][0]
            col = row_col_info[subplot_idx][1]

            if split_by_smax_id:
                subplot_idx += 1

            for j in range(n_traces):
                l_conc = l[j]
                color = get_colors_from_numeric_values([l_conc], min_lig, max_lig)[0]

                # Association
                x_assoc = np.array(time_assoc[fc_counter])
                y_assoc = np.array(raw_curves_assoc[fc_counter])
                if smooth_curves_fit:
                    y_assoc = median_filter(y_assoc, x_assoc, rolling_window)
                x_assoc = subset_data(x_assoc, max_points_per_trace)
                y_assoc = subset_data(y_assoc, max_points_per_trace)
                fig.add_trace(
                    go.Scatter(
                        x=x_assoc, y=y_assoc, mode='markers' if not smooth_curves_fit else 'lines',
                        marker=dict(size=marker_size, color=color),
                        line=dict(width=line_width, color=color),
                        name=f'Assoc {i+1} {l_conc:.2g}',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                if fitted_curves_assoc:
                    y_fit_assoc = np.array(fitted_curves_assoc[fc_counter])
                    y_fit_assoc = subset_data(y_fit_assoc, max_points_per_trace)
                    fig.add_trace(
                        go.Scatter(
                            x=x_assoc, y=y_fit_assoc, mode='lines',
                            line=dict(width=line_width, color='black'),
                            name=f'Assoc fit {i+1} {l_conc:.2g}',
                            showlegend=False
                        ),
                        row=row, col=col
                    )

                # Dissociation
                x_disso = np.array(time_disso[fc_counter])
                y_disso = np.array(raw_curves_disso[fc_counter])
                if smooth_curves_fit:
                    y_disso = median_filter(y_disso, x_disso, rolling_window)
                x_disso = subset_data(x_disso, max_points_per_trace)
                y_disso = subset_data(y_disso, max_points_per_trace)
                fig.add_trace(
                    go.Scatter(
                        x=x_disso, y=y_disso, mode='markers' if not smooth_curves_fit else 'lines',
                        marker=dict(size=marker_size, color=color),
                        line=dict(width=line_width, color=color),
                        name=f'Disso {i+1} {l_conc:.2g}',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                if fitted_curves_disso:
                    y_fit_disso = np.array(fitted_curves_disso[fc_counter])
                    y_fit_disso = subset_data(y_fit_disso, max_points_per_trace)
                    fig.add_trace(
                        go.Scatter(
                            x=x_disso, y=y_fit_disso, mode='lines',
                            line=dict(width=line_width, color='black'),
                            name=f'Disso fit {i+1} {l_conc:.2g}',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                fc_counter += 1

        # If not splitting by smax_id, we only increment the subplot index once
        if not split_by_smax_id:
            subplot_idx += 1

    for i in range(n_subplots):

        row = row_col_info[i][0]
        col = row_col_info[i][1]

        # Set the x-axis title only for the last row
        if row == nrows:
            title_text_x = 'Time (s)'
        else:
            title_text_x = ''
        # Set the y-axis title only for the first column
        if col == 1:
            title_text_y = 'Signal (a.u.)'
        else:
            title_text_y = ''
            
        fig.update_xaxes(
            title_text=title_text_x,
            showgrid=show_grid_x,
            showline=True,
            zeroline=False,
            ticks="outside",
            tickwidth=2,
            ticklen=8,
            title_font_size=font_size,
            tickfont_size=font_size,
            row=row, col=col
        )
        fig.update_yaxes(
            title_text=title_text_y,
            showgrid=show_grid_y,
            showline=True,
            zeroline=False,
            ticks="outside",
            tickwidth=2,
            ticklen=8,
            title_font_size=font_size,
            tickfont_size=font_size,
            row=row, col=col
        )

        # Add subplot title

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                colorscale='Viridis',
                cmin=np.log10(min_lig),
                cmax=np.log10(max_lig),
                colorbar=dict(
                    title='Ligand<br>concentration (μM)',
                    tickvals=[np.log10(min_lig), np.log10((min_lig*max_lig)**0.5), np.log10(max_lig)],
                    ticktext=[f"{min_lig:.2g}", f"{(min_lig*max_lig)**0.5:.2g}", f"{max_lig:.2g}"],
                    len=0.6,
                    outlinewidth=0,
                    ticks='outside',
                    tickfont=dict(size=font_size-2)
                ),
                showscale=True
            ),
            showlegend=False
        ),
        row=1, col=1
    )

    fig.update_layout(
        title=dict(text="Association-dissociation traces", font=dict(size=font_size + 2)),
        legend=dict(font=dict(size=font_size))
    )

    fig = config_fig(
        fig,
        name_for_download='Association-dissociation traces',
        export_format=plot_type,
        plot_width=plot_width,
        plot_height=plot_height,
        scale_factor=50,
        save=False
    )

    return fig