"""XY plotting helpers with reusable dataclass configurations.

XY is installed with ``pykingenie`` on Python 3.11+. Single-panel helpers return an
``xy.Chart``; subplot helpers return an ``xy.facets.FacetGrid``. Both support
``show()``, ``to_html()``, ``to_svg()``, ``to_png()`` and ``write_image()``.
No Plotly figures are constructed by this module.

Use PlotConfig, AxisConfig, LayoutConfig, LegendConfig and TraceConfig to
share settings across plots. Unspecified fields retain each helper's defaults.
All dimensions are in pixels.
"""

from copy import copy
from dataclasses import dataclass, fields, replace

import numpy as np

from .math import median_filter
from .processing import (
    get_colors_from_numeric_values,
    get_palette,
    sample_type_to_letter,
    subset_data,
)

__all__ = [
    "PlotConfig",
    "AxisConfig",
    "LayoutConfig",
    "LegendConfig",
    "TraceConfig",
    "config_fig",
    "plot_plate_info",
    "plot_traces",
    "plot_traces_all",
    "plot_steady_state",
    "plot_association_dissociation",
]


@dataclass
class PlotConfig:
    """General XY settings. Dimensions are pixels; ``None`` keeps helper defaults.

    Ranges are expressed in data units, including on logarithmic axes. An empty title
    hides the main title; ``None`` keeps the helper's usual title.
    """

    plot_width: int | None = None
    plot_height: int | None = None
    plot_type: str | None = None
    font_size: float | None = None
    title: str | None = None
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None

    def __post_init__(self):
        for name in ("plot_width", "plot_height"):
            _validate_count(getattr(self, name), name)
        _validate_size(self.font_size, "font_size", positive=True)
        self.x_range = _validate_range(self.x_range, "x_range")
        self.y_range = _validate_range(self.y_range, "y_range")


@dataclass
class AxisConfig:
    """Axis grids, ticks, labels, scales, and navigation limits.

    ``x_bounds="data"`` / ``y_bounds="data"`` constrain navigation to the
    observations. ``PlotConfig.x_range`` / ``y_range`` set the initial view.
    """

    showgrid_x: bool | None = None
    showgrid_y: bool | None = None
    n_x_axis_ticks: int | None = None
    n_y_axis_ticks: int | None = None
    axis_linewidth: float = 1
    axis_tickwidth: float = 2
    axis_gridwidth: float = 1
    tick_length: float = 8
    x_label: str | None = None
    y_label: str | None = None
    x_scale: str | None = None
    y_scale: str | None = None
    reverse_x: bool | None = None
    reverse_y: bool | None = None
    x_bounds: tuple[float, float] | str | None = None
    y_bounds: tuple[float, float] | str | None = None

    def __post_init__(self):
        for name in ("n_x_axis_ticks", "n_y_axis_ticks"):
            _validate_count(getattr(self, name), name)
        for name in (
            "axis_linewidth",
            "axis_tickwidth",
            "axis_gridwidth",
            "tick_length",
        ):
            _validate_size(getattr(self, name), name)
        for name in ("x_scale", "y_scale"):
            if getattr(self, name) not in (None, "linear", "log", "symlog", "time"):
                raise ValueError(f"{name} must be linear, log, symlog, or time")
        for name in ("x_bounds", "y_bounds"):
            value = getattr(self, name)
            if isinstance(value, str) and value == "data":
                continue
            setattr(self, name, _validate_range(value, name))


@dataclass
class LayoutConfig:
    """Subplot arrangement, spacing, titles, and shared axis domains.

    ``ncols`` chooses the number of columns; ``stacked=True`` uses one column.
    ``gap`` is in pixels and overrides fractional ``vertical_spacing``; with
    neither set, panels are 0.6 em (at least 8 px) apart.
    Shared axes start with a common domain; their interactions are independent.
    ``split_by_smax_id`` groups kinetics panels by response amplitude ID.
    """

    stacked: bool = False
    ncols: int | None = None
    vertical_spacing: float | None = None
    gap: int | None = None
    shared_xaxes: bool = False
    shared_yaxes: bool = False
    show_subplot_titles: bool = True
    outer_labels: bool | None = None
    extra_padding_y_label: float | None = None
    split_by_smax_id: bool = True

    def __post_init__(self):
        _validate_count(self.ncols, "ncols")
        _validate_count(self.gap, "gap", minimum=0)
        if self.stacked and self.ncols not in (None, 1):
            raise ValueError("stacked=True requires ncols=1 or None")
        if self.vertical_spacing is not None and not 0 <= self.vertical_spacing < 1:
            raise ValueError("vertical_spacing must be between 0 and 1")
        _validate_size(self.extra_padding_y_label, "extra_padding_y_label")


@dataclass
class LegendConfig:
    """Legend visibility, placement, typography, and interaction.

    Unset show/loc/toggle/highlight settings keep the helper's defaults
    (``"upper right"``, or ``"upper left"`` for steady-state curves). Enabling
    toggles for a composed line/marker trace toggles its named mark only.
    """

    show: bool | None = None
    loc: str | None = None
    ncols: int = 1
    title: str | None = None
    font_size: float | None = None
    toggle: bool | None = None
    highlight: bool | None = None

    def __post_init__(self):
        _validate_count(self.ncols, "legend ncols")
        _validate_size(self.font_size, "legend font_size", positive=True)


@dataclass
class TraceConfig:
    """Raw marks, fitted lines, and rendering precision.

    ``max_points`` sets the raw-trace sampling budget (100,000 total sensor
    points or 2,000 points per kinetics group by default). ``exact_lines``
    applies to raw sensor traces; it remains False unless explicitly enabled.
    ``show_fit`` defaults to False for steady state and True for kinetics.
    Kinetics observations can be median-smoothed with ``smooth_curves=True``
    and a ``rolling_window`` in seconds, without changing fitting input.
    """

    marker_size: float | None = None
    line_width: float | None = None
    opacity: float = 1
    fit_color: str = "black"
    fit_line_width: float | None = None
    exact_lines: bool | None = None
    max_points: int | None = None
    show_fit: bool | None = None
    smooth_curves: bool = False
    rolling_window: float = 0.1

    def __post_init__(self):
        for name in ("marker_size", "line_width", "fit_line_width"):
            _validate_size(getattr(self, name), name)
        _validate_count(self.max_points, "max_points")
        _validate_size(self.rolling_window, "rolling_window", positive=True)
        if not np.isfinite(self.opacity) or not 0 <= self.opacity <= 1:
            raise ValueError("opacity must be between 0 and 1")


def _validate_count(value, name, minimum=1):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_size(value, name, positive=False):
    if value is not None and (
        not np.isfinite(value) or value < 0 or (positive and value == 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )


def _validate_range(value, name):
    if value is None:
        return None
    values = np.asarray(value, dtype=float)
    if (
        values.shape != (2,)
        or not np.all(np.isfinite(values))
        or values[0] >= values[1]
    ):
        raise ValueError(f"{name} must contain two finite, increasing values")
    return tuple(float(item) for item in values)


def _resolve_config(config, cls, **defaults):
    """Merge partial configuration without mutating reusable user objects."""
    if config is not None:
        if not isinstance(config, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(config).__name__}")
        defaults.update(
            {
                field.name: getattr(config, field.name)
                for field in fields(config)
                if getattr(config, field.name) is not None
            }
        )
    return cls(**defaults)


def _xy():
    try:
        import xy
    except ImportError as exc:
        raise ImportError(
            "XY plotting requires Python 3.11+ and xy. "
            "Install with: pip install pykingenie"
        ) from exc
    return xy


def _plot_settings(config, *, width=900, height=420, font_size=14, title=None):
    return _resolve_config(
        config,
        PlotConfig,
        plot_width=width,
        plot_height=height,
        font_size=font_size,
        plot_type="png",
        title=title,
    )


def _styles(font_size, legend_font_size=None):
    # XY places a colorbar title a fixed 38 px from the bar, so larger
    # colorbar text would run into the tick labels.
    colorbar_font_size = min(font_size, 12)
    return {
        "title": {"font_size": font_size + 2},
        "axis_title": {"font_size": font_size},
        "tick_label": {"font_size": font_size},
        "legend_label": {"font_size": legend_font_size or font_size},
        "colorbar_title": {"font_size": colorbar_font_size},
        "colorbar_tick": {"font_size": colorbar_font_size},
    }


def _axis_font_style(font_size):
    # XY sizes the axis gutters from the axis style, not from the chart styles
    # used to draw the text, so both must carry the font size.
    return {"tick_label_size": font_size, "label_size": font_size}


def _x_label_offset(font_size, tick_length):
    # XY places the x title a fixed 24 px below the plot, while the tick labels
    # above it end tick_length + 4 px padding + 0.8 em (baseline) + 0.24 em
    # (descent) below the plot. Move the title past them with a 0.4 em gap.
    return max(0.0, tick_length + 4 + 1.44 * font_size - 24)


def _legend_style(font_size):
    # XY's static legend layout only reads the CSS-style font size, so without
    # it the box is sized for 11 px text and clips larger labels.
    return {"fontSize": f"{font_size}px"}


def _concentration_colorbar(low, high):
    """Colorbar settings for ligand concentrations in μM, colored on log10.

    XY has no log color scale for scatter marks, so colors use log10 values and
    the ticks are relabelled with concentrations, in the unit (mM, μM, nM or
    pM) whose labels stay short enough to clear XY's fixed title position.
    """
    log_low, log_high = np.log10(low), np.log10(high)
    domain = (
        (log_low - 0.5, log_high + 0.5) if log_high == log_low else (log_low, log_high)
    )
    factor, unit = next(
        ((f, u) for f, u in ((1e-3, "mM"), (1, "μM"), (1e3, "nM")) if high * f >= 1),
        (1e6, "pM"),
    )
    ticks = sorted({log_low, (log_low + log_high) / 2, log_high})
    labels = [
        np.format_float_positional(
            10**tick * factor, precision=2, unique=False, fractional=False, trim="-"
        )
        for tick in ticks
    ]
    return {
        "title": f"[Ligand] ({unit})",
        "domain": domain,
        "ticks": ticks,
        "labels": labels,
    }


def _first_point(marks):
    for mark in marks:
        x = np.asarray(getattr(getattr(mark, "x", None), "values", getattr(mark, "x", [])), dtype=float)
        y = np.asarray(getattr(getattr(mark, "y", None), "values", getattr(mark, "y", [])), dtype=float)
        if x.size and y.size:
            return float(x[0]), float(y[0])
    return 0.0, 0.0


def _axes(
    plot,
    axes,
    label_offset=None,
    *,
    x_domain=None,
    x_tick_count=None,
    x_tick_values=None,
    margin=None,
):
    """XY axes; ``x_domain``, ``x_tick_count`` and ``x_tick_values`` are helper
    defaults that ``PlotConfig.x_range`` and ``AxisConfig.n_x_axis_ticks``
    override."""
    xy = _xy()
    style = {
        "tick_length": axes.tick_length,
        "tick_width": axes.axis_tickwidth,
        "axis_width": axes.axis_linewidth,
        "grid_width": axes.axis_gridwidth,
        **_axis_font_style(plot.font_size),
    }
    if plot.x_range is not None:
        x_domain = None
    x_bounds = axes.x_bounds
    if x_bounds == "data" and x_domain is not None:
        # Let navigation reach the legend room as well as the observations.
        x_bounds = x_domain
    return [
        xy.x_axis(
            label=axes.x_label,
            type_=axes.x_scale,
            domain=plot.x_range if plot.x_range is not None else x_domain,
            margin=margin,
            bounds=x_bounds,
            reverse=bool(axes.reverse_x),
            tick_count=axes.n_x_axis_ticks or x_tick_count,
            tick_values=None if axes.n_x_axis_ticks else x_tick_values,
            grid=axes.showgrid_x,
            label_offset=_x_label_offset(plot.font_size, axes.tick_length),
            line=True,
            ticks=True,
            style=style,
        ),
        xy.y_axis(
            label=axes.y_label,
            type_=axes.y_scale,
            domain=plot.y_range,
            margin=margin,
            bounds=axes.y_bounds,
            reverse=bool(axes.reverse_y),
            tick_count=axes.n_y_axis_ticks,
            grid=axes.showgrid_y,
            label_offset=label_offset,
            line=True,
            ticks=True,
            style=style,
        ),
    ]


def _plot_area_width(chart_width, font_size):
    # Chart width minus a conservative estimate of the y tick labels and title.
    return max(60.0, chart_width - 5.2 * font_size - 22)


def _x_tick_count(chart_width, font_size):
    # One count for every panel, so equal domains get equal ticks even though
    # the first column is narrower (it holds the y title).
    return max(3, min(8, int(_plot_area_width(chart_width, font_size) / (4.5 * font_size))))


def _linear_x(axis):
    return axis.x_scale in (None, "linear") and not axis.reverse_x


def _legend_room_domain(marks, legend, font_size, chart_width):
    """x domain whose right side leaves room for an upper/center/lower right legend.

    XY can only draw legends inside the plot, so sensorgrams extend the x axis
    instead. Returns None when no right-side legend is drawn.
    """
    labels = [str(mark.name) for mark in marks if getattr(mark, "name", None)]
    if not (legend.show and labels and "right" in legend.loc):
        return None
    xs = [
        np.asarray(getattr(mark.x, "values", mark.x), dtype=float)
        for mark in marks
        if isinstance(getattr(mark, "x", None), (list, tuple, np.ndarray))
        or hasattr(getattr(mark, "x", None), "values")
    ]
    xs = np.concatenate([x[np.isfinite(x)] for x in xs]) if xs else np.array([])
    if not xs.size or xs.max() <= xs.min():
        return None
    # Legend width in pixels: handle (2 em), handle gap (0.8 em) and text
    # (~0.62 em per DejaVu character) per column, 2 em between columns,
    # 0.8 em of padding, and 1 em of inset/gap to the data.
    size = legend.font_size or font_size
    ncols = min(legend.ncols, len(labels))
    column = 2.8 * size + 0.62 * size * max(len(label) for label in labels)
    width = ncols * column + (ncols - 1) * 2 * size + 0.8 * size
    if legend.title:
        width = max(width, 0.62 * size * len(legend.title) + 0.8 * size)
    fraction = min(0.6, (width + size) / _plot_area_width(chart_width, font_size))
    low, high = float(xs.min()), float(xs.max())
    low -= 0.02 * (high - low)
    return low, low + (high - low) / (1 - fraction)


def _legend(config, font_size, *, show=True, toggle=True):
    xy = _xy()
    legend = _resolve_config(
        config,
        LegendConfig,
        show=show,
        loc="upper right",
        toggle=toggle,
        highlight=toggle,
    )
    return xy.legend(
        show=legend.show,
        loc=legend.loc,
        ncols=legend.ncols,
        title=legend.title,
        toggle=legend.toggle,
        highlight=legend.highlight,
        style=_legend_style(legend.font_size or font_size),
    )


def _configure_engine(engine, config):
    if config.plot_type is not None:
        engine.export_options = dict(
            engine.export_options or {}, formats=[config.plot_type]
        )
    if config.title is not None:
        engine.title = config.title
    if config.font_size is not None:
        engine.chrome_styles.update(_styles(config.font_size))
        # New dicts keep grids copied by config_fig from sharing styles.
        for options in engine.axis_options.values():
            options["style"] = dict(
                options.get("style") or {}, **_axis_font_style(config.font_size)
            )
        x_options = engine.axis_options.get("x")
        if x_options is not None:
            x_options["label_offset"] = _x_label_offset(
                config.font_size, x_options["style"].get("tick_length", 0)
            )
        engine.legend_options = dict(
            engine.legend_options or {},
            style=dict(
                (engine.legend_options or {}).get("style") or {},
                **_legend_style(config.font_size),
            ),
        )
    for axis, domain in (("x", config.x_range), ("y", config.y_range)):
        if domain is not None:
            if (
                engine.axis_options.get(axis, {}).get("type") == "log"
                and domain[0] <= 0
            ):
                raise ValueError(
                    f"{axis}_range must be positive for a logarithmic axis"
                )
            engine.axis_options.setdefault(axis, {})["domain"] = domain


def config_fig(
    fig, name_for_download, plot_config: PlotConfig | None = None, *, save=False
):
    """Configure an XY chart/grid and optionally export it using PlotConfig.

    Width and height are pixels. Omitted dimensions keep the current size.
    Configure before displaying live widgets and retain the returned value:
    resizing a grid returns a new grid. ``plot_type`` selects the saved format.
    """
    xy = _xy()
    from xy.facets import FacetGrid

    if not isinstance(fig, (xy.Chart, FacetGrid)):
        raise TypeError("Expected an XY Chart or FacetGrid")
    height = (
        fig.grid_height + (24 if fig.title else 0)
        if isinstance(fig, FacetGrid)
        else fig.height
    )
    config = _resolve_config(
        plot_config,
        PlotConfig,
        plot_width=fig.width,
        plot_height=height,
        plot_type="png",
    )
    width, height = config.plot_width, config.plot_height
    if isinstance(fig, xy.Chart):
        fig.width, fig.height = width, height
        # Chart.figure() is XY's public escape hatch for engine layout.
        engine = fig.figure()
        engine.width, engine.height = width, height
        _configure_engine(engine, config)
        if config.title is not None:
            fig.title = config.title
        if config.font_size is not None:
            fig.styles.update(_styles(config.font_size))
    else:
        title = config.title if config.title is not None else fig.title
        panel_height = max(
            1, (height - (fig.rows - 1) * fig.gap - (24 if title else 0)) // fig.rows
        )
        panel_width = max(120, (width - (fig.cols - 1) * fig.gap) // fig.cols)
        panels = [copy(panel) for panel in fig.figures]
        for panel in panels:
            panel.axis_options = {
                name: dict(options) for name, options in panel.axis_options.items()
            }
            panel.chrome_styles = {
                name: dict(style) for name, style in panel.chrome_styles.items()
            }
            panel.width, panel.height = panel_width, panel_height
            _configure_engine(panel, replace(config, title=None))
        fig = FacetGrid(
            panels,
            fig.labels,
            cols=fig.cols,
            width=width,
            height=panel_height,
            gap=fig.gap,
            title=title,
        )
    if save:
        try:
            fig.write_image(f"{name_for_download}.{config.plot_type}")
        except Exception as exc:
            raise RuntimeError(f"Error saving image: {exc}") from exc
    return fig


def _grid(
    charts,
    plot,
    axis,
    layout,
    legend_config,
    *,
    legend_toggle=True,
    legend_loc="upper right",
    legend_room=False,
    margin=None,
    x_tick_values=None,
    colorbar=None,
):
    """Arrange charts in a FacetGrid.

    ``legend_room`` extends each panel's linear x axis so a right-side legend
    clears its data. ``margin`` pads both automatic axis domains.
    ``colorbar`` (from ``_concentration_colorbar``) is drawn once, beside the
    top-right panel; XY would otherwise infer one in every panel.
    """
    xy = _xy()
    from xy.facets import FacetGrid

    if not charts:
        raise ValueError("No experiments or fittings available to plot")
    legend = _resolve_config(
        legend_config, LegendConfig, show=True, loc=legend_loc, toggle=legend_toggle
    )
    if layout.stacked or layout.ncols is not None:
        ncols = min(1 if layout.stacked else layout.ncols, len(charts))
    else:
        rows = min(2 if len(charts) < 9 else 3, len(charts))
        ncols = int(np.ceil(len(charts) / rows))
    nrows = int(np.ceil(len(charts) / ncols))
    width, height = plot.plot_width, plot.plot_height
    if layout.gap is not None:
        gap = layout.gap
    elif layout.vertical_spacing is not None:
        if not layout.vertical_spacing < 1 / max(1, nrows - 1):
            raise ValueError("vertical_spacing is too large for the subplot rows")
        gap = int(height * layout.vertical_spacing) if nrows > 1 else 0
    else:
        # Each panel already reserves room for its title and tick labels, so
        # a small font-relative gap is enough between rows and columns.
        gap = max(8, round(0.6 * plot.font_size))
    panel_height = max(
        1, (height - (nrows - 1) * gap - (24 if plot.title else 0)) // nrows
    )
    panel_width = max(120, (width - (ncols - 1) * gap) // ncols)
    linear_x = _linear_x(axis)
    x_tick_count = _x_tick_count(panel_width, plot.font_size) if linear_x else None
    domains = [None] * len(charts)
    if legend_room and linear_x:
        domains = [
            _legend_room_domain(chart.children, legend, plot.font_size, panel_width)
            for chart in charts
        ]
    colorbar_panel = min(ncols, len(charts)) - 1
    panels = []
    for i, chart in enumerate(charts):
        row, col = divmod(i, ncols)
        extra = []
        if colorbar is not None and i == colorbar_panel:
            extra = [xy.colorbar(title=colorbar["title"], ticks=colorbar["ticks"])]
            if colorbar.get("needs_color_mark"):
                # Constant-color lines carry no color scale, so an invisible
                # mark with the full range lets XY draw the colorbar.
                x0, y0 = _first_point(chart.children)
                extra.insert(
                    0,
                    xy.scatter(
                        [x0, x0],
                        [y0, y0],
                        color=list(colorbar["domain"]),
                        colormap="viridis",
                        color_domain=colorbar["domain"],
                        opacity=0,
                        density=False,
                    ),
                )
        elif colorbar is not None:
            extra = [xy.colorbar(show=False)]
        panel_axis = replace(
            axis,
            x_label=axis.x_label if not layout.outer_labels or row == nrows - 1 else "",
            y_label=axis.y_label if not layout.outer_labels or col == 0 else "",
        )
        panel = xy.chart(
            *chart.children,
            *extra,
            *_axes(
                plot,
                panel_axis,
                layout.extra_padding_y_label,
                x_domain=domains[i],
                x_tick_count=x_tick_count,
                x_tick_values=x_tick_values,
                margin=margin,
            ),
            _legend(legend, plot.font_size, toggle=legend_toggle),
            xy.export_config(formats=[plot.plot_type]),
            title=chart.title if layout.show_subplot_titles else None,
            width=panel_width,
            height=panel_height,
            styles=_styles(plot.font_size, legend.font_size),
        )
        engine = panel.figure()
        if colorbar is not None and engine.colorbar_options:
            # xy.colorbar() takes tick positions only; the renderers also read
            # paired tick labels from the compiled colorbar options.
            engine.colorbar_options = dict(
                engine.colorbar_options,
                ticks=list(colorbar["ticks"]),
                tick_labels=list(colorbar["labels"]),
            )
        panels.append(engine)
    for axis_id, shared in (("x", layout.shared_xaxes), ("y", layout.shared_yaxes)):
        if shared:
            ranges = [
                panel.build_payload()[0][f"{axis_id}_axis"]["range"] for panel in panels
            ]
            domain = (
                min(min(pair) for pair in ranges),
                max(max(pair) for pair in ranges),
            )
            for panel in panels:
                panel.axis_options[axis_id]["domain"] = domain
    return FacetGrid(
        panels,
        [chart.title or "" for chart in charts],
        cols=ncols,
        width=width,
        height=panel_height,
        gap=gap,
        title=plot.title,
    )


def _array(values):
    # Each sensor may contain several acquisition steps, or a flat list.
    if isinstance(values, (list, tuple)) and values and np.ndim(values[0]) > 0:
        return np.concatenate(values)
    return np.asarray(values)


def _single_chart(
    marks,
    plot,
    axis,
    legend_config=None,
    *,
    show_legend=True,
    legend_toggle=True,
    legend_room=False,
):
    xy = _xy()
    legend = _resolve_config(
        legend_config,
        LegendConfig,
        show=show_legend,
        loc="upper right",
        toggle=legend_toggle,
    )
    x_domain = (
        _legend_room_domain(marks, legend, plot.font_size, plot.plot_width)
        if legend_room and _linear_x(axis)
        else None
    )
    return xy.chart(
        *marks,
        *_axes(plot, axis, x_domain=x_domain),
        _legend(legend, plot.font_size, show=show_legend, toggle=legend_toggle),
        xy.export_config(formats=[plot.plot_type]),
        title=plot.title,
        width=plot.plot_width,
        height=plot.plot_height,
        styles=_styles(plot.font_size, legend.font_size),
    )


def plot_plate_info(
    pyKinetics,
    experiment_name,
    *,
    plot_config: PlotConfig | None = None,
    axis_config: AxisConfig | None = None,
    legend_config: LegendConfig | None = None,
    trace_config: TraceConfig | None = None,
):
    """Return a plate chart with well labels and sample metadata on hover."""
    xy = _xy()
    plot = _plot_settings(
        plot_config, font_size=18, title=f"Sample plate layout ({experiment_name})"
    )
    axis = _resolve_config(
        axis_config,
        AxisConfig,
        x_label="",
        y_label="",
        reverse_y=True,
        showgrid_x=False,
        showgrid_y=False,
    )
    trace = _resolve_config(trace_config, TraceConfig, marker_size=26, line_width=1)
    exp = pyKinetics.experiments[experiment_name]
    required = ("sample_row", "sample_column", "sample_type", "sample_id")
    if any(getattr(exp, key, None) is None for key in required):
        raise AttributeError(
            "Experiment does not have the required attributes: "
            + ", ".join(required)
            + "."
        )
    letters = [sample_type_to_letter(value) for value in exp.sample_type]
    marks = []
    for row, col, letter, sid, stype, conc in zip(
        exp.sample_row,
        exp.sample_column,
        letters,
        exp.sample_id,
        exp.sample_type,
        exp.sample_conc_labeled,
    ):
        marks.extend(
            [
                xy.scatter(
                    [col],
                    [row],
                    size=trace.marker_size,
                    opacity=trace.opacity,
                    density=False,
                    color="pink" if letter == "S" else "transparent",
                    stroke="black",
                    stroke_width=trace.line_width,
                    name=f"ID: {sid}; Type: {stype}; {conc}",
                ),
                xy.text(
                    col,
                    row,
                    letter,
                    dx=0,
                    dy=0,
                    anchor="middle",
                    style={"font_size": max(1, plot.font_size - 2)},
                ),
            ]
        )
    return _single_chart(marks, plot, axis, legend_config, show_legend=False)


def plot_traces(
    xs,
    ys,
    legends,
    colors,
    show,
    *,
    plot_config: PlotConfig | None = None,
    axis_config: AxisConfig | None = None,
    legend_config: LegendConfig | None = None,
    trace_config: TraceConfig | None = None,
):
    """Plot selected sensors, optionally connecting every displayed sample.

    TraceConfig controls sampling, opacity, marker size and line width.
    ``exact_lines=True`` retains every adjacent pair in standalone HTML;
    the default uses XY's faster automatic line reduction. Both modes use
    the same initial sampling budget for lines and markers.
    """
    xy = _xy()
    plot = _plot_settings(plot_config)
    axis = _resolve_config(axis_config, AxisConfig, showgrid_x=True, showgrid_y=True)
    trace = _resolve_config(
        trace_config,
        TraceConfig,
        marker_size=1,
        line_width=2,
        exact_lines=False,
        max_points=100000,
    )
    marks = []
    limit = max(1, int(trace.max_points / sum(show))) if sum(show) else trace.max_points
    for x, y, legend, color, visible in zip(xs, ys, legends, colors, show):
        if not visible:
            continue
        x, y = subset_data(_array(x), limit), subset_data(_array(y), limit)
        if trace.line_width > 0:
            name = legend if trace.marker_size <= 0 else None
            if trace.exact_lines and len(x) > 1:
                marks.append(
                    xy.segments(
                        x[:-1],
                        y[:-1],
                        x[1:],
                        y[1:],
                        name=name,
                        color=color,
                        width=trace.line_width,
                        opacity=trace.opacity,
                    )
                )
            else:
                marks.append(
                    xy.line(
                        x,
                        y,
                        name=name,
                        color=color,
                        width=trace.line_width,
                        opacity=trace.opacity,
                    )
                )
        if trace.marker_size > 0:
            marks.append(
                xy.scatter(
                    x,
                    y,
                    name=legend,
                    color=color,
                    size=trace.marker_size,
                    opacity=trace.opacity,
                    density=False,
                )
            )
    # XY toggles individual marks, so a line + scatter legend is informational.
    toggle = not (trace.line_width > 0 and trace.marker_size > 0)
    return _single_chart(
        marks, plot, axis, legend_config, legend_toggle=toggle, legend_room=True
    )


def plot_traces_all(
    pyKinetics,
    legends_df,
    *,
    plot_config: PlotConfig | None = None,
    axis_config: AxisConfig | None = None,
    layout_config: LayoutConfig | None = None,
    legend_config: LegendConfig | None = None,
    trace_config: TraceConfig | None = None,
):
    """Plot one sensor panel per experiment using reusable configuration objects."""
    plot = _plot_settings(
        plot_config, width=800, height=700, font_size=18, title="Traces"
    )
    axis = _resolve_config(
        axis_config,
        AxisConfig,
        x_label="Time (s)",
        y_label="Signal (a.u.)",
        showgrid_x=False,
        showgrid_y=False,
    )
    layout = _resolve_config(
        layout_config, LayoutConfig, outer_labels=False
    )
    trace = _resolve_config(trace_config, TraceConfig, marker_size=1, line_width=2)
    charts, offset = [], 0
    names = list(getattr(pyKinetics, "experiments", {}))
    for i, (xs, ys) in enumerate(
        zip(
            pyKinetics.get_experiment_properties("xs"),
            pyKinetics.get_experiment_properties("ys"),
        )
    ):
        rows = legends_df.iloc[offset : offset + len(xs)]
        charts.append(
            plot_traces(
                xs, ys, rows["Legend"], rows["Color"], rows["Show"], trace_config=trace
            )
        )
        charts[-1].title = names[i] if i < len(names) else f"Experiment {i + 1}"
        offset += len(xs)
    return _grid(
        charts,
        plot,
        axis,
        layout,
        legend_config,
        legend_toggle=trace.marker_size <= 0 or trace.line_width <= 0,
        legend_room=True,
    )


def plot_steady_state(
    pyKinetics,
    *,
    plot_config: PlotConfig | None = None,
    axis_config: AxisConfig | None = None,
    layout_config: LayoutConfig | None = None,
    legend_config: LegendConfig | None = None,
    trace_config: TraceConfig | None = None,
):
    """Plot steady-state observations on log concentration axes.

    Enable fitted curves with ``TraceConfig(show_fit=True)``.
    """
    xy = _xy()
    plot = _plot_settings(
        plot_config, width=1500, height=1000, font_size=18, title="Steady-state"
    )
    axis = _resolve_config(
        axis_config,
        AxisConfig,
        x_label="Ligand concentration (μM)",
        y_label="Signal (a.u.)",
        x_scale="log",
        showgrid_x=True,
        showgrid_y=True,
    )
    layout = _resolve_config(
        layout_config, LayoutConfig, outer_labels=True
    )
    trace = _resolve_config(
        trace_config, TraceConfig, marker_size=5, line_width=2, show_fit=False
    )
    fit_width = (
        trace.line_width if trace.fit_line_width is None else trace.fit_line_width
    )
    charts = []
    for fit in pyKinetics.fittings.values():
        marks = []
        colors = get_palette(len(fit.names))
        for i, concentrations in enumerate(fit.lig_conc_lst_per_id):
            x = np.asarray(concentrations)
            if trace.marker_size > 0:
                marks.append(
                    xy.scatter(
                        x,
                        fit.signal_ss[i],
                        name=fit.names[i],
                        color=colors[i],
                        size=trace.marker_size,
                        opacity=trace.opacity,
                        density=False,
                    )
                )
            if trace.show_fit and fit.signal_ss_fit is not None and fit_width > 0:
                order = np.argsort(x)
                marks.append(
                    xy.line(
                        x[order],
                        np.asarray(fit.signal_ss_fit[i])[order],
                        color=trace.fit_color,
                        width=fit_width,
                    )
                )
        charts.append(xy.chart(*marks, title=fit.names[0].rsplit("_", 1)[0]))
    # Pad both axes so markers at the extremes are not cut by the frame; XY's
    # automatic log domain leaves almost no room above the highest point.
    margin = 0.05
    decades = None
    concentrations = np.asarray(
        [
            value
            for fit in pyKinetics.fittings.values()
            for group in fit.lig_conc_lst_per_id
            for value in np.ravel(group)
        ],
        dtype=float,
    )
    concentrations = concentrations[np.isfinite(concentrations) & (concentrations > 0)]
    if axis.x_scale == "log" and concentrations.size:
        # XY labels only every other decade in narrow panels, which can leave
        # a single label, so list the decades explicitly.
        low, high = np.log10(concentrations.min()), np.log10(concentrations.max())
        pad = margin * (high - low)
        decades = 10.0 ** np.arange(np.ceil(low - pad), np.floor(high + pad) + 1)
        if len(decades) < 2:
            decades = None
    # Binding curves rise to the right, so the upper left corner is empty.
    return _grid(
        charts,
        plot,
        axis,
        layout,
        legend_config,
        legend_loc="upper left",
        margin=margin,
        x_tick_values=None if decades is None else decades.tolist(),
    )


def plot_association_dissociation(
    pyKinetics,
    *,
    plot_config: PlotConfig | None = None,
    axis_config: AxisConfig | None = None,
    layout_config: LayoutConfig | None = None,
    legend_config: LegendConfig | None = None,
    trace_config: TraceConfig | None = None,
):
    """Plot kinetics observations and fitted curves, colored by concentration.

    One colorbar beside the top-right panel shows the ligand concentration
    (log-scaled Viridis), shared by all panels. LayoutConfig controls
    splitting by Smax ID. TraceConfig controls fitted overlays, median
    smoothing and the sampling budget per Smax group.
    """
    xy = _xy()
    plot = _plot_settings(
        plot_config, width=1300, height=1000, title="Association-dissociation traces"
    )
    axis = _resolve_config(
        axis_config,
        AxisConfig,
        x_label="Time (s)",
        y_label="Signal (a.u.)",
        showgrid_x=False,
        showgrid_y=False,
    )
    layout = _resolve_config(
        layout_config, LayoutConfig, outer_labels=True
    )
    trace = _resolve_config(
        trace_config,
        TraceConfig,
        marker_size=4,
        line_width=2,
        max_points=2000,
        show_fit=True,
    )
    fit_width = (
        trace.line_width if trace.fit_line_width is None else trace.fit_line_width
    )
    fits = list(pyKinetics.fittings.values())
    if not fits:
        raise ValueError("No experiments or fittings available to plot")
    groups = [np.asarray(group) for fit in fits for group in fit.lig_conc_lst_per_id]
    if not groups or any(not group.size for group in groups):
        raise ValueError("Each Smax group must contain at least one trace")
    concentrations = np.concatenate(groups)
    if np.any(~np.isfinite(concentrations)) or np.any(concentrations <= 0):
        raise ValueError("Ligand concentrations must be finite and positive")
    low, high = concentrations.min(), concentrations.max()
    colorbar = _concentration_colorbar(low, high)
    colorbar["needs_color_mark"] = trace.smooth_curves or trace.marker_size <= 0
    charts = []
    for fit in fits:
        marks, counter = [], 0
        for i, group in enumerate(fit.lig_conc_lst_per_id):
            limit = max(1, int(np.ceil(trace.max_points / len(group))))
            for concentration in group:
                color = get_colors_from_numeric_values([concentration], low, high)[0]
                for phase, times, raw, fitted in (
                    ("Assoc", fit.time_assoc_lst, fit.assoc_lst, fit.signal_assoc_fit),
                    ("Disso", fit.time_disso_lst, fit.disso_lst, fit.signal_disso_fit),
                ):
                    x, y = np.asarray(times[counter]), np.asarray(raw[counter])
                    if trace.smooth_curves:
                        y = median_filter(y, x, trace.rolling_window)
                    x, y = subset_data(x, limit), subset_data(y, limit)
                    if trace.smooth_curves and trace.line_width > 0:
                        # Lines take one constant color, the closest Viridis step.
                        marks.append(
                            xy.line(
                                x,
                                y,
                                color=color,
                                width=trace.line_width,
                                opacity=trace.opacity,
                            )
                        )
                    elif not trace.smooth_curves and trace.marker_size > 0:
                        marks.append(
                            xy.scatter(
                                x,
                                y,
                                color=np.full(len(x), np.log10(concentration)),
                                colormap="viridis",
                                color_domain=colorbar["domain"],
                                size=trace.marker_size,
                                opacity=trace.opacity,
                                density=False,
                            )
                        )
                    if (
                        trace.show_fit
                        and fitted is not None
                        and len(fitted)
                        and fitted[counter] is not None
                        and fit_width > 0
                    ):
                        marks.append(
                            xy.line(
                                x,
                                subset_data(np.asarray(fitted[counter]), limit),
                                color=trace.fit_color,
                                width=fit_width,
                            )
                        )
                counter += 1
            if layout.split_by_smax_id:
                charts.append(xy.chart(*marks, title=fit.names[i]))
                marks = []
        if not layout.split_by_smax_id:
            charts.append(xy.chart(*marks, title=fit.names[0].rsplit("_", 1)[0]))
    return _grid(
        charts,
        plot,
        axis,
        layout,
        legend_config,
        legend_toggle=False,
        colorbar=colorbar,
    )
