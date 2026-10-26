# Note: this file contains AI-generated code, written with AI coding assistants.

"""Exercise the real XY renderer as well as the kinetics-to-chart mapping."""

import builtins
import inspect
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import struct
import numpy as np
import pandas as pd
import pytest

xy = pytest.importorskip("xy")
from pykingenie.utils import plotting_xy as plotting


def test_public_helpers_accept_configuration_objects_for_styling():
    data_parameters = {
        "fig",
        "name_for_download",
        "save",
        "pyKinetics",
        "experiment_name",
        "xs",
        "ys",
        "legends",
        "colors",
        "show",
        "legends_df",
    }
    configs = {
        "plot_config",
        "axis_config",
        "layout_config",
        "legend_config",
        "trace_config",
    }
    for name in plotting.__all__:
        helper = getattr(plotting, name)
        if inspect.isfunction(helper):
            assert (
                set(inspect.signature(helper).parameters) <= data_parameters | configs
            )


@pytest.fixture
def analyzer():
    times = [np.linspace(0, 4, 41) for _ in range(3)]
    fit = SimpleNamespace(
        names=["experiment_sensor_1", "experiment_sensor_2"],
        lig_conc_lst_per_id=[[10.0, 1.0], [3.0]],
        signal_ss=[[8.0, 2.0], [5.0]],
        signal_ss_fit=[[7.0, 3.0], [4.0]],
        time_assoc_lst=times,
        time_disso_lst=[x + 5 for x in times],
        assoc_lst=[x**2 for x in times],
        disso_lst=[16 - x**2 for x in times],
        signal_assoc_fit=[x**2 + 0.1 for x in times],
        signal_disso_fit=[16 - x**2 + 0.1 for x in times],
    )
    return SimpleNamespace(fittings={"experiment": fit})


def test_traces_selection_steps_sampling_and_rendering():
    x = np.arange(120000)
    chart = plotting.plot_traces(
        [[x[:60000], x[60000:]], x],
        [[x[:60000] * 2, x[60000:] * 2], x],
        ["visible", "hidden"],
        ["red", "blue"],
        [True, False],
    )
    traces = chart.figure().traces
    assert [trace.name for trace in traces] == [None, "visible"]
    assert [trace.kind for trace in traces] == ["line", "scatter"]
    assert len(traces[0].x.values) <= 100000
    np.testing.assert_equal(traces[0].y.values, traces[0].x.values * 2)
    assert chart.figure().legend_options["toggle"] is False
    assert "<svg" in chart.to_svg()
    assert plotting.plot_traces([], [], [], [], []).figure().traces == []


def test_traces_all_preserves_experiment_slices():
    properties = {
        "xs": [[[0, 1], [0, 1]], [[2, 3]]],
        "ys": [[[4, 5], [6, 7]], [[8, 9]]],
    }
    analyzer = SimpleNamespace(get_experiment_properties=properties.__getitem__)
    legends = pd.DataFrame(
        {
            "Legend": ["A", "B", "C"],
            "Color": ["red", "blue", "green"],
            "Show": [False, True, True],
        },
        index=[10, 20, 30],
    )
    grid = plotting.plot_traces_all(
        analyzer, legends, axis_config=plotting.AxisConfig(showgrid_x=True)
    )
    assert len(grid.figures) == 2
    assert grid.figures[0].traces[1].name == "B"
    assert grid.figures[1].traces[1].name == "C"
    for panel in grid.figures:
        spec, _ = panel.build_payload()
        assert spec["x_axis"]["label"] == "Time (s)"
        assert spec["y_axis"]["label"] == "Signal (a.u.)"
    assert "xy-facet-grid" in grid.to_html()


def test_exact_lines_keep_every_displayed_segment_in_html():
    x = np.arange(12001, dtype=float)
    y = np.sin(x * 1.7)
    properties = {"xs": [[x]], "ys": [[y]]}
    analyzer = SimpleNamespace(get_experiment_properties=properties.__getitem__)
    legends = pd.DataFrame({"Legend": ["A"], "Color": ["red"], "Show": [True]})
    grid = plotting.plot_traces_all(
        analyzer, legends, trace_config=plotting.TraceConfig(exact_lines=True)
    )
    spec, blob = grid.figures[0].build_payload(px_width=50)
    segments, points = spec["traces"]
    assert segments["kind"] == "segments"
    assert segments["tier"] == points["tier"] == "direct"
    assert segments["n_marks"] == points["n_marks"] - 1 == len(x) - 1

    def decode(column_id):
        column = spec["columns"][column_id]
        values = np.frombuffer(
            blob, dtype=np.float32, count=column["len"], offset=column["byte_offset"]
        ).astype(float)
        return values / column["scale"] + column["offset"]

    for coordinate in ("x", "y"):
        marker_values = decode(points[coordinate])
        np.testing.assert_allclose(
            decode(segments[f"{coordinate}0"]), marker_values[:-1], atol=1e-06
        )
        np.testing.assert_allclose(
            decode(segments[f"{coordinate}1"]), marker_values[1:], atol=1e-06
        )
    assert '"kind":"segments"' in grid.to_html()
    assert grid.to_png(scale=1).startswith(b"\x89PNG")
    automatic = plotting.plot_traces_all(analyzer, legends)
    default_spec, _ = automatic.figures[0].build_payload(px_width=50)
    assert default_spec["traces"][0]["tier"] == "decimated"


def test_exact_lines_preserve_gaps_and_line_only_legend():
    chart = plotting.plot_traces(
        [[0, 1, 2, 3, 4]],
        [[0, 1, np.nan, 3, 4]],
        ["A"],
        ["red"],
        [True],
        trace_config=plotting.TraceConfig(marker_size=0, exact_lines=True),
    )
    spec, _ = chart.figure().build_payload()
    assert spec["traces"][0]["name"] == "A"
    assert spec["traces"][0]["n_marks"] == 2
    assert chart.figure().legend_options.get("toggle", True) is True
    assert "<svg" in chart.to_svg()


def test_dataclass_configs_apply_without_mutation():
    plot = plotting.PlotConfig(
        plot_width=640,
        plot_height=320,
        font_size=12,
        title="Custom traces",
        x_range=(2, 8),
        y_range=(-2, 2),
    )
    axes = plotting.AxisConfig(
        showgrid_x=False,
        n_x_axis_ticks=4,
        axis_linewidth=3,
        axis_tickwidth=1,
        axis_gridwidth=2,
        tick_length=4,
        x_label="Seconds",
        y_label="Response",
        x_bounds=(0, 10),
    )
    legend = plotting.LegendConfig(show=False, loc="upper left", font_size=9)
    trace = plotting.TraceConfig(
        marker_size=6, line_width=0.5, opacity=0.4, exact_lines=True, max_points=100
    )
    original = [asdict(config) for config in (plot, axes, legend, trace)]
    x = np.linspace(0, 10, 12001)
    chart = plotting.plot_traces(
        [x],
        [np.sin(x)],
        ["A"],
        ["red"],
        [True],
        plot_config=plot,
        axis_config=axes,
        legend_config=legend,
        trace_config=trace,
    )
    spec, _ = chart.figure().build_payload()
    assert chart.width == 640 and chart.height == 320
    assert spec["title"] == "Custom traces"
    assert spec["x_axis"]["range"] == [2, 8]
    assert spec["y_axis"]["range"] == [-2, 2]
    assert spec["x_axis"]["label"] == "Seconds"
    assert spec["x_axis"]["tick_count"] == 4
    assert spec["x_axis"]["bounds"] == [0, 10]
    assert spec["x_axis"]["style"]["axis_width"] == 3
    assert spec["x_axis"]["style"]["grid_opacity"] == 0
    assert spec["traces"][0]["kind"] == "segments"
    assert spec["traces"][1]["n_marks"] <= 100
    assert chart.figure().traces[1].size_ch.constant == 6
    assert chart.figure().show_legend is False
    assert [asdict(config) for config in (plot, axes, legend, trace)] == original


def test_partial_configs_preserve_each_helpers_defaults(analyzer):
    config = plotting.PlotConfig(font_size=11)
    chart = plotting.plot_traces(
        [[0, 1]], [[1, 2]], ["A"], ["red"], [True], plot_config=config
    )
    grid = plotting.plot_steady_state(
        analyzer, plot_config=config, axis_config=plotting.AxisConfig(n_y_axis_ticks=5)
    )
    assert (chart.width, chart.height) == (900, 420)
    assert (grid.width, grid.grid_height + 24) == (1500, 1000)
    spec, _ = grid.figures[0].build_payload()
    assert spec["x_axis"]["scale"] == "log"
    assert spec["y_axis"]["tick_count"] == 5
    assert config.plot_width is None and config.plot_height is None
    assert config.title is None


def test_layout_configs_share_domains_and_preserve_log_axes(analyzer):
    original = analyzer.fittings["experiment"]
    analyzer.fittings = {}
    for i in range(3):
        fit = deepcopy(original)
        fit.names = [f"experiment_{i}_1", f"experiment_{i}_2"]
        fit.lig_conc_lst_per_id = [
            [value * 10**i for value in group] for group in fit.lig_conc_lst_per_id
        ]
        fit.signal_ss = [[value * 100**i for value in group] for group in fit.signal_ss]
        analyzer.fittings[str(i)] = fit
    config = plotting.PlotConfig(plot_width=1000, plot_height=700, title="Shared")
    layout = plotting.LayoutConfig(
        ncols=2, gap=12, shared_xaxes=True, shared_yaxes=True, show_subplot_titles=False
    )
    axes = plotting.AxisConfig(x_label="Concentration", y_label="Response")
    grid = plotting.plot_steady_state(
        analyzer,
        plot_config=config,
        axis_config=axes,
        layout_config=layout,
        legend_config=plotting.LegendConfig(show=False),
    )
    assert grid.cols == 2 and grid.rows == 2 and (grid.gap == 12)
    specs = [panel.build_payload()[0] for panel in grid.figures]
    assert all((spec["x_axis"]["scale"] == "log" for spec in specs))
    assert all(
        (spec["x_axis"]["range"] == specs[0]["x_axis"]["range"] for spec in specs)
    )
    assert all(
        (spec["y_axis"]["range"] == specs[0]["y_axis"]["range"] for spec in specs)
    )
    assert specs[0]["x_axis"]["label"] == ""
    assert specs[2]["x_axis"]["label"] == "Concentration"
    assert specs[1]["y_axis"]["label"] == ""
    assert all(
        (panel.title is None and (not panel.show_legend) for panel in grid.figures)
    )
    assert "Shared" in grid.to_svg()
    stacked = plotting.plot_steady_state(
        analyzer,
        plot_config=config,
        layout_config=replace(layout, stacked=True, ncols=None),
    )
    assert stacked.cols == 1 and stacked.rows == 3
    assert layout.ncols == 2 and layout.stacked is False


@pytest.mark.parametrize(
    "helper", ["plot_steady_state", "plot_association_dissociation"]
)
def test_trace_configs_control_fit_color_width_and_sampling(analyzer, helper):
    trace = plotting.TraceConfig(
        marker_size=7,
        line_width=0.5,
        opacity=0.3,
        fit_color="#f97316",
        fit_line_width=4,
        max_points=8,
    )
    grid = getattr(plotting, helper)(
        analyzer, trace_config=replace(trace, show_fit=True)
    )
    fitted = [t for panel in grid.figures for t in panel.traces if t.kind == "line"]
    observed = [
        t for panel in grid.figures for t in panel.traces if t.kind == "scatter"
    ]
    assert fitted and observed
    assert all(
        (t.style["color"] == trace.fit_color and t.style["width"] == 4 for t in fitted)
    )
    assert all(
        (t.size_ch.constant == 7 and t.style["opacity"] == 0.3 for t in observed)
    )
    if helper == "plot_association_dissociation":
        assert all((len(t.x.values) <= 8 for t in fitted + observed))


@pytest.mark.parametrize("grid_plot", [False, True])
def test_config_fig_dataclass_export_after_cached_render(analyzer, tmp_path, grid_plot):
    figure = (
        plotting.plot_steady_state(analyzer)
        if grid_plot
        else plotting.plot_traces([[1, 2]], [[3, 4]], ["A"], ["red"], [True])
    )
    figure.to_svg()
    config = plotting.PlotConfig(
        plot_width=600,
        plot_height=400,
        font_size=10,
        title="Resized",
        x_range=(1, 9),
        plot_type="svg",
    )
    resized = plotting.config_fig(
        figure, tmp_path / "configured", save=True, plot_config=config
    )
    assert resized.width == 600 and resized.title == "Resized"
    panels = resized.figures if grid_plot else [resized.figure()]
    assert all(
        (panel.build_payload()[0]["x_axis"]["range"] == [1, 9] for panel in panels)
    )
    assert all(
        (panel.chrome_styles["tick_label"]["font_size"] == 10 for panel in panels)
    )
    assert all(panel.export_options["formats"] == ["svg"] for panel in panels)
    assert (tmp_path / "configured.svg").is_file()
    if grid_plot:
        assert figure.figures[0].axis_options["x"]["domain"] is None
        assert figure.title == "Steady-state"


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (plotting.PlotConfig, {"plot_width": 0}),
        (plotting.PlotConfig, {"x_range": (2, 1)}),
        (plotting.AxisConfig, {"x_scale": "unknown"}),
        (plotting.LayoutConfig, {"ncols": 0}),
        (plotting.LayoutConfig, {"stacked": True, "ncols": 2}),
        (plotting.LayoutConfig, {"vertical_spacing": -0.1}),
        (plotting.LayoutConfig, {"vertical_spacing": 1}),
        (plotting.PlotConfig, {"font_size": 0}),
        (plotting.TraceConfig, {"marker_size": -1}),
        (plotting.TraceConfig, {"line_width": np.inf}),
        (plotting.TraceConfig, {"opacity": 1.1}),
        (plotting.TraceConfig, {"max_points": 0}),
        (plotting.LegendConfig, {"ncols": 0}),
    ],
)
def test_invalid_config_values(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_navigation_bounds_follow_data_without_changing_config():
    axes = plotting.AxisConfig(x_bounds="data", y_bounds="data")
    chart = plotting.plot_traces(
        [[0, 2]], [[-1, 3]], ["A"], ["red"], [True], axis_config=axes
    )
    spec, _ = chart.figure().build_payload()
    for axis_id, values in (("x", [0, 2]), ("y", [-1, 3])):
        axis = spec[f"{axis_id}_axis"]
        assert axis["bounds"] == axis["range"]
        assert axis["bounds"][0] <= min(values)
        assert axis["bounds"][1] >= max(values)
    assert axes.x_bounds == axes.y_bounds == "data"


@pytest.mark.parametrize(
    "parameter, expected",
    [
        ("plot_config", "PlotConfig"),
        ("axis_config", "AxisConfig"),
        ("layout_config", "LayoutConfig"),
        ("legend_config", "LegendConfig"),
        ("trace_config", "TraceConfig"),
    ],
)
def test_helpers_reject_wrong_configuration_types(analyzer, parameter, expected):
    with pytest.raises(TypeError, match=f"Expected {expected}, got dict"):
        plotting.plot_steady_state(analyzer, **{parameter: {}})


@pytest.mark.parametrize("axis_id, lower", [("x", 0), ("y", -1)])
def test_config_fig_rejects_nonpositive_log_ranges(axis_id, lower):
    chart = plotting.plot_traces(
        [[1, 2]],
        [[3, 4]],
        ["A"],
        ["red"],
        [True],
        axis_config=plotting.AxisConfig(x_scale="log", y_scale="log"),
    )
    chart.to_svg()  # Reconfigure an already-cached engine.
    config = plotting.PlotConfig(**{f"{axis_id}_range": (lower, 10)})
    with pytest.raises(ValueError, match=f"{axis_id}_range must be positive"):
        plotting.config_fig(chart, "unused", config)
    assert chart.figure().axis_options[axis_id]["domain"] is None
    assert "<svg" in chart.to_svg()


def test_config_fig_rejects_unsupported_figure_types():
    with pytest.raises(TypeError, match="Expected an XY Chart or FacetGrid"):
        plotting.config_fig(object(), "unused")


def test_stacked_spacing_depends_on_rows_and_pixel_gap_overrides_it(analyzer):
    fit = analyzer.fittings["experiment"]
    analyzer.fittings = {str(i): deepcopy(fit) for i in range(3)}
    layout = plotting.LayoutConfig(stacked=True, vertical_spacing=0.5)
    with pytest.raises(ValueError, match="too large for the subplot rows"):
        plotting.plot_steady_state(analyzer, layout_config=layout)
    grid = plotting.plot_steady_state(analyzer, layout_config=replace(layout, gap=8))
    assert grid.rows == 3 and grid.cols == 1 and grid.gap == 8


@pytest.mark.parametrize("font_size, expected", [(10, 8), (14, 8), (18, 11), (30, 18)])
def test_default_panel_gap_follows_font_size(font_size, expected):
    grid = _nine_panel_kinetics(plot_config=plotting.PlotConfig(font_size=font_size))
    assert grid.gap == expected
    # Explicit settings still win: a fraction of the height, or pixels.
    spaced = _nine_panel_kinetics(
        layout_config=plotting.LayoutConfig(vertical_spacing=0.08)
    )
    assert spaced.gap == 80
    assert _nine_panel_kinetics(layout_config=plotting.LayoutConfig(gap=0)).gap == 0


@pytest.mark.parametrize("groups", [[], [[]], [[10], []]])
def test_kinetics_rejects_empty_smax_groups(analyzer, groups):
    analyzer.fittings["experiment"].lig_conc_lst_per_id = groups
    with pytest.raises(
        ValueError, match="Each Smax group must contain at least one trace"
    ):
        plotting.plot_association_dissociation(analyzer)


@pytest.mark.parametrize("plot_fit", [False, True])
def test_steady_state_log_axes_and_sorted_fit(analyzer, plot_fit):
    grid = plotting.plot_steady_state(
        analyzer, trace_config=plotting.TraceConfig(show_fit=plot_fit)
    )
    panel = grid.figures[0]
    spec, _ = panel.build_payload()
    assert spec["x_axis"]["scale"] == "log"
    assert panel.title == "experiment"
    lines = [trace for trace in panel.traces if trace.kind == "line"]
    assert len(lines) == (2 if plot_fit else 0)
    if plot_fit:
        np.testing.assert_equal(lines[0].x.values, [1.0, 10.0])
        np.testing.assert_equal(lines[0].y.values, [3.0, 7.0])
    assert "<svg" in grid.to_svg()


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("with_fit", [False, True])
def test_kinetics_grouping_fits_sampling_and_smoothing(
    analyzer, split, smooth, with_fit
):
    fit = analyzer.fittings["experiment"]
    original = [values.copy() for values in fit.assoc_lst]
    if not with_fit:
        fit.signal_assoc_fit = fit.signal_disso_fit = None
    grid = plotting.plot_association_dissociation(
        analyzer,
        layout_config=plotting.LayoutConfig(split_by_smax_id=split),
        trace_config=plotting.TraceConfig(smooth_curves=smooth, max_points=10),
    )
    assert len(grid.figures) == (2 if split else 1)
    assert all((panel.legend_options["toggle"] is False for panel in grid.figures))
    traces = [trace for panel in grid.figures for trace in panel.traces]
    # Smoothed lines need an invisible mark to carry the colorbar's range.
    color_marks = [t for t in traces if t.style.get("opacity") == 0]
    assert len(color_marks) == (1 if smooth else 0)
    traces = [t for t in traces if t not in color_marks]
    assert len(traces) == (12 if with_fit else 6)
    for trace in traces:
        assert len(trace.x.values) == len(trace.y.values) <= 10
        assert trace.name is None
        if trace.kind == "line" and trace.style["color"] == "black":
            assert with_fit
        else:
            assert trace.kind == ("line" if smooth else "scatter")
    for before, after in zip(original, fit.assoc_lst):
        np.testing.assert_equal(before, after)
    assert "<svg" in grid.to_svg()


def test_plate_layout_metadata_and_reverse_axis():
    exp = SimpleNamespace(
        sample_row=["A", "B"],
        sample_column=[1, 2],
        sample_type=["SAMPLE", "BUFFER"],
        sample_id=["one", "two"],
        sample_conc_labeled=["1 μM", "0 μM"],
    )
    analyzer = SimpleNamespace(experiments={"plate": exp})
    chart = plotting.plot_plate_info(analyzer, "plate")
    spec, _ = chart.figure().build_payload()
    assert spec["y_axis"]["reverse"] is True
    assert "ID: one; Type: SAMPLE; 1 μM" == chart.figure().traces[0].name
    assert [
        node.text for node in chart.children if isinstance(node, xy.Annotation)
    ] == ["S", "B"]
    assert "<svg" in chart.to_svg()
    exp.sample_row = None
    with pytest.raises(AttributeError, match="sample_row"):
        plotting.plot_plate_info(analyzer, "plate")


@pytest.mark.parametrize("grid_plot", [False, True])
@pytest.mark.parametrize("format", ["png", "svg", "html"])
def test_config_and_real_exports(analyzer, tmp_path, grid_plot, format):
    figure = (
        plotting.plot_steady_state(analyzer)
        if grid_plot
        else plotting.plot_traces([[1, 2]], [[3, 4]], ["A"], ["red"], [True])
    )
    figure.to_svg()
    result = plotting.config_fig(
        figure,
        tmp_path / "chart",
        save=True,
        plot_config=plotting.PlotConfig(
            plot_width=400, plot_height=300, plot_type=format
        ),
    )
    assert result.width == 400
    output = (tmp_path / f"chart.{format}").read_bytes()
    if format == "png":
        assert output.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", output[16:24])
        assert width == 800
        assert height > 0
    else:
        assert (b"<svg" if format == "svg" else b"<!doctype html>") in output
    with pytest.raises(RuntimeError, match="Error saving image"):
        plotting.config_fig(
            result,
            tmp_path / "bad",
            save=True,
            plot_config=plotting.PlotConfig(plot_type="invalid"),
        )


_NUMBER = r"(-?[\d.]+)"


def _text_width(text, size):
    from xy import _textblock

    return _textblock.measure(text, float(size)).width


def _y_title_and_tick_label_extents(svg):
    """Horizontal extents of the rotated y-axis title and its tick labels."""
    import re

    from xy import _textblock

    title = re.search(
        r'<text x="([\d.]+)"[^>]*font-size="([\d.]+)"[^>]*rotate\(-90[^>]*>', svg
    )
    block = _textblock.measure("Signal (a.u.)", float(title.group(2)))
    # A -90 degree title grows to the left of its baseline.
    title_right = float(title.group(1)) + block.descent
    tick_lefts = [
        float(x) - _text_width(text, size)
        for x, size, text in re.findall(
            r'<text x="([\d.]+)"[^>]*font-size="([\d.]+)" text-anchor="end">'
            + _NUMBER
            + "</text>",
            svg,
        )
    ]
    return title_right, min(tick_lefts)


def _x_title_and_tick_label_extents(svg):
    """Vertical extents of the x-axis title and the x tick labels above it."""
    import re

    from xy import _textblock

    title = re.search(
        r'<text x="[\d.]+" y="([\d.]+)"[^>]*font-size="([\d.]+)"[^>]*>Time \(s\)</text>',
        svg,
    )
    title_top = (
        float(title.group(1))
        - _textblock.measure("Time (s)", float(title.group(2))).ascent
    )
    tick_bottoms = [
        float(y) + _textblock.measure(text, float(size)).descent
        for y, size, text in re.findall(
            r'<text x="[\d.]+" y="([\d.]+)"[^>]*font-size="([\d.]+)" '
            r'text-anchor="middle">' + _NUMBER + "</text>",
            svg,
        )
    ]
    return title_top, max(tick_bottoms)


def _legend_box_and_label_right_edges(svg, label):
    """Right edges of the legend background and of its widest label."""
    import re

    box = re.search(
        r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)"[^>]*'
        r'fill="rgba\(128,128,128,0\.08\)"',
        svg,
    )
    text = re.search(
        r'<text x="([\d.]+)"[^>]*font-size="([\d.]+)"[^>]*>' + re.escape(label),
        svg,
    )
    return (
        float(box.group(1)) + float(box.group(2)),
        float(text.group(1)) + _text_width(label, text.group(2)),
    )


@pytest.mark.parametrize("grid_plot", [False, True])
def test_axis_titles_and_legend_fit_at_large_font_sizes(analyzer, grid_plot):
    """XY sizes gutters and legends from its own styles, which must hold the font."""
    axis_config = plotting.AxisConfig(x_label="Time (s)", y_label="Signal (a.u.)")
    if grid_plot:
        label = "experiment_sensor_1"
        figure = plotting.config_fig(
            plotting.plot_steady_state(analyzer, axis_config=axis_config),
            "unused",
            plot_config=plotting.PlotConfig(font_size=20),
        )
        engines = figure.figures
    else:
        label = "0.0056 µM"
        figure = plotting.plot_traces(
            [np.linspace(0, 10000, 50)],
            [np.linspace(0, 0.4, 50)],
            [label],
            ["red"],
            [True],
            plot_config=plotting.PlotConfig(plot_height=1000, font_size=18),
            axis_config=axis_config,
        )
        engines = [figure.figure()]

    expected = 20 if grid_plot else 18
    for engine in engines:
        style = engine.axis_options["y"]["style"]
        assert style["tick_label_size"] == expected
        assert style["label_size"] == expected
        assert engine.legend_options["style"] == {"fontSize": f"{expected}px"}

    svg = figure.to_svg()
    title_right, tick_left = _y_title_and_tick_label_extents(svg)
    assert title_right < tick_left
    title_top, tick_bottom = _x_title_and_tick_label_extents(svg)
    assert title_top > tick_bottom
    box_right, label_right = _legend_box_and_label_right_edges(svg, label)
    assert label_right <= box_right


def _legend_left_and_data_right(engine, x_max):
    """Pixel x of the legend box's left edge and of the last observation."""
    import re

    svg = engine.to_svg()
    plot = re.search(
        r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)" height="[\d.]+"/>', svg
    )
    legend = re.search(
        r'<rect x="([\d.]+)" y="[\d.]+" width="[\d.]+"[^>]*'
        r'fill="rgba\(128,128,128,0\.08\)"',
        svg,
    )
    low, high = engine.build_payload()[0]["x_axis"]["range"]
    plot_x, plot_w = float(plot.group(1)), float(plot.group(2))
    return float(legend.group(1)), plot_x + (x_max - low) / (high - low) * plot_w


@pytest.mark.parametrize("font_size", [12, 18])
@pytest.mark.parametrize("width", [450, 900])
def test_sensorgram_legend_sits_right_of_the_data(font_size, width):
    times = np.linspace(0, 650, 60)
    chart = plotting.plot_traces(
        [times] * 2,
        [np.linspace(0, 1, 60)] * 2,
        ["0.0056 µM", "0.13 µM"],
        ["red", "blue"],
        [True, True],
        plot_config=plotting.PlotConfig(plot_width=width, font_size=font_size),
    )
    legend_left, data_right = _legend_left_and_data_right(chart.figure(), 650)
    assert data_right < legend_left


def test_sensorgram_legend_room_respects_explicit_settings():
    def x_range(**configs):
        chart = plotting.plot_traces(
            [np.linspace(0, 650, 60)], [np.linspace(0, 1, 60)], ["A"], ["red"], [True],
            **configs,
        )
        return chart.figure().build_payload()[0]["x_axis"]["range"]

    assert x_range()[1] > 700
    assert x_range(plot_config=plotting.PlotConfig(x_range=(0, 650))) == [0, 650]
    # Left-side or hidden legends need no room.
    assert x_range(legend_config=plotting.LegendConfig(loc="upper left"))[1] < 700
    assert x_range(legend_config=plotting.LegendConfig(show=False))[1] < 700


def _nine_panel_kinetics(n_panels=9, **configs):
    times = np.linspace(0, 300, 31)
    fit = SimpleNamespace(
        names=[f"sample_{i}" for i in range(n_panels)],
        lig_conc_lst_per_id=[[0.13, 0.0056]] * n_panels,
        time_assoc_lst=[times] * (2 * n_panels),
        time_disso_lst=[times + 300] * (2 * n_panels),
        assoc_lst=[times / 300] * (2 * n_panels),
        disso_lst=[1 - times / 300] * (2 * n_panels),
        signal_assoc_fit=None,
        signal_disso_fit=None,
    )
    return plotting.plot_association_dissociation(
        SimpleNamespace(fittings={"sample": fit}), **configs
    )


def test_kinetics_panels_share_ticks():
    import re

    grid = _nine_panel_kinetics()
    ranges = {tuple(p.build_payload()[0]["x_axis"]["range"]) for p in grid.figures}
    assert len(ranges) == 1
    ticks = {
        tuple(re.findall(r'text-anchor="middle">([\d.]+)</text>', p.to_svg()))
        for p in grid.figures
    }
    assert len(ticks) == 1


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("font_size", [14, 18])
def test_kinetics_colorbar_is_shown_once_with_concentration_labels(smooth, font_size):
    import re

    from xy import _textblock

    grid = _nine_panel_kinetics(
        plot_config=plotting.PlotConfig(font_size=font_size),
        trace_config=plotting.TraceConfig(smooth_curves=smooth),
    )
    # 3 columns: only the top-right panel draws the shared colorbar.
    with_colorbar = [i for i, p in enumerate(grid.figures) if p.colorbar_options]
    assert with_colorbar == [2]
    options = grid.figures[2].colorbar_options
    assert options["domain"] == pytest.approx([np.log10(0.0056), np.log10(0.13)])
    assert options["tick_labels"] == ["5.6", "27", "130"]
    assert options["label"] == "[Ligand] (nM)"
    assert all(not p.show_legend or not any(t.name for t in p.traces) for p in grid.figures)

    svg = grid.figures[2].to_svg()
    title = re.search(
        r'<text x="([\d.]+)"[^>]*font-size="([\d.]+)"[^>]*>\[Ligand\] \(nM\)</text>',
        svg,
    )
    title_left = float(title.group(1)) - _textblock.measure("[", float(title.group(2))).ascent
    for label in options["tick_labels"]:
        tick = re.search(
            rf'<text x="([\d.]+)"[^>]*font-size="([\d.]+)"[^>]*>{re.escape(label)}</text>',
            svg,
        )
        assert tick, label
        right = float(tick.group(1)) + _text_width(label, tick.group(2))
        assert right < title_left


@pytest.mark.parametrize(
    "low, high, unit, labels",
    [
        (0.0056, 0.13, "nM", ["5.6", "27", "130"]),
        (0.5, 20.0, "μM", ["0.5", "3.2", "20"]),
        (2000.0, 5000.0, "mM", ["2", "3.2", "5"]),
        (1e-6, 1e-4, "pM", ["1", "10", "100"]),
        (0.05, 0.05, "nM", ["50"]),
    ],
)
def test_concentration_colorbar_picks_a_short_unit(low, high, unit, labels):
    colorbar = plotting._concentration_colorbar(low, high)
    assert colorbar["title"] == f"[Ligand] ({unit})"
    assert colorbar["labels"] == labels
    assert colorbar["domain"][0] < colorbar["domain"][1]


def _plot_rect(engine):
    import re

    rect = re.search(
        r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)" height="[\d.]+"/>',
        engine.to_svg(),
    )
    return float(rect.group(1)), float(rect.group(2))


@pytest.mark.parametrize("n_panels, cols", [(9, 3), (18, 6)])
def test_colorbar_panel_keeps_the_same_plot_width(n_panels, cols):
    import re

    grid = _nine_panel_kinetics(n_panels)
    assert grid.cols == cols
    # Column 0 also holds the y title; compare the other columns.
    widths = {
        round(_plot_rect(panel)[1])
        for i, panel in enumerate(grid.figures)
        if i % cols
    }
    assert len(widths) == 1
    # Ordinary panels in the colorbar's column keep the ordinary width and
    # leave the room under the colorbar empty.
    assert {panel.width for i, panel in enumerate(grid.figures) if i != cols - 1} == {
        grid.panel_width
    }
    assert grid.figures[cols - 1].width == grid.panel_width + 104

    svg = grid.to_svg()
    panels = re.findall(r'<svg x="(\d+)" y="(\d+)" width="(\d+)"', svg)
    assert len(panels) == n_panels
    right_edges = [int(x) + int(w) for x, _y, w in panels]
    assert max(right_edges) <= grid.width
    first_row = sorted(int(x) for x, y, _w in panels if y == panels[0][1])
    assert all(b - a >= grid.panel_width for a, b in zip(first_row, first_row[1:]))

    png = grid.to_png(scale=1)
    assert struct.unpack(">II", png[16:24])[0] == grid.width
    html = grid.to_html()
    assert f"minmax(0, {grid.panel_width + 104}fr)" in html
    assert f"#xy-facet-{2 * cols - 1}{{width:" in html


def test_config_fig_keeps_the_wide_colorbar_panel():
    grid = _nine_panel_kinetics()
    resized = plotting.config_fig(
        grid, "unused", plot_config=plotting.PlotConfig(plot_width=900)
    )
    assert type(resized) is type(grid) and resized.width == 900
    assert resized.figures[2].width == resized.panel_width + 104
    widths = [
        _plot_rect(panel)[1]
        for i, panel in enumerate(resized.figures)
        if i % resized.cols
    ]
    # XY may widen an ordinary panel's right margin by a few pixels for an
    # overhanging x tick label; the colorbar panel's margin already covers it.
    assert max(widths) - min(widths) <= 6


def test_steady_state_legend_names_only_the_smax_id_when_there_are_several():
    fit = SimpleNamespace(
        names=["wt dtt 2.5 mM - reg (b)_id_7", "wt dtt 2.5 mM - reg (b)_id_8"],
        lig_conc_lst_per_id=[[0.001, 0.13]] * 2,
        signal_ss=[[0.1, 0.3]] * 2,
        signal_ss_fit=None,
    )
    single = SimpleNamespace(
        names=["wt dtt 2.5 mM - imd_id_6"],
        lig_conc_lst_per_id=[[0.001, 0.13]],
        signal_ss=[[0.1, 0.3]],
        signal_ss_fit=None,
    )
    grid = plotting.plot_steady_state(
        SimpleNamespace(fittings={"sample": fit, "single": single})
    )
    assert [t.name for t in grid.figures[0].traces] == ["Smax ID 7", "Smax ID 8"]
    # A single Smax ID gets no legend entry.
    assert [t.name for t in grid.figures[1].traces] == [None]
    assert '>Smax ID' not in grid.figures[1].to_svg()
    assert plotting._smax_label("custom name") == "custom name"


def test_steady_state_pads_axes_labels_decades_and_puts_legend_upper_left():
    import re

    fit = SimpleNamespace(
        names=[f"sample_{i}" for i in range(6)],
        lig_conc_lst_per_id=[[0.001, 0.0056, 0.03, 0.13]] * 6,
        signal_ss=[[0.02, 0.1, 0.25, 0.38]] * 6,
        signal_ss_fit=None,
    )
    grid = plotting.plot_steady_state(
        SimpleNamespace(fittings={"sample": fit}),
        plot_config=plotting.PlotConfig(plot_width=1340, font_size=18),
    )
    for panel in grid.figures:
        spec = panel.build_payload()[0]
        (x_low, x_high), (y_low, y_high) = spec["x_axis"]["range"], spec["y_axis"]["range"]
        assert np.log10(0.13 / x_high) < -0.05 and np.log10(x_low / 0.001) < -0.05
        assert y_high - 0.38 > 0.01 and 0.02 - y_low > 0.01
        assert spec["legend"]["loc"] == "upper left"
        labels = re.findall(r'text-anchor="middle">([\d.]+)</text>', panel.to_svg())
        assert labels == ["0.001", "0.01", "0.1"]
    custom = plotting.plot_steady_state(
        SimpleNamespace(fittings={"sample": fit}),
        legend_config=plotting.LegendConfig(loc="lower right"),
    )
    assert custom.figures[0].build_payload()[0]["legend"]["loc"] == "lower right"


def test_empty_and_invalid_inputs(analyzer):
    with pytest.raises(ValueError, match="No experiments or fittings"):
        plotting.plot_steady_state(SimpleNamespace(fittings={}))
    with pytest.raises(ValueError, match="No experiments or fittings"):
        plotting.plot_association_dissociation(SimpleNamespace(fittings={}))
    with pytest.raises(ValueError, match="max_points"):
        plotting.plot_association_dissociation(
            analyzer, trace_config=plotting.TraceConfig(max_points=0)
        )
    analyzer.fittings["experiment"].lig_conc_lst_per_id[0][0] = 0
    with pytest.raises(ValueError, match="concentrations must be finite and positive"):
        plotting.plot_association_dissociation(analyzer)
    with pytest.raises(ValueError, match="plot_width"):
        plotting.config_fig(
            xy.chart(), "unused", plot_config=plotting.PlotConfig(plot_width=0)
        )


def test_missing_xy_dependency(monkeypatch):
    original_import = builtins.__import__

    def without_xy(name, *args, **kwargs):
        if name == "xy":
            raise ModuleNotFoundError("No module named 'xy'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_xy)
    with pytest.raises(ImportError, match="pip install pykingenie"):
        plotting.plot_traces([], [], [], [], [])


def test_octet_data_end_to_end():
    from pykingenie.main import KineticsAnalyzer
    from pykingenie.octet import OctetExperiment

    folder = Path(__file__).resolve().parents[1] / "test_files"
    experiment = OctetExperiment("test")
    experiment.read_sensor_data(
        sorted((str(path) for path in folder.glob("230309*.frd")))
    )
    experiment.read_sample_plate_info(str(folder / "230309_ExpMethod.fmf"))
    analyzer = KineticsAnalyzer()
    analyzer.add_experiment(experiment, "test_octet")
    assert plotting.plot_plate_info(analyzer, "test_octet").to_svg().startswith("<svg")
    grid = plotting.plot_traces_all(analyzer, analyzer.get_legends_table())
    assert len(grid.figures) == 1
    assert len(grid.to_png(scale=1)) > 100
    experiment.align_association(experiment.sensor_names)
    experiment.align_dissociation(experiment.sensor_names)
    experiment.subtraction(experiment.sensor_names[:7], experiment.sensor_names[7])
    analyzer.merge_ligand_conc_df()
    analyzer.generate_fittings(analyzer.combined_ligand_conc_df.iloc[:8].copy())
    analyzer.submit_steady_state_fitting()
    analyzer.submit_kinetics_fitting(
        fitting_model="one_to_one",
        fitting_region="association_dissociation",
        shared_smax=True,
    )
    assert (
        "<svg"
        in plotting.plot_steady_state(
            analyzer, trace_config=plotting.TraceConfig(show_fit=True)
        ).to_svg()
    )
    assert "xy-facet-grid" in plotting.plot_association_dissociation(analyzer).to_html()


def test_legend_room_needs_spread_x_data_and_fits_the_legend_title():
    def x_high(xs, legend_config=None):
        chart = plotting.plot_traces(
            [xs], [np.linspace(0, 1, len(xs))], ["A"], ["red"], [True],
            legend_config=legend_config,
        )
        return chart.figure().build_payload()[0]["x_axis"]["range"][1]

    # A single time point leaves no span to extend.
    assert plotting._legend_room_domain(
        [xy.scatter([5.0, 5.0], [0.0, 1.0], name="A")],
        plotting.LegendConfig(show=True, loc="upper right"),
        14,
        900,
    ) is None
    times = np.linspace(0, 650, 60)
    title = plotting.LegendConfig(title="Sensors on the octet plate")
    assert x_high(times, title) > x_high(times)


def test_first_point_falls_back_to_the_origin_without_data():
    assert plotting._first_point([]) == (0.0, 0.0)
    assert plotting._first_point([xy.scatter([2.0], [3.0])]) == (2.0, 3.0)
    # Annotations such as the residual zero line carry no data points.
    assert plotting._first_point([xy.hline(0), xy.scatter([2.0], [3.0])]) == (2.0, 3.0)


def test_colorbar_grid_writes_svg_and_rejects_bad_scales(tmp_path):
    grid = _nine_panel_kinetics()
    path = tmp_path / "grid.svg"
    assert grid.to_svg(path) == path.read_text()
    with pytest.raises(ValueError, match="scale"):
        grid._compose_rgba(0)


def test_steady_state_keeps_automatic_ticks_within_one_decade():
    import re

    fit = SimpleNamespace(
        names=["sample_id_0"],
        lig_conc_lst_per_id=[[0.02, 0.03, 0.05]],
        signal_ss=[[0.1, 0.2, 0.3]],
        signal_ss_fit=None,
    )
    panel = plotting.plot_steady_state(SimpleNamespace(fittings={"s": fit})).figures[0]
    assert panel.axis_options["x"]["tick_values"] is None
    assert re.findall(r'text-anchor="middle">([\d.]+)</text>', panel.to_svg())


def _kinetics_fit(names):
    times = np.linspace(0, 300, 11)
    n = len(names)
    return SimpleNamespace(
        names=names,
        lig_conc_lst_per_id=[[0.1]] * n,
        signal_ss=[[0.5]] * n,
        signal_ss_fit=None,
        time_assoc_lst=[times] * n,
        time_disso_lst=[times + 300] * n,
        assoc_lst=[times / 300] * n,
        disso_lst=[1 - times / 300] * n,
        signal_assoc_fit=None,
        signal_disso_fit=None,
    )


def test_panel_titles_use_sample_names_and_smax_ids_only_when_needed():
    analyzer = SimpleNamespace(
        fittings={
            "wt - reg": _kinetics_fit(["wt - reg_id_1", "wt - reg_id_2"]),
            "wt - imd": _kinetics_fit(["wt - imd_id_0"]),
        }
    )

    def titles(grid):
        return [panel.title for panel in grid.figures]

    assert titles(plotting.plot_association_dissociation(analyzer)) == [
        "wt - reg (Smax ID 1)",
        "wt - reg (Smax ID 2)",
        "wt - imd",
    ]
    grouped = plotting.LayoutConfig(split_by_smax_id=False)
    assert titles(
        plotting.plot_association_dissociation(analyzer, layout_config=grouped)
    ) == ["wt - reg", "wt - imd"]
    assert titles(plotting.plot_steady_state(analyzer)) == ["wt - reg", "wt - imd"]


def _has_zero_line(panel):
    return {"kind": "rule", "axis": "y", "value": 0.0}.items() <= next(
        a for a in panel.build_payload()[0]["annotations"]
    ).items()


def test_steady_state_residuals_subtract_the_fit_and_skip_unfitted_samples(analyzer):
    unfitted = deepcopy(analyzer.fittings["experiment"])
    unfitted.signal_ss_fit = None
    analyzer.fittings["unfitted"] = unfitted
    grid = plotting.plot_steady_state_residuals(analyzer)

    assert [panel.title for panel in grid.figures] == ["experiment"]
    panel = grid.figures[0]
    traces = [t for t in panel.traces if t.kind == "scatter"]
    # Observed [[8, 2], [5]] minus fitted [[7, 3], [4]].
    np.testing.assert_allclose(traces[0].y.values, [1, -1])
    np.testing.assert_allclose(traces[1].y.values, [1])
    assert [t.name for t in traces] == ["experiment_sensor_1", "experiment_sensor_2"]
    assert _has_zero_line(panel)
    assert panel.axis_options["x"]["type"] == "log"
    assert panel.axis_options["y"]["label"] == "Residual (a.u.)"
    assert panel.build_payload()[0]["legend"]["loc"] == "upper left"
    assert "<svg" in grid.to_svg()

    analyzer.fittings = {"unfitted": unfitted}
    with pytest.raises(ValueError, match="No steady-state fits available"):
        plotting.plot_steady_state_residuals(analyzer)


@pytest.mark.parametrize("split", [False, True])
def test_kinetics_residuals_subtract_the_fit_with_a_shared_colorbar(analyzer, split):
    fit = analyzer.fittings["experiment"]
    grid = plotting.plot_association_dissociation_residuals(
        analyzer,
        layout_config=plotting.LayoutConfig(split_by_smax_id=split),
        trace_config=plotting.TraceConfig(max_points=1000),
    )
    assert len(grid.figures) == (2 if split else 1)
    # The colorbar sits beside the top-right panel.
    assert [i for i, p in enumerate(grid.figures) if p.colorbar_options] == [
        min(grid.cols, len(grid.figures)) - 1
    ]
    traces = [t for panel in grid.figures for t in panel.traces]
    # Observed x**2 minus fitted x**2 + 0.1, for both phases of three traces.
    assert len(traces) == 6
    for trace in traces:
        assert trace.kind == "scatter"
        np.testing.assert_allclose(trace.y.values, -0.1, atol=1e-12)
    assert all(_has_zero_line(panel) for panel in grid.figures)
    assert grid.figures[0].axis_options["y"]["label"] == "Residual (a.u.)"
    assert len(fit.assoc_lst) == 3


def test_kinetics_residuals_smooth_skip_unfitted_traces_and_need_a_fit(analyzer):
    fit = analyzer.fittings["experiment"]
    fit.signal_disso_fit = [None] * 3
    grid = plotting.plot_association_dissociation_residuals(
        analyzer, trace_config=plotting.TraceConfig(smooth_curves=True)
    )
    traces = [t for panel in grid.figures for t in panel.traces]
    lines = [t for t in traces if t.kind == "line"]
    # Association only, as lines; one invisible mark carries the colorbar.
    assert len(lines) == 3
    assert [t.style.get("opacity") for t in traces if t.kind == "scatter"] == [0]
    # Hidden markers still give one panel per fitted Smax group.
    hidden = plotting.plot_association_dissociation_residuals(
        analyzer, trace_config=plotting.TraceConfig(marker_size=0)
    )
    assert len(hidden.figures) == 2
    assert not [t for p in hidden.figures for t in p.traces if t.style.get("opacity")]

    fit.signal_assoc_fit = None
    fit.signal_disso_fit = None
    with pytest.raises(ValueError, match="No kinetics fits available"):
        plotting.plot_association_dissociation_residuals(analyzer)
