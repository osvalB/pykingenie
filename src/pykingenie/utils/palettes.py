"""
Color palettes for plotting.

This module provides color palettes that can be used for visualizing data
in plots. These palettes are collections of colors represented as hex values.
"""

__all__ = [
    "VIRIDIS",
    "set1_colors",
    "set3_colors"
]

VIRIDIS = [
    "#440154", "#471063", "#481d6f", "#472a7a", "#414487", "#3c4f8a", "#375a8c",
    "#32648e", "#2a788e", "#26828e", "#228b8d", "#1f958b", "#22a884", "#2cb17e",
    "#3bbb75", "#4ec36b", "#7ad151", "#95d840", "#b0dd2f", "#cae11f", "#fde725"
]
"""
Viridis color palette.

A perceptually uniform color map that is readable by those with colorblindness.
Contains 21 hex color values transitioning from dark purple to yellow.
"""

set1_colors = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#ffff33",  # yellow
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999"   # gray
]
"""
Set1 color palette.

A qualitative color palette suitable for categorical data.
Contains 9 distinct colors that are visually distinguishable.
"""

set3_colors = [
    "#8dd3c7",  # teal
    "#ffffb3",  # light yellow
    "#bebada",  # lavender
    "#fb8072",  # salmon
    "#80b1d3",  # light blue
    "#fdb462",  # peach
    "#b3de69",  # light green
    "#fccde5",  # pink
    "#d9d9d9",  # gray
    "#bc80bd",  # purple
    "#ccebc5",  # mint
    "#ffed6f"   # yellow
]
"""
Set3 color palette.

A qualitative color palette with softer, pastel-like colors.
Contains 12 distinct colors suitable for categorical data with many categories.
"""