"""Helpers for rendering notebook figures in a GitHub-friendly static format."""

from IPython.display import Image, display
import plotly.io as pio


def show_plotly_static(fig, width=1200, height=700, scale=2):
    """Render a Plotly figure as a static PNG in notebook output.

    This is useful for GitHub notebook rendering, where interactive Plotly
    JavaScript figures are not always displayed.
    """
    png_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=scale)
    display(Image(data=png_bytes))
