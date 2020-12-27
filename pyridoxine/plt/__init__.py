""" plt: plotting snippets

    This is a subpackage of pyridoxine.
    This package, "plt", is supposed to work with matplotlib.
"""

# I don't understand why the first import is necessary;
# with or without it, pyridoxine.plt.defaults.func can be used without any problem
from . import defaults
# explicit import to avoid exposing external modules imported at the level of pyridoxine.plt.func
from .defaults import \
    plt_params, \
    turn_off_minor_labels, \
    astro_style, \
    ax_labeling, \
    cut_space, \
    fig_labeling, \
    add_subplot_axis, \
    add_customized_colorbar, \
    add_aligned_colorbar, \
    make_square, \
    get_ax_size_in_pixels, \
    exact_size_figure, \
    draw_grid
