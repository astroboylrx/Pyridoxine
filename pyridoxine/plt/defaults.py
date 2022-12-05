""" Provide default setup for matplotlib.pyplot """

from astropy.visualization import astropy_mpl_style
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plt_params(size="large"):
    """ Give matplotlib params based on size """

    if size in ["pre", "presentation", "slide", "ppt", "talk"]:
        new_params = {
            'figure.figsize': (16, 12),
            'savefig.dpi': 300,
            'lines.linewidth': 1.5,
            'axes.labelsize': 48,
            'axes.linewidth': 1.0,
            'axes.titlesize': 52,
            'xtick.labelsize': 48,
            'ytick.labelsize': 48,
            'legend.fontsize': 44,
            'legend.frameon': True,
            'legend.handlelength': 1.5,
            'xtick.top': True,
            'xtick.direction': "in",
            'xtick.minor.visible': True,
            'xtick.major.size': 10,
            'xtick.minor.size': 5,
            'xtick.major.width': 1.5,
            'xtick.minor.width': 1,
            'ytick.right': True,
            'ytick.direction': "in",
            'ytick.minor.visible': True,
            'ytick.major.size': 10,
            'ytick.minor.size': 5,
            'ytick.major.width': 1.5,
            'ytick.minor.width': 1
        }
    elif size in ["paper", "draft", "ms", "manuscript", "aas"]:
        new_params = {
            'figure.figsize': (14, 10),
            'savefig.dpi': 300,
            'lines.linewidth': 2,
            'axes.labelsize': 32,
            'axes.linewidth': 1.0,
            'axes.titlesize': 36,
            'xtick.labelsize': 32,
            'ytick.labelsize': 32,
            'legend.fontsize': 30,
            'legend.frameon': True,
            'legend.handlelength': 1.5,
            'xtick.top': True,
            'xtick.direction': "in",
            'xtick.minor.visible': True,
            'xtick.major.size': 8,
            'xtick.minor.size': 4,
            'xtick.major.width': 1.5,
            'xtick.minor.width': 1,
            'ytick.right': True,
            'ytick.direction': "in",
            'ytick.minor.visible': True,
            'ytick.major.size': 8,
            'ytick.minor.size': 4,
            'ytick.major.width': 1.5,
            'ytick.minor.width': 1
        }
    elif size in ["l", "large", "big", "Large", "Big"]:
        new_params = {
            'figure.figsize': (12, 9),
            'savefig.dpi': 300,
            'lines.linewidth': 1.0,
            'axes.labelsize': 20,
            'axes.linewidth': 0.75,
            'axes.titlesize': 20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 18,
            'legend.frameon': True,
            'legend.handlelength': 1.5,
            'xtick.top': True,
            'ytick.right': True
            }
    elif size in ["m", "medium", "Medium", "middle", "Middle"]:
        new_params = {
            'figure.figsize': (10, 8),
            'savefig.dpi': 300,
            'lines.linewidth': 1.0,
            'axes.labelsize': 18,
            'axes.linewidth': 0.75,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'axes.titlesize': 18,
            'legend.fontsize': 16,
            'legend.frameon': True,
            'legend.handlelength': 1.5,
            'xtick.top': True,
            'ytick.right': True
            }
    elif size in ["s", "small", "Small"]:
        new_params = {
            'figure.figsize': (8, 6),
            'savefig.dpi': 300,
            'lines.linewidth': 0.75,
            'axes.labelsize': 14,
            'axes.linewidth': 0.75,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'axes.titlesize': 14,
            'legend.fontsize': 14,
            'legend.frameon': True,
            'legend.handlelength': 1.5,
            'xtick.top': True,
            'ytick.right': True
            }
    else:
        raise ValueError("Wrong size specified: ", size)

    plt.rcParams.update(new_params)


def turn_off_minor_labels(ax):
    """ Turn off the tick labels for minor ticks """

    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

def minor_ticks_on_log_axis(ax, which=None):

    if which == 'x' or which == 'both':
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    if which == 'y' or which == 'both':
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))

def astro_style():
    """ Feed astropy style to matplotlib """

    plt.style.use('default')
    plt.style.use(astropy_mpl_style)


def ax_labeling(ax, **kwargs):
    """
    Set labels and title for a plot
    :param ax: should be a class 'matplotlib.axes._subplots.AxesSubplot'
    :param kwargs: accepting x/y/z, x/y/zl, x/y/zlabel, x/y/z_label, title
    :return: None
    """

    if len(kwargs) == 0:
        return None

    x_label, y_label, z_label = None, None, None
    title = None

    if 'x' in kwargs:
        x_label = kwargs.get('x')
    if 'xl' in kwargs:
        x_label = kwargs.get('xl')
    if 'xlabel' in kwargs:
        x_label = kwargs.get('xlabel')
    if 'x_label' in kwargs:
        x_label = kwargs.get('x_label')

    if 'y' in kwargs:
        y_label = kwargs.get('y')
    if 'yl' in kwargs:
        y_label = kwargs.get('yl')
    if 'ylabel' in kwargs:
        y_label = kwargs.get('ylabel')
    if 'y_label' in kwargs:
        y_label = kwargs.get('y_label')

    if 'z' in kwargs:
        z_label = kwargs.get('z')
    if 'zl' in kwargs:
        z_label = kwargs.get('zl')
    if 'zlabel' in kwargs:
        z_label = kwargs.get('zlabel')
    if 'z_label' in kwargs:
        z_label = kwargs.get('z_label')

    if 't' in kwargs:
        title = kwargs.get('t')
    if 'title' in kwargs:
        title = kwargs.get('title')

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if z_label is not None:
        ax.set_zlabel(z_label)
    if title is not None:
        ax.set_title(title)

    return None


def cut_space(fig, ax, space=None, tickside=None):
    """
    Fine-tuning subplots in a figure to cut space between them
    :param fig: Figure object
    :param ax: Axes object
    :param space: two element array specifying hspace and wspace
    :return:
    """

    if space is None:
        space = [0, 0]
    if tickside is None:
        tickside = 'x'

    fig.subplots_adjust(hspace=space[0], wspace=space[1])

    if ax.ndim == 2:
        # hide x ticks for top plots; hide y ticks for right plots
        plt.setp([a.get_xticklabels() for a in ax[:-1, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in ax[:, 1:]], visible=False)
    elif ax.ndim == 1:
        if tickside == 'x':
            plt.setp([a.get_xticklabels() for a in ax[:-1]], visible=False)
        if tickside == 'y':
            plt.setp([a.get_yticklabels() for a in ax[1:]], visible=False)


def fig_labeling(fig, **kwargs):
    """
    Set labels and title for a figure with subplots
    :param fig: Figure object
    :param kwargs: will be passed to ax_labeling
    :return: None
    """

    # Creating a big subplot to cover the two subplots and then set the common labels.
    bigax = fig.add_subplot(111)
    # Turn off axis lines and ticks of the big subplot
    bigax.spines['top'].set_visible(False)
    bigax.spines['bottom'].set_visible(False)
    bigax.spines['left'].set_visible(False)
    bigax.spines['right'].set_visible(False)
    bigax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bigax.tick_params(axis=u'both', which=u'both', length=0)
    # Set labels
    ax_labeling(bigax, **kwargs)
    bigax.patch.set_alpha(0.0)

    if "yoffset" in kwargs:
        bigax.yaxis.set_label_coords(kwargs.get("yoffset"), 0.5)
    if "yo" in kwargs:
        bigax.yaxis.set_label_coords(kwargs.get("yo"), 0.5)
    if "toffset" in kwargs:
        if 't' in kwargs:
            title = kwargs.get('t')
        if 'title' in kwargs:
            title = kwargs.get('title')
        bigax.set_title(title, y=kwargs.get("toffset"))
    return bigax


def add_subplot_axis(ax, rect, **kwargs):
    """
    Add an axis object to plot embedded figure
    :param ax: big plot axis object to embed subplots
    :param rect: [x, y, width, height]
    :return: the embedded axis object
    """

    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    width *= rect[2]
    height *= rect[3]

    in_ax_position = ax.transAxes.transform(rect[0:2])
    inv_trans_figure = fig.transFigure.inverted()  # create a transform from display to data coordinates
    in_fig_position = inv_trans_figure.transform(in_ax_position)
    x = in_fig_position[0]
    y = in_fig_position[1]

    subax = fig.add_axes([x, y, width, height], **kwargs)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def add_customized_colorbar(fig, minmax, pos, cmap_name="viridis", ori='h', log=False, **kwargs):
    """
    Add an customized colorbar in current figure
    :param minmax: a list of min/max values
    :param pos: [(x, y) of lower left point, and then (width, height)] of the colorbar
    :param cmap_name: colormap's name, default is viridis
    :param ori: orientation, default is horizontal
    :param log: whether to have a log scaling colorbar
    :param kwargs: other keywords, for example, label=r“$\Sigma$”
    :return: a matplotlib.colorbar.ColorbarBase object to manipulate
             e.g., cbar.set_ticks([0.01, 0.1, 1.0])  # in log scale
                   cbar.set_ticklabels([r"0.01", r"0.1", r"1.0"])
                   cbar.ax.text(1.05, 0.0, r"$\Sigma$", fontsize=16)
    """

    norm = mpl.colors.Normalize(minmax[0], minmax[1])
    if log:
        norm = mpl.colors.LogNorm(10**minmax[0], 10**minmax[1])

    cmap = plt.get_cmap(cmap_name)
    cax = fig.add_axes(pos)
    if ori == 'h':
        ori = 'horizontal'
    if ori == 'v':
        ori = 'vertical'

    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation=ori, extend=u'max', **kwargs)

    return cbar


def add_aligned_colorbar(fig, ax, im, **kwargs):
    """
    Add an aligned colorbar to the current axis
    :param fig: the figure object
    :param ax: axis object
    :param im: image object return by imshow, pcolorfast, etc
    :param kwargs: other keywords, for example, size, pad, etc.
    :return: the colorbar object
    """

    divider = make_axes_locatable(ax)
    pos = kwargs.get("pos", "right")
    # adjust orientation for different locations
    orientation = 'vertical'
    if pos in ['top', 'bottom']:
        orientation = 'horizontal'

    size = kwargs.get("size", "5%")
    pad = kwargs.get("pad", 0.05)
    # REF: https://stackoverflow.com/a/26566276/4009531
    # If user set a customized aspect ratio for im, it is possible that
    # colorbar still uses the original extent and becomes longer/shorter than the axes.
    # One may set aspect here to adjust it
    if "aspect" in kwargs:
        cax = divider.append_axes(pos, size=size, pad=pad, aspect=kwargs["aspect"])
    else:
        cax = divider.append_axes(pos, size=size, pad=pad)
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    # Change tick position to top (with the default tick position "bottom",
    # ticks overlap the image).
    if pos in ['top', 'left']:
        cax.xaxis.set_ticks_position('top')
    return cb


def make_square(ax):
    """ Make axes frame square instead of equal scale """

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))


def get_ax_size_in_pixels(fig, ax):
    """ Get the size of axis frame in pixels """

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # I found there is always 7 pixels overhead than the real picture, weird
    return bbox.width*fig.dpi-7, bbox.height*fig.dpi-7


def exact_size_figure(size, ax_pos=None, dpi=100):
    """
    create a figure with exact pixels in the plotting frame
    :param figsize: figure size in inches
    :param ax_pos: ax position, [x_lower_left, y_lower_left, width, height]
    :param dpi: dots per inch, default 100
    :return: a figure object and an axes object
    PS: using bbox_inches='tight' will change this setup unexpectedly
    """

    # to avoid mutable default arguments, use None and assign values in function
    # a new list is created each time the function is defined, and the same list is used in each successive call,
    # which means a user-defined ax_pos will affect the second-call
    if ax_pos is None:
        ax_pos = [0, 0, 1, 1]
    fig = plt.figure(figsize=size, dpi=dpi)
    ax = plt.Axes(fig, ax_pos)
    fig.add_axes(ax)
    return fig, ax


def draw_grid(origin, ending, Nx, ax=None, figsize=None, lw=None):
    """
    Draw a grid layout for visualizing simulations
    :param origin: the origin coordinates of the simulation domain
    :param ending: the ending coordinates of the simulation domain
    :param Nx: the resolution in grid cells
    :param ax:
    :param figsize: customized figure size
    :return: a Figure object and an Axes object for further plotting
    """

    new_ax_flag = False
    if ax is None:
        if figsize is None:
            figsize = (8, 8 * (ending[1] - origin[1]) / (ending[0] - origin[0]))
        plt_params("medium")
        fig, ax = plt.subplots(figsize=figsize)
        new_ax_flag = True

    if lw is None:
        lw = 0.2

    x = np.linspace(origin[0], ending[0], Nx[0]+1)
    y = np.linspace(origin[1], ending[1], Nx[1]+1)
    x_edge = np.array([origin[0], ending[0]])
    y_edge = np.array([origin[1], ending[1]])

    for item in x:
        ax.add_artist(plt.Line2D([item, item], y_edge, color='grey', lw=lw, alpha=0.5))
    for item in y:
        ax.add_artist(plt.Line2D(x_edge, [item, item], color='grey', lw=lw, alpha=0.5))

    ax.set_xlim([origin[0], ending[0]])
    ax.set_ylim([origin[1], ending[1]])

    if new_ax_flag:
        return fig, ax
