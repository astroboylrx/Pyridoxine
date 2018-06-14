""" Provide default setup for matplotlib.pyplot """

from astropy.visualization import astropy_mpl_style
import matplotlib as mpl
import matplotlib.pyplot as plt


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
            'legend.handlelength': 1.5
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
            'legend.handlelength': 1.5
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
            'legend.handlelength': 1.5
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
            'legend.handlelength': 1.5
            }
    else:
        raise ValueError("Wrong size specified: ", size)

    plt.rcParams.update(new_params)


def turn_off_minor_labels(ax):
    """ Turn off the tick labels for minor ticks """

    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


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

    if len(kwargs) is 0:
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


def cut_space(fig, ax, space=None):
    """
    Fine-tuning subplots in a figure to cut space between them
    :param fig: Figure object
    :param ax: Axes object
    :param space: two element array specifying hspace and wspace
    :return:
    """

    if space is None:
        space = [0, 0]

    fig.subplots_adjust(hspace=space[0], wspace=space[1])

    # hide x ticks for top plots; hide y ticks for right plots
    plt.setp([a.get_xticklabels() for a in ax[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in ax[:, 1]], visible=False)


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
