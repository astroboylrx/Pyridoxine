""" Provide default setup for matplotlib.pyplot """


def plt_params(size="large"):
    """ Give matplotlib params based on size """

    if size in ["large", "big", "Large", "Big"]:
        return {
            'figure.figsize': (12, 9),
            'savefig.dpi': 300,
            'lines.linewidth': 0.6,
            'axes.labelsize': 20,
            'axes.linewidth': 0.75,
            'axes.titlesize': 20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 18,
            'legend.frameon': True,
            'legend.handlelength': 1.5
            }
    elif size in ["medium", "Medium", "middle", "Middle"]:
        return {
            'figure.figsize': (10, 8),
            'savefig.dpi': 300,
            'lines.linewidth': 0.6,
            'axes.labelsize': 18,
            'axes.linewidth': 0.75,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'axes.titlesize': 18,
            'legend.fontsize': 16,
            'legend.frameon': True,
            'legend.handlelength': 1.5
            }
    else:
        raise ValueError("Wrong size specified: ", size)


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


def cut_space(plt, fig, ax):
    """
    Fine-tuning subplots in a figure to cut space between them
    :param plt: matplotlib.pyplot
    :param fig: Figure object
    :param ax: Axes object
    :return:
    """


    fig.subplots_adjust(hspace=0, wspace=0)

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
    bigax.spines['top'].set_color('none')
    bigax.spines['bottom'].set_color('none')
    bigax.spines['left'].set_color('none')
    bigax.spines['right'].set_color('none')
    bigax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    # Set labels
    ax_labeling(bigax, **kwargs)
    bigax.patch.set_alpha(0.0)


