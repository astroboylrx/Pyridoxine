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
