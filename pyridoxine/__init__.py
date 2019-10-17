""" Pyridoxine: My Personal Handy Python Snippets.

    This package is created to collect code snippets used in my work.
    They should be simple and useful, like pyridoxine to human beings.

    Notes to myself: 
        1, Subpackages put in "__all__" needs to be explicitly imported, for
        example, you cannot directly use "mpl.pyplot" after import matplotlib.
        You need to explicitly "import matplotlib.pyplot as plt". <- (Not true in IPython)
        This behavior makes "import pyridoxine" more efficient and also
        avoid possible side effects of importing subpackages.
        But "from pyridoxine import *" will import the entire "__all__".
        2, Using "from . import subpackage" will need users to type
        "pyridoxine.subpackage.spam()" to call spam() from subpackage.
        3, Using "from .subpackage import *" will only need users to type
        "pyridoxine.spam()" to call spam() from subpackage.
        4, Wildcard imports (from <module> import *) should be avoided, 
        as they make it unclear which names are present in the namespace, 
        confusing both readers and many automated tools. 
        There is one defensible use case for a wildcard import, 
        which is to republish an internal interface as part of a public API.
        NumPy uses this defensible case a lot.
"""

# Module level "dunders" (i.e. names with two leading and two trailing 
# underscores) such as __all__, __author__, __version__, etc. should be
# placed after the module docstring but before any import statements
# except from __future__ imports.
__version__ = "0.2.4"
__author__ = "Rixin Li"
__all__ = ["help_info", "plt", "athena", "utility"]


def help_info():
    """ Print Basic Help Info """

    print("""
    **********************************************************************
    * Pyridoxine: Handy Python Snippets for Athena Data
    * 
    * This package contains useful Python snippets for analyzing
    * simulation data produced by code Athena. 
    * I hope they are simple and useful, like pyridoxine to human beings.
    * 
    * Author: Rixin Li
    * Current Version: 0.2.4
    * Note: This module is very native and under development.
    **********************************************************************
    """)
