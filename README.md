[![PyPI version](https://badge.fury.io/py/pyridoxine.svg)](https://badge.fury.io/py/pyridoxine)

# Pyridoxine: Handy Python Snippets for Athena Data

This branch contains useful Python snippets for analyzing simulation data produced by code [Athena](https://github.com/PrincetonUniversity/Athena-Cversion).

`Pyridoxine` is able to read `vtk` and `lis` files efficiently and the users can then post-process these data.  [This Jupyter Notebook](https://gist.github.com/astroboylrx/332611f562e4817c011800353ddb5a21) gives a simple demo.

## Installation

You may install `Pyridoxine` by this command:

```bash
pip install -U pyridoxine
```

Or, you may try the most updated `Pyridoxine` by this command:

```bash
pip install -U -e git+git://github.com/astroboylrx/Pyridoxine@Athena#egg=Pyridoxine
```

It will automatically install all the required modules. Note that `#egg=Pyridoxine` is not a comment here. It is to explicitly state the package name.