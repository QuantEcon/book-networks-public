Economic Networks Code Book
===========================

**AUTHOR**: [John Stachurski](https://johnstachurski.net/), [Thomas J. Sargent](http://www.tomsargent.com/)


This Jupyter book provides code to accompany the second edition of the
textbook [Economic Networks: Theory and
        Computation](), published by the [MIT
Press](https://mitpress.mit.edu/).  The code recreates figures from the book and gives solutions to exercises.


The only code provided at this stage is Python.  A Julia version is lacking
only due to time constraints.  If there are readers who would like to create a
Julia version, please get in touch via email or the [issue
tracker]().  A MATLAB version is also
welcome --- although I am not personally familiar with the language.

```{note}
To run the code contained here, please install the latest version of
[Anaconda Python](https://www.anaconda.com/).  For a few of the programs you will also
need the [QuantEcon Python library](https://quantecon.org/quantecon-py/) and
the [Interpolation library](https://github.com/EconForge/interpolation.py)
(install as required following the instructions on each library page).

If you encounter errors, please open an
[issue](https://github.com/jstac/edtc-code/issues) and copy and paste your
error message.
```

The Python code contained in these notes is accelerated through a combination of
[NumPy](https://numpy.org/) and just-in-time compilation
(via [Numba](http://numba.pydata.org/)).  For those who require it, QuantEcon
provides a fast-paced [introduction to scientific computing with
Python](https://python-programming.quantecon.org/) with background on
these topics.


This code book is created using [Jupyter Book](https://jupyterbook.org/intro.html).