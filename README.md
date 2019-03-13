
# Lectures in Quantitative Economics Test Site

This is a sandpit version of the main RST lecture source repo, which is https://github.com/QuantEcon/lecture-source-py.

For instructions on how to operate it, see that repository.  Operation is
essentially the same.

In short:

1) Download and install [Anacoda](https://www.anaconda.com/distribution/) for your platform .

2) Download or clone this repository.

3) Enter your local copy of the repository and run `make setup`.

To transform the `rst` files in to `ipynb` files, enter the repo and run `make notebooks`.

The resulting `ipynb` files are stored in a temporary `_build` directory at the root level of the repository.

To view the notebooks run `make view`

Additionally you can view a particular lecture directly:

* Example: `make view lecture=about_py`

The [main repo](https://github.com/QuantEcon/lecture-source-py) contains further suggestions on workflow.

