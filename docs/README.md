The tutorials are included as Jupyter Notebooks and, due to some apparent errors
with versions and packages, requires a special setup. Follow this procedure:

1. Install `pandoc` using brew (if on MacOS) or via website instructions (if not).
2. Setup a virtual environment in the root directory with `virtualenv vent`
3. Activate the environment and install all project and docs dependencies
```
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-docs.txt
```
4. You should now be able to build the docs using `make html`