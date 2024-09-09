# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nullcal'
copyright = '2024, Isaac C.F. Wong'
author = 'Isaac C.F. Wong'
release = 'v0.3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'numpydoc',
    'nbsphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx_tabs.tabs',
    "sphinx.ext.linkcode",
    'myst_parser'
]
autosummary_generate = True


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


def linkcode_resolve(domain, info):
    """
    Adapted from https://github.com/aaugustin/websockets/blob/8e1628a14e0dd2ca98871c7500484b5d42d16b67/docs/conf.py
    """
    if domain != 'py':
        return None
    if not info['module']:
        return None

    try:
        mod = importlib.import_module(info["module"])
        if "." in info["fullname"]:
            objname, attrname = info["fullname"].split(".")
            obj = getattr(mod, objname)
            try:
                # object is a method of a class
                obj = getattr(obj, attrname)
            except AttributeError:
                # object is an attribute of a class
                return None
        else:
            obj = getattr(mod, info["fullname"])

        try:
            file = inspect.getsourcefile(obj)
            lines = inspect.getsourcelines(obj)
        except TypeError:
            # e.g. object is a typing.Union
            return None
        file = f"{project}/{''.join(file.split(f'{project}/')[1:])}"
        start, end = lines[1], lines[1] + len(lines[0]) - 1
    except Exception:
        return

    return f"{GITURL}/-/tree/{GITHASH}/{file}#L{start}-L{end}"
