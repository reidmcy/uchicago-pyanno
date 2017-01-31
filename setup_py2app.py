"""pyanno package setup definition"""

from setuptools import setup, find_packages

# ---- add ETS recipes for py2app

import types
import py2app.recipes

def ets_check(cmd, mf):
    m = mf.findNode('pyface')
    if m is None or m.filename is None:
        return None
    return dict(
        packages = ['pyface','enable','kiva','traits','wx','traitsui','chaco']
    )

py2app.recipes.ets = types.ModuleType('py2app.recipes.ets')
py2app.recipes.ets.check = ets_check


# ---- /add ETS recipes for py2app


from setup import setup_dict

APP = ['pyanno/ui/main.py']
OPTIONS = {'argv_emulation': True, 'packages': 'pyanno'}

setup_py2app_dict = dict(
    app=APP,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)

setup_py2app_dict.update(setup_dict)

if __name__ == '__main__':
    setup(**setup_py2app_dict)
