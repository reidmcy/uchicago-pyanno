"""2to3'd pyanno package setup definition"""


from setuptools import setup, find_packages


with open('README') as f:
    LONG_DESCRIPTION = f.read()


setup_dict = dict(
    name = "pyanno3",
    version = "2.0.2",
    packages = find_packages(),

    description = 'Package for curating data annotation efforts.',
    long_description = LONG_DESCRIPTION,

    url = 'https://github.com/reidmcy/uchicago-pyanno',
    download_url = "https://github.com/reidmcy/uchicago-pyanno/archive/0.1.tar.gz",

    license='BSD',
    platforms = ["Any"],

    package_data = {
      '': ['*.txt', 'README', 'data/*'],
    },
    include_package_data = True,

    install_requires = ['scipy', 'numpy', 'traits'],

    entry_points = {
      'console_scripts': [],
      'gui_scripts': [
          'pyanno-ui = pyanno.ui.main:main',
      ],
      #'setuptools.installation': [
      #  'eggsecutable = pyanno.ui.main:main',
      #]
    },
    #scripts = ['examples/mle_sim.py',
    #           'examples/map_sim.py',
    #           'examples/rzhetsky_2009/mle.py',
    #           'examples/rzhetsky_2009/map.py' ],
)

if __name__ == '__main__':
    setup(**setup_dict)
