try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from dataman.__version__ import __version__

setup(name='dataman',
      description='Data Manager',
      author='Ronny Eichler',
      author_email='ronny.eichler@gmail.com',
      version=__version__,
      install_requires=['nose', 'termcolor', 'vispy', 'numpy', 'tqdm', 'scipy', 'matplotlib', 'h5py', 'hdf5storage',
                        'scikit-learn', 'datashader', 'pandas', 'pyyaml'],
      packages=['dataman'],
      entry_points="""[console_scripts]
            dm=dataman.dm:main""")
