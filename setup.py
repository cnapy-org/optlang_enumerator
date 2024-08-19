from setuptools import setup

setup(name='optlang_enumerator',
    version='0.0.12',
    description='Module for enumerating minimal cut sets',
    url='https://github.com/cnapy-org/optlang_enumerator',
    author='Axel von Kamp',
    author_email='vonkamp@mpi-magdeburg.mpg.de',
    license='Apache License 2.0',
    packages=['optlang_enumerator'],
    package_dir={'optlang_enumerator': 'optlang_enumerator'},
    install_requires=['numpy==1.23', 'scipy', 'cobra>=0.26.3', 'optlang', 'efmtool_link>=0.0.6', 'sympy>=1.12', 'swiglpk'],
    zip_safe=False)
