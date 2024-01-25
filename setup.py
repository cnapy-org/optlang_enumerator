from setuptools import setup

setup(name='optlang_enumerator',
    version='0.0.9',
    description='Enumeration of multiple solutions to a MILP with optlang.',
    url='https://github.com/cnapy-org/optlang_enumerator.git',
    author='Axel von Kamp',
    author_email='axelk1@gmx.de',
    license='Apache License 2.0',
    packages=['optlang_enumerator'],
    install_requires=['numpy<1.24', 'scipy', 'cobra', 'optlang', 'efmtool_link', 'sympy>=1.12', 'swiglpk'],
    zip_safe=False)
