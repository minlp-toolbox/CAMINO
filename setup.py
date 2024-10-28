# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""A repository for MINLP solvers."""

from setuptools import setup, find_packages

setup(
    name="camino",
    version="0.1.1",
    description="Collection of Algorithms for Mixed-Integer Nonlinear Optimization",
    url="https://github.com/minlp-toolbox/CAMINO",
    author="Andrea Ghezzi, Wim Van Roy",
    author_email="andrea.ghezzi@imtek.uni-freiburg.de, wim.vr@hotmail.com",
    license="GPL-3.0",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "numpy",
        "pandas",
        "casadi",
        "scipy",
        "pytz",
        "matplotlib",
        "parameterized",
        "timeout-decorator",
        "tox",
        "colored",
        "seaborn",
        "argcomplete",
    ],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ]
)
