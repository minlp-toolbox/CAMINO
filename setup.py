# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""A repository for MINLP solvers."""

from setuptools import setup, find_packages

setup(
    name="caminopy",
    version="0.1.3",
    description="Collection of Algorithms for Mixed-Integer Nonlinear Optimization",
    url="https://github.com/minlp-toolbox/CAMINO",
    author="Andrea Ghezzi, Wim Van Roy",
    author_email="andrea.ghezzi@imtek.uni-freiburg.de, wim.vr@hotmail.com",
    license="GPL-3.0",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy >= 1.24.4",
        "pandas >= 2.0.3",
        "casadi >= 3.6.1",
        "scipy >= 1.10.1",
        "pytz >= 2024.2",
        "matplotlib >= 3.7.5",
        "parameterized >= 0.9.0",
        "timeout-decorator >= 0.5.0",
        "tox >= 4.1.2",
        "colored >= 1.4.4",
        "argcomplete >= 3.5.1",
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
