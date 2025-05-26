# CAMINO: Collection of Algorithms for Mixed-Integer Nonlinear Optimization
This software package provides a Python/CasADi-based implementation of several algorithms for solving mixed-integer nonlinear programs (MINLPs).

## Installation

Set up and activate a fresh Python virtual environment (**required** Python >= 3.8)

```
python -m venv env
source env/bin/activate
```

### Option 1: from PyPi
```
pip install caminopy
python -m camino run <solver> <problem>
```

**Example**: running outer approximation solver (`oa`) on the example `dummy`
```
python -m camino run oa dummy
```

More info by running
```
python -m camino -h
```

### Option 2: from GitHub
Clone the repository and open the directory
```
git clone git@github.com:minlp-toolbox/CAMINO.git
cd CAMINO
```
From the root of the repository
```
pip install .
python camino run <solver> <problem>
```

More info by running
```
python camino -h
```

### Optional: Install pycombina

To use the solver 'cia', the packages relies on pycombina. If you want to use this solver, install pycombina using:

- Install gcc
- Set up and activate a fresh Python virtual environment (Python >= 3.7 should work)
- If you want to use pycombina for comparison, install the dependencies listed at https://pycombina.readthedocs.io/en/latest/install.html#install-on-ubuntu-18-04-debian-10, then clone and build pycombina by running:

```
        git clone https://github.com/adbuerger/pycombina.git
        cd pycombina
        git submodule init
        git submodule update
        python setup.py install
```

## Usage
### Command line

There are some predefined problems inside the library that you can test using the terminal using:
```
python -m camino run <solver> <problem>
```
You can provide any of the solvers together with a problem such as "dummy", "to_car" or "particle".

### Batch runner

A second commandline option to use the library is by running a large set of external nl-files.
```
python -m camino batch <solver> <output-folder> <nl-files>
```
In this case, all problems provided in the commandline will be started one by one and the output of the run is placed in the output folder. In case you stop this process before the end of the run, you can always restart the progress from where the last solved problem  by using the same command.

### Inside your python code

In the folder `docs/` we provide two python scripts `example.py` and `stats_analysis.py`.
- `example.py` shows how to a user can define its own MINLP and call one of the algorithm implemented in this library to solve it.
- `stats_analysis.py` shows how one can retrieve the statistics stored by running the algorithms. More advanced statistics analysis is left to the user.\
  **Note that:** to save stats set the env variable `LOG_DATA=1` by runnning `export LOG_DATA=1` from a terminal console.

## Options

### Setting environment variables

You can enable or change options using environment variables ([default]):
| Environment variable |     Value    | Description                 |
| ---------------------- | ------------ | ----------------------------|
|         DEBUG          |  True/[False]  | Toggle debugging output     |
|         LOG_DATA       |  True/[False]  | Toggle saving statistics    |
|        MIP_SOLVER      | [gurobi]/highs/cbc | Configure MIP solver        |

**Example:** To enable DEBUG mode type in your terminal
```
export DEBUG=True
```

### Available MINLP solvers/algorithms

**New?**: the algorithm is novel and created by the authors of this software.\
**CVX guarantee?**: the algorithm converges to the global optimum when a *convex* MINLP is given.

| Solvers | Description                                                  | New?                                              | CVX guarantee? |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
| gbd        | Generalized Benders Decomposition ([Geoffrion, 1972](https://www.researchgate.net/profile/Arthur-Geoffrion/publication/230872895_Generalized_Benders_Decomposition/links/554a25f20cf29752ee7b8013/Generalized-Benders-Decomposition.pdf)) |                    | x |
| gbd-qp      | Adaptation of the Generalized Benders Decomposition with an hessian to the cost function | x |  |
| oa | Outer approximation ([Fletcher, Leyffer, 1994](http://dx.doi.org/10.1007/BF01581153)) |  | x |
| oa-qp | Quadratic outer approximation ([Fletcher, Leyffer, 1994](http://dx.doi.org/10.1007/BF01581153)) |  |  |
| oa-i | Outer approximation improved with safeguard for nonlinear constraints| x | x |
| oa-qp-i | Quadratic outer approximation improved with safeguard for nonlinear constraints| x |  |
| s-b-miqp | Sequential Benders-based MIQP ([Ghezzi, Van Roy, et al, 2024](https://arxiv.org/pdf/2404.11786)) | x | x |
| s-b-miqp-early-exit | S-B-MIQP with heuristic for early termination ([Ghezzi, Van Roy, et al, 2024](https://arxiv.org/pdf/2404.11786)) | x |  |
| s-v-miqp | Sequential Voronoi-based MIQP with exact or Gauss-Newton Hessian ([Ghezzi et al, 2023](https://publications.syscop.de/Ghezzi2023a.pdf)) |  |  |
| s-tr-milp | Sequential MILP trust region approach ([De Marchi, 2023](https://doi.org/10.48550/arXiv.2310.17285)) *Accept only linear constraints!*| |
| fp | Feasibility Pump for MINLP ([Bertacco, et al, 2007](https://doi.org/10.1016/j.disopt.2006.10.001)) |  |  |
| ofp | Objective Feasibility Pump for MINLP ([Sharma, et al, 2016](https://doi.org/10.1007/s10589-015-9792-y)) |  |  |
| rofp | Random Objective Feasibility Pump | x |  |
| bonmin | ([Bonami, et al, 2006](https://doi.org/10.1016/j.disopt.2006.10.011)) -- Same as bonmin-bb |  | x |
| bonmin-bb | A nonlinear branch-and-bound algorithm based on solving a continuous  nonlinear  program  at  each  node  of  the  search  tree  and  branching on variables  ([Gupta, Ravindran, 1980](https://www.coin-or.org/Bonmin/bib.html#Gupta80Nonlinear)) |  | x |
| bonmin-hyb | A  hybrid  outer-approximation  /  nonlinear  programming  based     branch-and-cut algorithm  ([Bonami et al. 2008](http://domino.research.ibm.com/library/cyberdig.nsf/1e4115aea78b6e7c85256b360066f0d4/fdb4630e33bd2876852570b20062af37?OpenDocument)) |  | x |
| bonmin-oa | An  outer-approximation  based  decomposition  algorithm  ([Duran, Grossmann, 1986](https://www.coin-or.org/Bonmin/bib.html#DG)), ([Fletcher, Leyffer, 1994](http://dx.doi.org/10.1007/BF01581153)) |  | x |
| bonmin-qg | An outer-approximation based branch-and-cut algorithm  ([Quesada, Grossmann, 1994](http://dx.doi.org/10.1016/0098-1354(92)80028-8)) |  | x |
| bonmin-ifp | An iterated feasibility pump algorithm   ([Bonami, et al, 2009](http://dx.doi.org/10.1007/s10107-008-0212-2)) |  |  |
| cia | Combinatorial Integral Approximation ([Sager, et al, 2011](https://link.springer.com/article/10.1007/s00186-011-0355-4)) using `pycombina` ([Buerger, et al, 2020](https://publications.syscop.de/Buerger2020a.pdf)) -- installation instructions below|  |  |
| nlp | Solve the canonical relaxation of the MINLP (integers are relaxed to continuous variables) |  |  |
| nlp-fxd | Fix the integer variables of the MINLP and solve the corresponding NLP|  |  |
| mip | Solve the given MILP/MIQP |  | x |

### Warmstart

It is possible to **warm start** every solver with the solution of another one by concatenating with a `+` the desired solvers when executing `python3 camino` or `python3 -m camino`.
For instance, to use the solution of the feasibility pump as a warm start to sequential Benders-based MIQP, execute the following:

```
python camino run fp+s-b-miqp <problem>
```

## Issues with CasADi function eval
Some examples we provided make use of CasADi subroutines (e.g., `bspline`) which only accept `casadi MX` symbolics. If you experience the following error (or similar)
```
.../casadi/core/function_internal.cpp:2013: 'eval_sx' not defined for BSplineInterpolant
```
Please go to `settings.py` and update the default value for `_CASADI_VAR` to `ca.MX`.

## Citing

If you find this project useful, please consider giving it a :star: or citing it if your work is scientific:
```bibtex
@software{camino,
  author = {Ghezzi, Andrea and Van Roy, Wim},
  license = {GPL-3.0},
  month = oct,
  title = {CAMINO: Collection of Algorithms for Mixed-Integer Nonlinear Optimization},
  url = {https://github.com/minlp-toolbox/CAMINO},
  version = {0.1.1},
  year = {2024}
}
```

## Contributing
Contributions and feedback are welcomed via GitHub PR and issues!

## License
This software is under GPL-3.0 license, please check [LICENSE](https://github.com/minlp-toolbox/CAMINO/blob/main/LICENSE) for more details.
