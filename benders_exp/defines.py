"""Global defines."""

from os import path, makedirs, environ
import casadi as ca

SOURCE_FOLDER = path.dirname(path.abspath(__file__))
_DATA_FOLDER = path.join(SOURCE_FOLDER, "../data")
IMG_DIR = path.join(SOURCE_FOLDER, "../results/figures")
CACHE_FOLDER = path.join(SOURCE_FOLDER, "../data/cache")

if not path.exists(IMG_DIR):
    makedirs(IMG_DIR)

if not path.exists(CACHE_FOLDER):
    makedirs(CACHE_FOLDER)


WITH_JIT = False
WITH_LOGGING = True
WITH_PLOT = False
CASADI_VAR = ca.SX
WITH_DEBUG = environ.get("DEBUG", False)
IPOPT_SETTINGS = {
    # "ipopt.tol": 1e-2,
    # "ipopt.dual_inf_tol": 2,
    # "ipopt.constr_viol_tol": 1e-3,
    # "ipopt.compl_inf_tol": 1e-3,
    # "ipopt.linear_solver": "ma27",
    # "ipopt.max_cpu_time": 3600.0,
    # "ipopt.max_iter": 6000,
    # "ipopt.acceptable_tol": 0.2,
    # "ipopt.acceptable_iter": 8,
    # "ipopt.acceptable_constr_viol_tol": 10.0,
    # "ipopt.acceptable_dual_inf_tol": 10.0,
    # "ipopt.acceptable_compl_inf_tol": 10.0,
    # "ipopt.acceptable_obj_change_tol": 1e-1,
    # "ipopt.mu_strategy": "adaptive",
    # "ipopt.mu_target": 1e-4,
}
GUROBI_SETTINGS = {
    "gurobi.MIPGap": 0.05,
    "gurobi.NumericFocus": 1,
    # "gurobi.PoolSearchMode": 2,  # Default 0
    # "gurobi.PoolSolutions": 100,  # Default 10
    # "gurobi.PoolObjBound" # Discard avoce this value
}
EPS = 1e-5

if not WITH_DEBUG:
    IPOPT_SETTINGS.update({
        "ipopt.print_level": 0,
        "verbose": False,
        "print_time": 0,
    })
    GUROBI_SETTINGS.update({
        "verbose": False,
        "print_time": 0,
        "gurobi.OutputFlag": 0,
    })
