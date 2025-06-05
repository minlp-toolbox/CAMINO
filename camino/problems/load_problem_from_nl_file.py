# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Load a MINLP from a nl file."""

from camino.settings import Settings, GlobalSettings
from camino.problems import MinlpProblem, MinlpData, \
    MetaDataOcp
import casadi as ca
import numpy as np


def create_from_nl_file(file, compiled=True):
    """Load from NL file."""
    from camino.utils.cache import CachedFunction, return_func
    import hashlib
    # Create an NLP instance
    nl = ca.NlpBuilder()

    # Parse an NL-file
    nl.import_nl(file, {"verbose": False})
    print(f"Loading MINLP with: {nl.repr()}")

    if not isinstance(nl.x[0], GlobalSettings.CASADI_VAR):
        raise Exception(
            f"Set GlobalSettings.CASADI_VAR to {type(nl.x[0])} in defines!")

    idx = np.where(np.array(nl.discrete))

    if compiled:
        key = str(hashlib.md5(file.encode()).hexdigest())[:64]
        x = ca.vcat(nl.x)
        problem = MinlpProblem(
            x=x,
            f=CachedFunction(f"f_{key}", return_func(
                ca.Function("f", [x], [nl.f])))(x),
            g=CachedFunction(f"g_{key}", return_func(
                ca.Function("g", [x], [ca.vcat(nl.g)])))(x),
            idx_x_integer=idx[0].tolist(),
            p=[]
        )
    else:
        problem = MinlpProblem(
            x=ca.vcat(nl.x),
            f=nl.f, g=ca.vcat(nl.g),
            idx_x_integer=idx[0].tolist(),
            p=[]
        )
    if nl.f.is_constant():
        raise Exception("No objective!")

    problem.hessian_not_psd = True
    data = MinlpData(x0=np.array(nl.x_init),
                     _lbx=np.array(nl.x_lb),
                     _ubx=np.array(nl.x_ub),
                     _lbg=np.array(nl.g_lb),
                     _ubg=np.array(nl.g_ub), p=[])

    from camino.solvers import inspect_problem, set_constraint_types
    set_constraint_types(problem, *inspect_problem(problem, data))
    s = Settings()

    s.OBJECTIVE_TOL = 1e-8
    s.CONSTRAINT_TOL = 1e-8
    s.CONSTRAINT_INT_TOL = 1e-3
    s.MINLP_TOLERANCE = 1e-2
    s.MINLP_TOLERANCE_ABS = 1e-2
    s.BRMIQP_GAP = 0.1
    s.LBMILP_GAP = 0.01
    s.TIME_LIMIT = 300
    s.TIME_LIMIT_SOLVER_ONLY = False
    s.USE_RELAXED_AS_WARMSTART = True
    s.IPOPT_SETTINGS = {
        "ipopt.linear_solver": "ma27",
        "ipopt.max_cpu_time": s.TIME_LIMIT / 4,
        "ipopt.max_iter": 1000,
        "ipopt.constr_viol_tol": s.CONSTRAINT_TOL,
        # "ipopt.mu_strategy": "adaptive",
        # "ipopt.mu_oracle": "probing",
        # "ipopt.bound_relax_factor": 0,
        # "ipopt.honor_original_bounds": "yes",
        "ipopt.print_level": 0,
        # # Options used within Bonmin
        # "ipopt.gamma_phi": 1e-8,
        # "ipopt.gamma_theta": 1e-4,
        # "ipopt.required_infeasibility_reduction": 0.1,
        # "ipopt.expect_infeasible_problem": "yes",
        # "ipopt.warm_start_init_point": "yes",
    }
    s.MIP_SETTINGS_ALL["gurobi"] = {
        "gurobi.MIPGap": 0.1,
        "gurobi.FeasibilityTol": s.CONSTRAINT_INT_TOL,
        "gurobi.IntFeasTol": s.CONSTRAINT_INT_TOL,
        "gurobi.Heuristics": 0.05,
        "gurobi.PoolSearchMode": 0,
        "gurobi.PoolSolutions": 5,
        "gurobi.Threads": 1,
        "gurobi.TimeLimit": s.TIME_LIMIT / 2,
        "gurobi.output_flag": 0,
    }
    s.BONMIN_SETTINGS = {
        "bonmin.time_limit": s.TIME_LIMIT,
        "bonmin.tree_search_strategy": "dive",
        "bonmin.node_comparison": "best-bound",
        "bonmin.allowable_fraction_gap": Settings.MINLP_TOLERANCE,
        "bonmin.allowable_gap": Settings.MINLP_TOLERANCE_ABS,
        "bonmin.constr_viol_tol":  s.CONSTRAINT_TOL,
        "bonmin.linear_solver": "ma27",
        "bonmin.bound_relax_factor": 1e-14,
        "bonmin.honor_original_bounds": "yes",
    }
    s.WITH_DEBUG = False

    return problem, data, s
