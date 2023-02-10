"""Quick and dirty implementation."""

from os import path
import matplotlib.pyplot as plt
from sys import argv
from copy import deepcopy
from dataclasses import dataclass
import datetime as dt
from typing import List, Dict, Any, Optional
import numpy as np

from benders_exp.nlpsolver import NLPSolverRel  # NLPSolverBin
from benders_exp.ambient import Ambient
from benders_exp.defines import _PATH_TO_NLP_OBJECT, _NLP_OBJECT_FILENAME
from benders_exp.nlpsetup import NLPSetupMPC
from benders_exp.predictor import Predictor
from benders_exp.simulator import Simulator
from benders_exp.state import State
from benders_exp.timing import TimingMPC
import casadi as ca
from benders_exp.utils import tic, toc  # , DebugCallBack


WITH_JIT = False
WITH_LOGGING = True
WITH_PLOT = False
CASADI_VAR = ca.MX
IPOPT_SETTINGS = {
    # "ipopt.tol": 1e-2,
    # "ipopt.dual_inf_tol": 2,
    # "ipopt.constr_viol_tol": 1e-3,
    # "ipopt.compl_inf_tol": 1e-3,
    "ipopt.linear_solver": "ma27",
    "ipopt.max_cpu_time": 3600.0,
    "ipopt.max_iter": 6000,
    "ipopt.acceptable_tol": 0.2,
    "ipopt.acceptable_iter": 8,
    "ipopt.acceptable_constr_viol_tol": 10.0,
    "ipopt.acceptable_dual_inf_tol": 10.0,
    "ipopt.acceptable_compl_inf_tol": 10.0,
    "ipopt.acceptable_obj_change_tol": 1e-1,
    "ipopt.mu_strategy": "adaptive",
    "ipopt.mu_target": 1e-4,
}


if WITH_PLOT:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


@dataclass
class Stats:
    """Collect stats."""

    data: Dict[str, float]

    def __getitem__(self, key):
        """Get attribute."""
        if key not in self.data:
            return 0
        return self.data[key]

    def __setitem__(self, key, value):
        """Set item."""
        self.data[key] = value

    def print(self):
        """Print statistics."""
        print("Statistics")
        for k, v in data:
            print(f"\t{k}: {v}")


@dataclass
class MinlpProblem:
    """Minlp problem description."""

    f: CASADI_VAR
    g: CASADI_VAR
    x: CASADI_VAR
    p: CASADI_VAR
    idx_x_bin: List[float]


def visualize_cut(g_k, x_bin, nu):
    """Visualize cut."""
    xx, yy = np.meshgrid(range(10), range(10))
    cut = ca.Function("t", [x_bin, nu], [g_k])
    z = np.zeros(xx.shape)
    for i in range(10):
        for j in range(10):
            z[i, j] = cut(ca.vertcat(xx[i, j], yy[i, j]), 0).full()[0, 0]

    ax.plot_surface(xx, yy, z, alpha=0.2)
    plt.show(block=False)
    plt.pause(1)


@dataclass
class MinlpData:
    """Nlp data."""

    p: List[float]
    x0: ca.DM
    _lbx: ca.DM
    _ubx: ca.DM
    _lbg: ca.DM
    _ubg: ca.DM
    solved: bool
    prev_solution: Optional[Dict[str, Any]] = None

    @property
    def _sol(self):
        """Get safely previous solution."""
        if self.prev_solution is not None:
            return self.prev_solution
        else:
            return {"f": -ca.inf, "x": ca.DM(self.x0)}

    @property
    def obj_val(self):
        """Get float value."""
        return float(self._sol['f'])

    @property
    def x_sol(self):
        """Get x solution."""
        return self._sol['x']

    @property
    def lam_g_sol(self):
        """Get lambda g solution."""
        return self._sol['lam_g']

    @property
    def lam_x_sol(self):
        """Get lambda g solution."""
        return self._sol['lam_x']

    @property
    def lbx(self):
        """Get lbx."""
        return deepcopy(self._lbx)

    @property
    def ubx(self):
        """Get ubx."""
        return deepcopy(self._ubx)

    @property
    def lbg(self):
        """Get lbx."""
        return deepcopy(self._lbg)

    @property
    def ubg(self):
        """Get ubx."""
        return deepcopy(self._ubg)


def to_0d(array):
    """To zero dimensions."""
    if isinstance(array, np.ndarray):
        return array.squeeze()
    else:
        return array.full().squeeze()


def extract():
    """Extract original problem."""
    startup_time = dt.datetime.fromisoformat("2010-08-19 06:00:00+02:00")
    timing = TimingMPC(startup_time=startup_time)

    ambient = Ambient(timing=timing)
    ambient.update()

    state = State()
    state.initialize()

    simulator = Simulator(timing=timing, ambient=ambient, state=state)
    simulator.solve()

    predictor = Predictor(
        timing=timing,
        ambient=ambient,
        state=state,
        previous_solver=simulator,
        solver_name="predictor",
    )
    predictor.solve(n_steps=0)

    # simulator.b_data
    nlpsolver_rel = NLPSolverRel(
        timing=timing,
        ambient=ambient,
        previous_solver=simulator,
        predictor=predictor,
        solver_name="nlpsolver_rel",
    )

    nlpsetup_mpc = NLPSetupMPC(timing=timing)
    nlpsetup_mpc._setup_nlp(True)

    binary_values = []
    binary_values.extend(nlpsetup_mpc.idx_b)
    # binary_values.extend(nlpsetup_mpc.idx_sb)
    # binary_values.extend(nlpsetup_mpc.idx_sb)
    # binary_values.extend(nlpsetup_mpc.idx_sb_red)

    nlpsolver_rel._store_previous_binary_solution()
    nlpsolver_rel._setup_nlpsolver()
    nlpsolver_rel._set_states_bounds()
    nlpsolver_rel._set_continuous_control_bounds()
    nlpsolver_rel._set_binary_control_bounds()
    nlpsolver_rel._set_nlpsolver_bounds_and_initials()

    nlp_args = nlpsolver_rel._nlpsolver_args
    problem = MinlpProblem(**nlpsetup_mpc.nlp, idx_x_bin=binary_values)
    data = MinlpData(x0=nlp_args['x0'],
                     _lbx=nlp_args['lbx'],
                     _ubx=nlp_args['ubx'],
                     _lbg=nlp_args['lbg'],
                     _ubg=nlp_args['ubg'],
                     p=nlp_args['p'], solved=True)

    return problem, data


def create_dummy_problem(p_val=[1000, 3]):
    """Create a dummy problem."""
    x = CASADI_VAR.sym("x", 3)
    x0 = np.array([0, 4, 100])
    idx_x_bin = [0, 1]
    p = CASADI_VAR.sym("p", 2)
    f = (x[0] - 4.1)**2 + (x[1] - 4.0)**2 + x[2] * p[0]
    g = ca.vertcat(
        -x[2],
        (x[0]**2 + x[1]**2 - x[2] - p[1]**2)
    )
    lbg = -np.array([ca.inf, ca.inf])
    ubg = np.array([0, 0])
    lbx = -1e3 * np.ones((3,))
    ubx = np.array([ca.inf, ca.inf, ca.inf])

    problem = MinlpProblem(x=x, f=f, g=g, p=p, idx_x_bin=idx_x_bin)
    data = MinlpData(x0=x0, _ubx=ubx, _lbx=lbx,
                     _ubg=ubg, _lbg=lbg, p=p_val, solved=True)
    return problem, data


def create_dummy_problem_2():
    """Create a dummy problem."""
    x = CASADI_VAR.sym("x", 2)
    x0 = np.array([0, 4])
    idx_x_bin = [0]
    p = CASADI_VAR.sym("p", 1)
    f = x[0]**2 + x[1]
    g = ca.vertcat(
        x[1],
        -(x[0]**2 + x[1] - p[0]**2)
    )
    ubg = np.array([ca.inf, ca.inf])
    lbg = np.array([0, 0])
    lbx = -1e3 * np.ones((2,))
    ubx = np.array([ca.inf, ca.inf])

    problem = MinlpProblem(x=x, f=f, g=g, p=p, idx_x_bin=idx_x_bin)
    data = MinlpData(x0=x0, _ubx=ubx, _lbx=lbx,
                     _ubg=ubg, _lbg=lbg, p=[3], solved=True)
    return problem, data


def make_bounded(problem: MinlpProblem, new_inf=1e5):
    """Make bounded."""
    problem.lbx[problem.lbx < -new_inf] = -new_inf
    problem.ubx[problem.ubx > new_inf] = new_inf
    problem.lbg[problem.lbg < -1e9] = -1e9
    problem.ubg[problem.ubg > 1e9] = 1e9


class SolverClass:
    """Create solver class."""

    def __init___(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create a solver class."""
        self.stats = stats
        self.solver = None

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve the problem."""

    def collect_stats(self):
        """Collect statistics."""
        stats = self.solver.stats()
        return stats["success"], stats


class NlpSolver(SolverClass):
    """
    Create NLP solver.

    This solver solves the NLP problem. This is either relaxed or
    the binaries are fixed.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None, is_orig=False):
        """Create NLP problem."""
        super(NlpSolver, self).__init___(problem, stats)
        if options is None:
            if WITH_LOGGING:
                options = {}
            else:
                options = {"ipopt.print_level": 0,
                           "verbose": False, "print_time": 0}

        self.idx_x_bin = problem.idx_x_bin
        options.update(IPOPT_SETTINGS)
        # self.callback = DebugCallBack(
        #     'mycallback', problem.x.shape[0],
        #     problem.g.shape[0], problem.p.shape[0]
        # )
        # self.callback.add_to_solver_opts(options, 50)

        if is_orig:
            # TODO: Clutter!
            path_to_nlp_object = path.join(
                _PATH_TO_NLP_OBJECT, _NLP_OBJECT_FILENAME
            )

            self.solver = ca.nlpsol(
                "nlp", "ipopt", path_to_nlp_object, options
            )
        else:
            options.update({"jit": WITH_JIT})
            self.solver = ca.nlpsol("nlpsol", "ipopt", {
                "f": problem.f, "g": problem.g, "x": problem.x, "p": problem.p
            }, options)

    def solve(self, nlpdata: MinlpData, set_x_bin=False) -> MinlpData:
        """Solve NLP."""
        lbx = nlpdata.lbx
        ubx = nlpdata.ubx
        if set_x_bin:
            lbx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])
            ubx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])

        new_sol = self.solver(
            p=nlpdata.p, x0=nlpdata.x0,  # _sol[:nlpdata.x0.shape[0]],
            lbx=lbx, ubx=ubx,
            lbg=nlpdata.lbg, ubg=nlpdata.ubg
        )
        # self.callback.save(new_sol["x"])

        nlpdata.solved, stats = self.collect_stats()
        self.stats["nlp.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["nlp.iter"] += stats["iter_count"]
        if not nlpdata.solved:
            print("NLP not solved")
        else:
            nlpdata.prev_solution = new_sol
        return nlpdata


class BendersMasterMILP(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create benders master MILP."""
        super(BendersMasterMILP, self).__init___(problem, stats)
        if options is None:
            if WITH_LOGGING:
                options = {}
            else:
                options = {"verbose": False,
                           "print_time": 0, "gurobi.output_flag": 0}

        self.grad_f_x_bin = ca.Function(
            "gradient_f_x_bin",
            [problem.x, problem.p], [ca.gradient(
                problem.f, problem.x
            )[problem.idx_x_bin]],
            {"jit": WITH_JIT}
        )
        self.f = ca.Function(
            "f", [problem.x, problem.p], [problem.f],
            {"jit": WITH_JIT}
        )
        self.g = ca.Function(
            "g", [problem.x, problem.p], [problem.g],
            {"jit": WITH_JIT}
        )
        self.jac_g_bin = ca.Function(
            "jac_g_bin", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)[:, problem.idx_x_bin]],
            {"jit": WITH_JIT}
        )
        self.idx_x_bin = problem.idx_x_bin
        self.nr_x_bin = len(problem.idx_x_bin)
        self._x_bin = CASADI_VAR.sym("x_bin", self.nr_x_bin)
        self._nu = CASADI_VAR.sym("nu", 1)
        self._g = np.array([])
        self.nr_g = 0
        self.options = options.copy()
        self.options["discrete"] = [1] * (self.nr_x_bin + 1)
        self.options["discrete"][-1] = 0
        self.options["gurobi.MIPGap"] = 0.05
        self.g_shape_orig = problem.g.shape[0]
        self.x_shape = max(problem.x.shape)

    def _generate_cut_equation(self, nlpdata, prev_feasible):
        """Generate a cut."""
        if prev_feasible:
            grad_f_k = self.grad_f_x_bin(
                nlpdata.x_sol[:self.x_shape], nlpdata.p)
            jac_g_k = self.jac_g_bin(nlpdata.x_sol[:self.x_shape], nlpdata.p)
            lambda_k = grad_f_k - jac_g_k.T @ -nlpdata.lam_g_sol
            f_k = self.f(nlpdata.x_sol[:self.x_shape], nlpdata.p)
            g_k = (
                f_k + lambda_k.T @ (self._x_bin -
                                    nlpdata.x_sol[self.idx_x_bin])
                - self._nu
            )
        else:  # Not feasible solution
            h_k = self.g(nlpdata.x_sol[:self.x_shape], nlpdata.p)
            jac_h_k = self.jac_g_bin(nlpdata.x_sol[:self.x_shape], nlpdata.p)
            lam_g = nlpdata.lam_g_sol[:self.g_shape_orig] - \
                nlpdata.lam_g_sol[self.g_shape_orig:]
            g_k = lam_g.T @ (h_k + jac_h_k @ (self._x_bin -
                             nlpdata.x_sol[self.idx_x_bin]))

        return g_k

    def solve(self, nlpdata: MinlpData, prev_feasible=True) -> MinlpData:
        """solve."""
        g_k = self._generate_cut_equation(nlpdata, prev_feasible)

        if WITH_PLOT:
            visualize_cut(g_k, self._x_bin, self._nu)

        self._g = ca.vertcat(self._g, g_k)
        self.nr_g += 1

        self.solver = ca.qpsol(f"benders{self.nr_g}", "gurobi", {
            "f": self._nu, "g": self._g,
            "x": ca.vertcat(self._x_bin, self._nu),
        }, self.options)

        solution = self.solver(
            x0=ca.vertcat(nlpdata.x_sol[self.idx_x_bin], nlpdata.obj_val),
            lbx=ca.vertcat(nlpdata.lbx[self.idx_x_bin], -1e5),
            ubx=ca.vertcat(nlpdata.ubx[self.idx_x_bin], ca.inf),
            lbg=-ca.inf * np.ones(self.nr_g),
            ubg=np.zeros(self.nr_g)
        )
        x_full = nlpdata.x_sol.full()[:self.x_shape]
        x_full[self.idx_x_bin] = solution['x'][:-1]
        solution['x'] = x_full
        nlpdata.prev_solution = solution
        nlpdata.solved = self.collect_stats()[0]
        self.stats["milp_benders.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["milp_benders.iter"] += stats["iter_count"]
        return nlpdata


class BendersConstraintMILP(BendersMasterMILP):
    """
    Create benders constraint MILP.

    By an idea of Moritz D. and Andrea R.
    Given the ordered sequence of integer solutions:
        Y := {y1, y2, ..., yN}
    such that J(y1) >= J(y2) >= ... >= J(yN) we define the
    benders polyhedral B := {y in R^n_y:
        J(y_i) + Nabla J(yi) T (y - yi) <= J(y_N),
        forall i = 1,...,N-1
    }

    This milp solves:
        min F(y,z| y_bar, z_bar)
        s.t ub >= H_L(y,z| y_bar, z_bar) >= lb
        with y in B

    For this implementation, since the original formulation implements:
        J(y_i) + Nabla J(yi) T (y - yi) <= nu,
        meaning: nu == J(y_N)
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create the benders constraint MILP."""
        super(BendersConstraintMILP, self).__init___(problem, stats, options)
        self.jac_g = ca.Function(
            "jac_g", [problem.x, problem.p],
            [ca.jacobian(problem.g, problem.x)],
            {"jit": WITH_JIT}
        )
        self.y_N_val = ca.inf

    def solve(self, nlpdata: MinlpData, prev_feasible=True, integer=False) -> MinlpData:
        """Solve."""
        # TODO: Fix shape!
        # J(y_i) + Nablda J(yi)
        g_k = self._generate_cut_equation(nlpdata, prev_feasible)
        self._g = ca.vertcat(self._g, g_k)
        self.nr_g += 1
        if integer:
            self.y_N_val = min(self.y_N_val, nlpdata.obj_val)

        f_lin = self.grad_f_x(nlpdata.x_sol[:self.x_shape], nlpdata.p)
        g_lin = self.g(nlpdata.x_sol[:self.x_shape], nlpdata.p)
        jac_g = self.jac_g(nlpdata.x_sol[:self.x_shape], nlpdata.p)
        self.solver = ca.qpsol(f"benders_constraint{self.nr_g}", "gurobi", {
            "f": f_lin.T * self.x,
            "g": ca.vertcat(
                g_lin + jac_g @ self.x,
                g_k
            ),
            "x": self.x, "p": self._nu
        })
        nlpdata.prev_solution = self.solver(
            x0=nlpdata.obj_val,
            lbx=nlpdata.lbx, ubx=nlpdata.ubx,
            lbg=ca.vertcat(nlpdata.lbg, -ca.inf * np.ones(self.nr_g)),
            ubg=ca.vertcat(nlpdata.ubg, np.zeros(self.nr_g)),
            p=self.y_N_val
        )

        nlpdata.solved = self.collect_stats()[0]
        self.stats["milp_bconstraint.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["milp_bconstraint.iter"] += stats["iter_count"]
        return nlpdata


class FeasibilityNLP(SolverClass):
    """Create benders master problem."""

    def __init__(self, problem: MinlpProblem, stats: Stats, options=None):
        """Create benders master MILP."""
        super(FeasibilityNLP, self).__init___(problem, stats)
        if options is None:
            if WITH_LOGGING:
                options = {}
            else:
                options = {
                    "ipopt.print_level": 0, "verbose": False, "print_time": 0
                }

        self.nr_g = problem.g.shape[0]
        s_lbg = CASADI_VAR.sym("s_lbg", self.nr_g)
        lbg = CASADI_VAR.sym("lbg", self.nr_g)
        ubg = CASADI_VAR.sym("ubg", self.nr_g)

        g = ca.vertcat(
            problem.g - lbg + s_lbg,
            ubg + s_lbg - problem.g
        )
        self.lbg = np.zeros((self.nr_g * 2, 1))
        self.ubg = ca.inf * np.ones((self.nr_g * 2, 1))
        f = ca.sum1(s_lbg)
        x = ca.vertcat(problem.x, s_lbg)
        p = ca.vertcat(problem.p, lbg, ubg)

        self.idx_x_bin = problem.idx_x_bin
        options.update({"jit": WITH_JIT})
        options.update(IPOPT_SETTINGS)
        self.solver = ca.nlpsol("nlpsol", "ipopt", {
            "f": f, "g": g, "x": x, "p": p
        }, options)

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """solve."""
        print("FEASIBILITY")
        lbx = deepcopy(nlpdata.lbx)
        ubx = deepcopy(nlpdata.ubx)
        lbx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])
        ubx[self.idx_x_bin] = to_0d(nlpdata.x_sol[self.idx_x_bin])

        nlpdata.prev_solution = self.solver(
            x0=ca.vertcat(
                nlpdata.x_sol[:nlpdata.x0.shape[0]
                              ], np.zeros((self.nr_g * 1, 1))
            ),
            lbx=ca.vertcat(lbx, np.zeros((self.nr_g * 1, 1))),
            ubx=ca.vertcat(ubx, ca.inf * np.ones((self.nr_g * 1, 1))),
            lbg=self.lbg,
            ubg=self.ubg,
            p=ca.vertcat(nlpdata.p, nlpdata.lbg, nlpdata.ubg)
        )
        nlpdata.solved = self.collect_stats()[0]
        self.stats["fnlp.time"] += sum(
            [v for k, v in stats.items() if "t_proc" in k]
        )
        self.stats["fnlp.iter"] += stats["iter_count"]
        if not nlpdata.solved:
            print("MILP not solved")
        return nlpdata


def benders_algorithm(problem, data, stats, is_orig=False):
    """Create benders algorithm."""
    tic()
    toc()
    print("Setup NLP solver...")
    nlp = NlpSolver(problem, stats, is_orig=is_orig)
    toc()
    print("Setup FNLP solver...")
    fnlp = FeasibilityNLP(problem, stats)
    toc()
    print("Setup MILP solver...")
    benders_milp = BendersMasterMILP(problem, stats)
    t_load = toc()

    print("Solver initialized.")
    # Benders algorithm
    lb = -ca.inf
    ub = ca.inf
    tolerance = 0.04
    feasible = True
    data = nlp.solve(data)
    x_bar = data.x_sol
    x_star = x_bar
    prev_feasible = True
    while lb + tolerance < ub and feasible:
        toc()
        # Solve MILP-BENDERS and set lower bound:
        data = benders_milp.solve(data, prev_feasible=prev_feasible)
        feasible = data.solved
        lb = data.obj_val
        # x_hat = data.x_sol

        # Obtain new linearization point for NLP:
        data = nlp.solve(data, set_x_bin=True)
        x_bar = data.x_sol
        prev_feasible = data.solved
        if not prev_feasible:
            data = fnlp.solve(data)
            x_bar = data.x_sol
            print("Infeasible")
        elif data.obj_val < ub:
            ub = data.obj_val
            x_star = x_bar
            print("Feasible")

        print(f"{ub=} {lb=}")
        print(f"{x_bar=}")

    t_total = toc()
    print(f"{t_total=} of with calc: {t_total - t_load}")
    return data, x_star


def idea_algorithm(problem, data, stats, is_orig=False):
    """Create benders algorithm."""
    tic()
    toc()
    print("Setup NLP solver...")
    nlp = NlpSolver(problem, stats, is_orig=is_orig)
    toc()
    print("Setup FNLP solver...")
    fnlp = FeasibilityNLP(problem, stats)
    toc()
    print("Setup MILP solver...")
    benders_milp = BendersConstraintMILP(problem, stats)
    toc(reset=True)

    print("Solver initialized.")
    # Benders algorithm
    lb = -ca.inf
    ub = ca.inf
    tolerance = 0.04
    feasible = True
    data = nlp.solve(data)
    x_bar = data.x_sol
    x_star = x_bar
    prev_feasible = True
    while lb + tolerance < ub and feasible:
        toc()
        # Solve MILP-BENDERS and set lower bound:
        data = benders_milp.solve(data, prev_feasible=prev_feasible)
        feasible = data.solved
        lb = data.obj_val
        # x_hat = data.x_sol

        # Obtain new linearization point for NLP:
        data = nlp.solve(data, set_x_bin=True)
        x_bar = data.x_sol
        prev_feasible = data.solved
        if not prev_feasible:
            data = fnlp.solve(data)
            x_bar = data.x_sol
            print("Infeasible")
        elif data.obj_val < ub:
            ub = data.obj_val
            x_star = x_bar
            print("Feasible")

        print(f"{ub=} {lb=}")
        print(f"{x_bar=}")

    return data, x_star


if __name__ == "__main__":
    if len(argv) == 1:
        print("Usage: mode problem")
        print("Available modes are: benders, idea, ...")
        print("Available problems are: dummy, dummy2, orig, ...")

    if len(argv) > 1:
        mode = argv[1]
    else:
        mode = "benders"

    if len(argv) > 2:
        problem = argv[2]
    else:
        problem = "orig"

    new_inf = 1e3
    if problem == "dummy":
        problem, data = create_dummy_problem()
    elif problem == "dummy2":
        problem, data = create_dummy_problem_2()
    elif problem == "orig":
        problem, data = extract()
        new_inf = 1e5
    else:
        raise Exception(f"No {problem=}")

    # make_bounded(data, new_inf=new_inf)
    print("Problem loaded")
    stats = Stats({})
    if mode == "benders":
        data, x_star = benders_algorithm(
            problem, data, stats, is_orig=(problem == "orig")
        )
    elif mode == "idea":
        data, x_star = idea_algorithm(problem, data,  stats)

    stats.print()
    print(x_star)
    if WITH_PLOT:
        plt.show()
