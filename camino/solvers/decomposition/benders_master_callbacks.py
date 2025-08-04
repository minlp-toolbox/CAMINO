import logging
import casadi as ca
import numpy as np

from camino.data import MinlpData
from camino.solvers.subsolvers.nlp import NlpSolver
from camino.utils import colored


logger = logging.getLogger(__name__)


class MipsolCallback(ca.Callback):
    def __init__(self, name, opts={}, nlpdata: MinlpData = None, nlp_solver: NlpSolver = None, fnlp_solver: NlpSolver = None):
        ca.Callback.__init__(self)
        # Define the callback signature: 5 inputs, 3 outputs
        # Inputs: x_solution, obj_val, obj_best, obj_bound, sol_count
        # Outputs: add_lazy_flag, A_lazy, b_lazy

        # Input sparsities
        self.nlp_solver = nlp_solver
        self.fnlp_solver = fnlp_solver
        self.nlpdata = nlpdata
        self.construct(name, opts)

    def get_n_in(self):
        return 5

    def get_n_out(self):
        return 3

    def get_sparsity_in(self, i):
        if i == 0:  # x_solution
            return ca.Sparsity.dense(self.nlpdata.x_sol.shape) # nx variables
        elif i == 1:  # obj_val
            return ca.Sparsity.dense(1, 1)
        elif i == 2:  # obj_best
            return ca.Sparsity.dense(1, 1)
        elif i == 3:  # obj_bound
            return ca.Sparsity.dense(1, 1)
        elif i == 4:  # sol_count
            return ca.Sparsity.dense(1, 1)
        else:
            return ca.Sparsity(0, 0)

    def get_sparsity_out(self, i):
        if i == 0:  # add_lazy_flag
            return ca.Sparsity.dense(1, 1)
        elif i == 1:  # A_lazy (constraint coefficients)
            return ca.Sparsity.dense(self.nlpdata.x_sol.shape)  # cns dimension (nx x 1)
        elif i == 2:  # b_lazy (right-hand side)
            return ca.Sparsity.dense(1, 1)
        else:
            return ca.Sparsity(0, 0)

    def add_benders_cut(self):
        lambda_k = -self.nlpdata.lam_x_sol[self.nlp_solver.idx_x_integer]
        A_k = ca.vertcat(lambda_k, -1)
        b_k = lambda_k.T @ self.nlpdata.x_sol[self.nlp_solver.idx_x_integer] - self.nlpdata.obj_val
        return A_k, b_k

    # def _generate_cut_equation(
    #     self, x, x_sol, x_sol_sub_set, lam_g, lam_x, p, lbg, ubg, prev_feasible
    # ):
    #     if prev_feasible:
    #         lambda_k = -lam_x[self.nlp_solver.idx_x_integer]
    #         f_k = self.nlp_solver.f(x_sol, p)
    #         g_k = f_k + lambda_k.T @ (x - x_sol_sub_set) - self._nu
    #         print(f"{lambda_k=}, {f_k=}, {g_k=}")
    #     else:  # Not feasible solution
    #         h_k = self.g(x_sol, p)
    #         jac_h_k = self.jac_g_bin(x_sol, p)
    #         g_k = lam_g.T @ (
    #             h_k
    #             + jac_h_k @ (x - x_sol_sub_set)
    #             - (lam_g > 0) * np.where(np.isinf(ubg), 0, ubg)
    #             + (lam_g < 0) * np.where(np.isinf(lbg), 0, lbg)
    #         )

    #     return g_k

    # def _add_solution(self, nlpdata, x_sol, lam_g_sol, lam_x_sol, prev_feasible):
    #     """Create cut."""
    #     g_k = self._generate_cut_equation(
    #         x_sol,
    #         x_sol[: self.nlpdata.x0.shape[0]],
    #         x_sol[self.nlp_solver.idx_x_integer],
    #         lam_g_sol,
    #         lam_x_sol,
    #         nlpdata.p,
    #         nlpdata.lbg,
    #         nlpdata.ubg,
    #         prev_feasible,
    #     )
    #     self.cut_id += 1
    #     return g_k, 0

    # The actual callback function that gets executed
    def eval(self, arg):
        # print("MIPSOL callback executed!")
        # Extract the arguments
        x_solution = np.array(arg[0])
        obj_val = float(arg[1])
        obj_best = float(arg[2])
        obj_bound = float(arg[3])
        sol_count = int(arg[4])
        # print(f"  Current solution: x = {x_solution}")
        # print(f"  Objective value: {obj_val}")
        # print(f"  Best objective: {obj_best}")
        # print(f"  Objective bound: {obj_bound}")
        # print(f"  Solution count: {sol_count}")

        self.nlpdata.x_sol[self.nlp_solver.idx_x_integer] = x_solution[self.nlp_solver.idx_x_integer]

        self.nlpdata = self.nlp_solver.solve(self.nlpdata, set_x_bin=True)
        if not np.all(self.nlpdata.solved_all):
            # Solve NLPF(y^k)
            self.nlpdata = self.fnlp_solver.solve(self.nlpdata)
            logger.info(colored("Feasibility NLP solved.", "yellow"))

        # A_lazy, b_lazy = self._add_solution(self.nlpdata, self.nlpdata.x_sol, self.nlpdata.lam_g_sol, self.nlpdata.lam_x_sol, self.nlpdata.solved)
        A_lazy, b_lazy = self.add_benders_cut()
        lazy_flag = ca.DM(1)

        # print(f"  A_lazy: {A_lazy}, A_lazy_shape: {A_lazy.shape}, A_lazy_type: {type(A_lazy)}")
        # print(f"  b_lazy: {b_lazy}, b_lazy_shape: {b_lazy.shape}, b_lazy_type: {type(b_lazy)}")
        # print(f"{lazy_flag=}")

        # Return the outputs
        return [lazy_flag, A_lazy, b_lazy]