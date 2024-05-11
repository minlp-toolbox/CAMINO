"""Base algorithm for feasibility pumps."""

from copy import deepcopy
from minlp_algorithms.settings import Settings
from minlp_algorithms.solvers.subsolvers.nlp import NlpSolver
from minlp_algorithms.stats import Stats
from minlp_algorithms.data import MinlpData
from minlp_algorithms.problem import MinlpProblem
from minlp_algorithms.solvers.pumps.utils import integer_error, create_rounded_data, perturbe_x, any_equal, \
        random_perturbe_x
from minlp_algorithms.utils import toc, logging
from minlp_algorithms.utils.conversion import to_0d
from minlp_algorithms.solvers import MiSolverClass

logger = logging.getLogger(__name__)


class PumpBase(MiSolverClass):

    def __init__(
        self,
        problem: MinlpProblem,
        data: MinlpData,
        stats: Stats,
        settings: Settings,
        pump,
        nlp=None,
    ):
        """Create a solver class."""
        super(PumpBase, self).__init__(problem, data, stats, settings)
        self.pump = pump
        self.idx_x_bin = problem.idx_x_bin
        if nlp is None:
            nlp = NlpSolver(problem, stats, settings)
        self.nlp = nlp

    def solve(self, nlpdata: MinlpData, integers_relaxed: bool = False) -> MinlpData:
        """Solve the problem."""
        if self.stats.relaxed is None:
            if not integers_relaxed:
                nlpdata = self.nlp.solve(nlpdata)
            self.stats.relaxed = nlpdata
        else:
            nlpdata = self.stats.relaxed

        relaxed_value = nlpdata.obj_val
        prev_x = []
        distances = [integer_error(nlpdata.x_sol[self.idx_x_bin])]
        while distances[-1] > self.settings.CONSTRAINT_TOL and self.stats["iter_nr"] < self.settings.PUMP_MAX_ITER:
            datarounded = create_rounded_data(nlpdata, self.idx_x_bin)
            require_restart = False
            for i, sol in enumerate(datarounded.solutions_all):
                new_x = to_0d(sol["x"])
                perturbe_remaining = self.settings.PARALLEL_SOLUTIONS
                while any_equal(new_x, prev_x, self.idx_x_bin) and perturbe_remaining > 0:
                    new_x = perturbe_x(to_0d(nlpdata.solutions_all[i]["x"]), self.idx_x_bin)
                    perturbe_remaining -= 1

                datarounded.prev_solutions[i]["x"] = new_x
                if perturbe_remaining == 0:
                    require_restart = True

                prev_x.append(new_x)

            if not require_restart:
                data = self.pump.solve(datarounded, int_error=distances[-1], obj_val=relaxed_value)
                distances.append(integer_error(data.x_sol[self.idx_x_bin]))

            if (
                len(distances) > self.settings.PUMP_MAX_STEP_IMPROVEMENTS
                and distances[-self.settings.PUMP_MAX_STEP_IMPROVEMENTS - 1] < distances[-1]
            ) or require_restart:
                data.prev_solutions[0]["x"] = random_perturbe_x(data.x_sol, self.idx_x_bin)
                data = self.pump.solve(data, int_error=distances[-1], obj_val=relaxed_value)
                distances.append(integer_error(data.x_sol[self.idx_x_bin]))

            # Added heuristic, not present in the original implementation
            if distances[-1] < self.settings.CONSTRAINT_INT_TOL:
                datarounded = self.nlp.solve(create_rounded_data(data, self.idx_x_bin), True)
                if self.update_best_solutions(datarounded):
                    return self.get_best_solutions(datarounded)

            self.stats["iter_nr"] += 1
            logger.info(f"Iteration {self.stats['iter_nr']} finished")

        self.stats["total_time_calc"] += toc(reset=True)
        data = self.nlp.solve(data, True)
        self.update_best_solutions(data)
        return self.get_best_solutions(data)

    def reset(self, nlpdata: MinlpData):
        """Reset problem data."""

    def warmstart(self, nlpdata: MinlpData):
        """Warmstart."""
        self.update_best_solutions(nlpdata)
