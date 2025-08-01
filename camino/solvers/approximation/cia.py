# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

"""Implementation of the CIA algorithm."""

import copy
import datetime
import casadi as ca
import numpy as np
from typing import Tuple
from camino.solvers.subsolvers.nlp import NlpSolver
from camino.solvers import SolverClass, Stats, MinlpProblem, MinlpData
from camino.settings import Settings
from camino.utils import toc, logging
from camino.utils.conversion import to_0d
try:
    from pycombina import BinApprox, CombinaBnB
except Exception:
    class BinApprox:
        pass

    class CombinaBnB:
        pass

logger = logging.getLogger(__name__)


def simulate(x0, u, f_dyn):
    N = u.shape[0]
    x = []
    for t in range(N):
        if t == 0:
            x.append(to_0d(f_dyn(x0, u[t, :])))
        else:
            x.append(to_0d(f_dyn(x[-1], u[t, :])))
    return np.array(x).flatten().tolist()

def to_list(dt, min_time, nr_b):
    """Create a min up or downtime list."""
    if isinstance(min_time, int):
        return [dt * min_time for _ in range(nr_b)]
    else:
        return [dt * min_time[i] for i in range(nr_b)]


class PycombinaSolver(SolverClass):
    """
    Create NLP solver.

    This solver solves an NLP problem. This is either relaxed or
    the binaries are fixed.
    """

    def __init__(self, problem: MinlpProblem, stats: Stats, s: Settings):
        """Create Pycombina solver."""
        super(PycombinaSolver, self).__init__(problem, stats, s)
        self.idx_x_integer = problem.idx_x_integer
        self.meta = copy.deepcopy(problem.meta)

    def solve(self, nlpdata: MinlpData) -> MinlpData:
        """Solve CIA problem."""
        b_rel = to_0d(nlpdata.x_sol)[self.meta.idx_bin_control]
        if len(b_rel.shape) == 1:  # flatten array
            b_rel = b_rel.reshape(-1, self.meta.n_discrete_control)
        b_rel = np.hstack([np.asarray(b_rel), np.array(
            1-b_rel.sum(axis=1).reshape(-1, 1))])  # Make sos1 structure

        # Ensure values are not out of range due to numerical effects
        b_rel[b_rel < 0] = 0
        b_rel[b_rel > 1.0] = 1

        N = b_rel.shape[0] + 1
        if isinstance(self.meta.dt, list):
            if isinstance(self.meta.dt[0], datetime.timedelta):
                time_array = [0]
                for t in self.meta.dt:
                    time_array.append(time_array[-1] + t.total_seconds())
                time_array = np.array(time_array)
            else:
                raise NotImplementedError()
        else:
            # Assumes uniform grid
            time_array = np.arange(0, N*self.meta.dt, self.meta.dt)
        binapprox = BinApprox(time_array, b_rel)

        value_set = False
        if self.meta.min_downtime is not None:
            value_set = True
            if isinstance(self.meta.min_downtime, np.ndarray):
                if self.meta.min_downtime.shape[0] == self.meta.n_discrete_control:
                    min_downtimes = np.concatenate(
                        [self.meta.min_downtime, np.zeros(1)])
                    binapprox.set_min_down_times(min_downtimes)
            else:
                binapprox.set_min_down_times(
                    to_list(self.meta.dt, self.meta.min_downtime, b_rel.shape[1]))

        if self.meta.min_uptime is not None:
            value_set = True
            if isinstance(self.meta.min_uptime, np.ndarray):
                if self.meta.min_uptime.shape[0] == self.meta.n_discrete_control:
                    min_uptimes = np.concatenate(
                        [self.meta.min_uptime, np.zeros(1)])
                    binapprox.set_min_up_times(min_uptimes)
            else:
                binapprox.set_min_up_times(
                    to_list(self.meta.dt, self.meta.min_uptime, b_rel.shape[1]))

        # binapprox.set_n_max_switches(...)
        # binapprox.set_max_up_times(...)

        if not value_set:
            raise Exception("Minimum uptime or downtime needs to be set!")

        combina = CombinaBnB(binapprox)
        combina.solve()
        b_bin = binapprox.b_bin[:-1, :].T.flatten()
        idx_bin_control = np.array(
            self.meta.idx_bin_control).flatten().tolist()
        nlpdata.x_sol[idx_bin_control] = b_bin

        return nlpdata
