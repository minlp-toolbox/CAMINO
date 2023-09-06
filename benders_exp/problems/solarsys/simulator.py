# Adrian Buerger, 2022
# Adapted by Andrea Ghezzi and Wim Van Roy, 2023

import numpy as np
from datetime import timedelta

import logging
from benders_exp.problems.solarsys.system import System, ca
from benders_exp.problems.solarsys.ambient import Ambient
from benders_exp.cache_utils import CachedFunction
from benders_exp.utils import convert_to_flat_list


logger = logging.getLogger(__name__)


class Simulator(System):

    @property
    def time_grid(self):
        return self._timing.time_grid

    @property
    def time_steps(self):
        return [self._dt.total_seconds() for _ in range(self._N)] # NOTE Uniform grid

    @property
    def x_data(self):
        try:
            return np.asarray(self._x_data)
        except AttributeError:
            msg = "Simulation results (states) not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    @property
    def u_data(self):
        try:
            return self._u_data
        except AttributeError:
            msg = "Continuous controls not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    @property
    def b_data(self):
        try:
            return self._b_data
        except AttributeError:
            msg = "Binary controls not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    @property
    def c_data(self):
        try:
            return self._c_data
        except AttributeError:
            msg = "Ambient parameters not available yet, call solve() first."
            logging.error(msg)
            raise RuntimeError(msg)

    def _get_ambient_paramaters(self):

        self._c_data = []
        for i in range(self._N):
            self._c_data.append(convert_to_flat_list(
                self.nc,
                self.c_index,
                self._ambient.interpolate(self._ambient.get_t0() + i * self._dt)))
        self._c_data = ca.DM(self._c_data)

    def __init__(self, ambient: Ambient, N: int, dt: timedelta):

        logger.debug("Initializing simulator ...")
        super().__init__()
        self._ambient = ambient
        self._N = N
        self._dt = dt
        self._integrator = self.get_integrator()
        self._get_ambient_paramaters()
        logger.debug("Simulator initialized.")

    def _set_initial_state(self):

        self._x_data = [convert_to_flat_list(self.nx, self.x_index, self.get_default_initial_state())]

    def _setup_controls(self):

        self._u_data = []
        self._b_data = []
        self._remaining_min_up_time = 0

    def _initialize_controls(self):

        self._u_data.append(np.zeros(self.nu))
        self._b_data.append(np.zeros(self.nb))

    def _set_b_ac(self, pos):

        T_hts = self._x_data[-1][self.x_index["T_hts"][0]]
        T_lts = self._x_data[-1][self.x_index["T_lts"]]

        T_amb = self.c_data[pos, self.c_index["T_amb"]]

        if self._remaining_min_up_time > 0:

            self._b_data[-1][self.b_index["b_ac"]] = self._b_data[-2][
                self.b_index["b_ac"]
            ]

            if self._b_data[-1][self.b_index["b_ac"]] == 1:
                self._remaining_min_up_time -= self.time_steps[pos]

        elif (T_hts < self.p["T_ac_ht_min"]) or (T_lts < self.p["T_ac_lt_min"]):

            self._b_data[-1][self.b_index["b_ac"]] = 0

        elif (
            (T_hts > self.p["T_ac_ht_min"] + self.p_csim["dT_ac_ht"])
            or (T_lts < self.p["T_ac_lt_max"] - self.p_csim["dT_ac_lt"])
            or (T_lts > self.p["T_ac_lt_min"] + self.p_csim["dT_ac_lt"])
        ):

            self._b_data[-1][self.b_index["b_ac"]] = 1

            try:
                if (self._b_data[-2][self.b_index["b_ac"]] != 1) and (
                    self._b_data[-1][self.b_index["b_ac"]] == 1
                ):
                    self._remaining_min_up_time = (
                        self.p_op["acm"]["min_up_time"][self.b_index["b_ac"]]
                        - self.time_steps[pos]
                    )
            except IndexError:
                self._remaining_min_up_time = 0

    def _set_b_hp(self):

        T_lts = self._x_data[-1][self.x_index["T_lts"]]

        if self._b_data[-1][self.b_index["b_ac"]] == 1:

            self._b_data[-1][self.b_index["b_hp"]] = 0

        elif T_lts > 20.0:

            self._b_data[-1][self.b_index["b_hp"]] = 1

    def _set_v_ppsc(self):

        T_fpsc = self._x_data[-1][self.x_index["T_fpsc"]]
        T_vtsc = self._x_data[-1][self.x_index["T_vtsc"]]
        T_hts = self._x_data[-1][self.x_index["T_hts"][0]]

        dT = max(T_fpsc, T_vtsc) - T_hts

        v_ppsc = (
            self.p_op["v_ppsc"]["max"]
            / (self.p_csim["dT_sc_ub"] - self.p_csim["dT_sc_lb"])
        ) * (
            max(self.p_csim["dT_sc_lb"], min(self.p_csim["dT_sc_ub"], dT))
            - self.p_csim["dT_sc_lb"]
        )

        self._u_data[-1][self.u_index["v_ppsc"]] = v_ppsc

    def _set_p_mpsc(self):

        T_fpsc = self._x_data[-1][self.x_index["T_fpsc"]]
        T_vtsc = self._x_data[-1][self.x_index["T_vtsc"]]

        dT = T_fpsc - T_vtsc

        p_mpsc = (
            (self.p_op["p_mpsc"]["max"] - self.p_op["p_mpsc"]["min_real"])
            / (self.p_csim["dT_vtsc_fpsc_ub"] - self.p_csim["dT_vtsc_fpsc_lb"])
        ) * (
            max(self.p_csim["dT_vtsc_fpsc_lb"], min(self.p_csim["dT_vtsc_fpsc_ub"], dT))
            - self.p_csim["dT_vtsc_fpsc_lb"]
        )

        self._u_data[-1][self.u_index["p_mpsc"]] = p_mpsc

    def _set_v_pssc(self):

        T_shx_ssc = self._x_data[-1][self.x_index["T_shx_ssc"][-1]]
        T_hts = self._x_data[-1][self.x_index["T_hts"][0]]

        dT = T_shx_ssc - T_hts

        v_pssc = (
            self.p_op["v_pssc"]["max"]
            / (self.p_csim["dT_sc_ub"] - self.p_csim["dT_sc_lb"])
        ) * (
            max(self.p_csim["dT_sc_lb"], min(self.p_csim["dT_sc_ub"], dT))
            - self.p_csim["dT_sc_lb"]
        )

        self._u_data[-1][self.u_index["v_pssc"]] = v_pssc

    def _set_mdot_o_hts_b(self):

        T_shx_ssc = self._x_data[-1][self.x_index["T_shx_ssc"][-1]]
        T_sc_feed_max = self.p_op["T_sc"]["T_feed_max"]

        dT = T_shx_ssc - T_sc_feed_max

        mdot_o_hts_b = (
            (1.0 / (self.p_csim["dT_o_hts_b_ub"] - self.p_csim["dT_o_hts_b_lb"]))
            * (
                max(self.p_csim["dT_o_hts_b_lb"], min(self.p_csim["dT_o_hts_b_ub"], dT))
                - self.p_csim["dT_o_hts_b_lb"]
            )
            * self.p["mdot_ssc_max"]
            * self._u_data[-1][self.u_index["v_pssc"]]
        )

        self._u_data[-1][self.u_index["mdot_o_hts_b"]] = mdot_o_hts_b

    def _set_mdot_i_hts_b(self):

        T_hts_m = self._x_data[-1][self.x_index["T_hts"][1]]
        T_i_hts_b_active = self.p_csim["T_i_hts_b_active"]

        dT = T_hts_m - T_i_hts_b_active

        mdot_i_hts_b = (
            (1.0 / (self.p_csim["dT_i_hts_b_ub"] - self.p_csim["dT_i_hts_b_lb"]))
            * (
                max(self.p_csim["dT_i_hts_b_lb"], min(self.p_csim["dT_i_hts_b_ub"], dT))
                - self.p_csim["dT_i_hts_b_lb"]
            )
            * self._b_data[-1][self.b_index["b_ac"]]
            * self.p["mdot_ac_ht"]
        )

        self._u_data[-1][self.u_index["mdot_i_hts_b"]] = mdot_i_hts_b

    def _set_controls(self, pos):

        self._initialize_controls()

        self._set_b_ac(pos)
        self._set_b_hp()

        self._set_v_ppsc()
        self._set_p_mpsc()
        self._set_v_pssc()
        self._set_mdot_o_hts_b()
        self._set_mdot_i_hts_b()

    def _run_step(self, pos, step):

        try:
            self._x_data.append(
                np.squeeze(
                    self._integrator(
                        x0=self.x_data[-1],
                        p=ca.veccat(
                            step, self.c_data[pos, :], self.u_data[-1], self.b_data[-1]
                        ),
                    )["xf"]
                )
            )
        except RuntimeError:

            print(
                (
                    "An error occurred during integration, "
                    "repeating states of previous integration step"
                )
            )

        self._x_data.append(self._x_data[-1])

    def _finalize_simulation_results(self):

        self._x_data = ca.horzcat(*self.x_data).T
        self._u_data = ca.horzcat(*self.u_data).T
        self._b_data = ca.horzcat(*self.b_data).T

    def _run_simulation(self):

        logger.info("Running simulation ...")

        self._set_initial_state()
        self._setup_controls()

        for pos, step in enumerate(self.time_steps):

            self._set_controls(pos)
            self._run_step(pos, step)

        self._finalize_simulation_results()

        logger.info("Simulation finished.")

    def solve(self):

        self._run_simulation()


if __name__ == "__main__":
    from datetime import timedelta
    from ambient import Ambient
    from benders_exp.utils import setup_logger, logging

    setup_logger(logging.DEBUG)

    ambient = Ambient()
    dt = timedelta(seconds=900)

    simulator = Simulator(ambient=ambient, N=10, dt=dt)
    simulator.solve()