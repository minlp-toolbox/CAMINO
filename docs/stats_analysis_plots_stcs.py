# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import pandas as pd
from camino.problems.solarsys import create_stcs_problem
from camino.utils import latexify
from camino.utils.conversion import convert_to_flat_list, to_0d
import numpy as np
import matplotlib.pyplot as plt
from camino.problems.solarsys.ambient import Ambient, Timing
from camino.problems.solarsys.system import System
from camino.settings import GlobalSettings
import pickle

filename = "2025-07-04_18:32:02_nlp_stcs_generic.pkl"

file_path = os.path.join(GlobalSettings.OUT_DIR, filename)
with open(file_path, 'rb') as f:
    stats = pickle.load(f)

stats_df = pd.DataFrame(stats)
print(stats_df.columns)
print(stats_df.head())


best_iter_row = stats_df.loc[stats_df["sol_obj"] == stats_df["sol_obj"].min()]
best_sol_obj = stats_df.loc[stats_df["sol_obj"] == stats_df["sol_obj"].min(), "sol_obj"].values
best_sol_x = stats_df.loc[stats_df["sol_obj"] == stats_df["sol_obj"].min(), "sol_x"]
best_sol_x = best_sol_x.iloc[0]


timing = Timing()
n_steps = timing.N
ambient = Ambient(timing)
system = System()
starting_time = ambient.get_t0()
daytime_array = np.zeros(n_steps)
daytime_array = [starting_time]
for dt in ambient.time_steps:
    daytime_array.append(daytime_array[-1] + dt)
daytime_array = daytime_array[:n_steps+1]
tk = ambient.get_t0()
params = []
params.append(convert_to_flat_list(system.nc, system.c_index, ambient.interpolate(tk)))
for k in range(n_steps):
    dt = ambient.time_steps[k]
    tk += dt
    params.append(convert_to_flat_list(system.nc, system.c_index, ambient.interpolate(tk-dt)))
params = pd.DataFrame(params)
params.columns = ["T_amb","I_fpsc","I_vtsc","Qdot_c","P_pv_kWp","p_g"]


# Create problem, get idx, ...
prob, data, s = create_stcs_problem()

meta = prob.meta
state = to_0d(best_sol_x)[meta.idx_state].reshape(-1, meta.n_state)
state = np.vstack([meta.initial_state, state])
control_u = to_0d(best_sol_x)[meta.idx_control].reshape(-1, meta.n_continuous_control)
control_u = np.vstack([control_u[0].reshape(1,-1), control_u])
control_b = to_0d(best_sol_x)[meta.idx_bin_control].reshape(-1, meta.n_discrete_control)
control_b = np.vstack([control_b[0].reshape(1,-1), control_b])

state = pd.DataFrame(state)
state.columns = ['T_hts_0', 'T_hts_1', 'T_hts_2', 'T_hts_3',
    "T_lts","T_fpsc","T_fpsc_s","T_vtsc","T_vtsc_s","T_pscf","T_pscr",
    "T_shx_psc_0", "T_shx_psc_1", "T_shx_psc_2", "T_shx_psc_3",
    "T_shx_ssc_0", "T_shx_ssc_1", "T_shx_ssc_2", "T_shx_ssc_3"]
control_u = pd.DataFrame(control_u)
control_u.columns = ["v_ppsc", "p_mpsc", "v_pssc", "P_g", "mdot_o_hts_b", "mdot_i_hts_b"]
control_b = pd.DataFrame(control_b)
control_b.columns = ["b_ac", "b_fc", "b_hp"]


# optimal trajectory
latexify()
import matplotlib
matplotlib.rcParams.update({"lines.linewidth":1})
fig, axs = plt.subplots(5, 1, figsize=(6, 8), sharex=True)

axs[0].plot(daytime_array, state.T_fpsc, color = "#d62728ff", label = "$T_\mathsf{fpsc}$")
axs[0].plot(daytime_array, state.T_vtsc, color = "#d6272880", label = "$T_\mathsf{vtsc}$")
axs[0].axhline(98, color='gray', linestyle='--')
axs[0].set_ylim(0,105)
axs[0].set_ylabel("Temp. (°C)")
axs[0].legend(loc="upper right", framealpha=0.0)
ax0a = axs[0].twinx()
ax0a.plot(daytime_array, params.I_fpsc/1e3, color = "darkorange", label = "$I_\mathsf{fpsc}$", linestyle='-.')
ax0a.plot(daytime_array, params.I_vtsc/1e3, color = "gold", label = "$I_\mathsf{vtsc}$", linestyle='-.')
ax0a.set_ylim(0,1)
ax0a.set_ylabel("Irrad. (kW/m²)")
ax0a.legend(loc="lower center", framealpha=0.0, bbox_to_anchor=(0.3, 0.05))
ax0a.spines["top"].set_visible(False)

axs[1].plot(daytime_array, state.T_hts_0, color = "#d62728ff", label = "$T_\mathsf{ht,1}$")
axs[1].plot(daytime_array, state.T_hts_3, color = "#d6272880", label = "$T_\mathsf{ht,4}$")
axs[1].axhline(98, color='gray', linestyle='--')
axs[1].set_ylim(35,100)
axs[1].set_ylabel("Temp. (°C)")
axs[1].legend(loc="upper left", ncol=1, framealpha=0.0)

axs[2].plot(daytime_array, state.T_lts, color = "#1f77b4ff", label = "$T_\mathsf{lt,1}$")
axs[2].axhline(8, color='gray', linestyle='--')
axs[2].axhline(18, color='gray', linestyle='--')
axs[2].set_ylim(5,25)
axs[2].set_ylabel("Temp. (°C)")
axs[2].legend(loc="upper left", ncol=1, framealpha=0.0)

axs[3].step(daytime_array, control_b.b_ac, where="post", color = "#1f77b4ff", label = "$b_\mathsf{acm}$", marker='.', markersize=3)
axs[3].step(daytime_array, control_b.b_fc, where="post", color = "#ff7f0eff", label = "$b_\mathsf{fc}$", marker='.', markersize=3)
axs[3].step(daytime_array, control_b.b_hp, where="post", color = "C2", label = "$b_\mathsf{hp}$", marker='.', markersize=3)
# axs[3].step(daytime_array, control_b_rel.b_ac, where="post", color = "#1f77b4ff", linestyle=':')
# axs[3].step(daytime_array, control_b_rel.b_fc, where="post", color = "#ff7f0eff", linestyle=':')
# axs[3].step(daytime_array, control_b_rel.b_hp, where="post", color = "C2", linestyle=':')
axs[3].set_ylim(0, 1.1)
axs[3].set_ylabel("Status \{0, 1\}")
axs[3].legend(loc="upper right", framealpha=0.0)

axs[4].plot(daytime_array, params.T_amb, "-.", label = "$T_\mathsf{amb}$")
axs[4].set_ylim(10, 30)
axs[4].set_ylabel("Temp. (°C)")
ax4a = axs[4].twinx()
ax4a.plot(daytime_array, params.Qdot_c/1000, "-.", color='tab:orange', label = "$\dot{Q}_\mathsf{lc}$")
# ax4a.set_ylim(0, 8)
ax4a.set_ylabel("Load (kW)")
ax4a.spines["top"].set_visible(False)
l , h = axs[4].get_legend_handles_labels()
l1, h1, = ax4a.get_legend_handles_labels()
axs[4].legend(l+l1, h+h1, loc="lower center", framealpha=0.0)


axs[0].set_xlim(daytime_array[0], daytime_array[-1])
axs[-1].set_xlabel("Daytime (dd-hh)")
axs[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%H'))

for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# if SAVE_FIG:
fig.savefig(os.path.join("results", filename+"_plot_traj.pdf"), bbox_inches='tight')

plt.show()
