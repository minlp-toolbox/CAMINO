# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from camino.settings import GlobalSettings
from camino.problems.problem_collection import create_ocp_unstable_system
import pickle

from camino.utils import latexify
from camino.utils.conversion import to_0d

filename = "2025-07-09_17:21:38_cia_unstable_ocp_generic.pkl"

file_path = os.path.join(GlobalSettings.OUT_DIR, filename)
with open(file_path, 'rb') as f:
    stats = pickle.load(f)

stats_df = pd.DataFrame(stats)
print(stats_df.columns)
print(stats_df.head())

best_iter_idx = stats_df.loc[(stats_df['success'] == True)].sort_values("sol_obj").index[0]
best_sol_obj = stats_df.loc[stats_df.index == best_iter_idx, "sol_obj"]
best_sol_x = stats_df.loc[stats_df.index == best_iter_idx, "sol_x"].iloc[0]

# Construct the stcs problem
prob, data, s = create_ocp_unstable_system()

meta = prob.meta
state = to_0d(best_sol_x)[meta.idx_state].reshape(-1, meta.n_state)
state = np.vstack([meta.initial_state, state])
control_b = to_0d(best_sol_x)[meta.idx_bin_control].reshape(-1, meta.n_discrete_control)
control_b = np.vstack([control_b, control_b[-1, :]])
time_array = np.arange(0, (prob.meta.N_horizon+1)*prob.meta.dt, prob.meta.dt)

# Plot optimal trajectory
latexify()
import matplotlib
matplotlib.rcParams.update({"lines.linewidth":1})
fig, axs = plt.subplots(2, 1, figsize=(3, 3), sharex=True)

axs[0].plot(time_array, state, color="tab:blue", label = "$T_\mathsf{fpsc}$")
axs[0].axhline(0.7, color='red', linestyle=':')
axs[0].set_ylabel("$x$")
axs[0].set_ylim(0.65, 0.95)
axs[1].step(time_array, control_b, color="tab:orange", label = "$T_\mathsf{fpsc}$", marker='.', where="post")
axs[1].set_ylabel("$u$")
axs[1].set_xlabel("Time (s)")
axs[1].set_xlim(0, 1.5)

plt.tight_layout()


plt.show()