# This file is part of CAMINO
# Copyright (C) 2024  Andrea Ghezzi, Wim Van Roy, Sebastian Sager, Moritz Diehl
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
from colored import stylize
import pandas as pd
import seaborn as sns
from camino.problems.problem_collection import create_ocp_unstable_system
from camino.problems.solarsys import create_stcs_problem
from camino.utils import colored, latexify
from camino.utils.conversion import convert_to_flat_list, to_0d
import numpy as np
import matplotlib.pyplot as plt
from camino.problems.solarsys.ambient import Ambient, Timing
from camino.problems.solarsys.system import System
from camino.settings import GlobalSettings
import pickle

latexify()





def plot_stcs(stats_df, best_iter_idx, best_sol_obj, best_sol_x, solution_method, simplified):

    if 's-b-miqp' in solution_method:
        # Plot objective of iterations
        # Due to solution pool there are multiple NLP solution per iteration, we consider the one with lowest objective.

        # print(stats_df.loc[(stats_df['sol_obj']>= 2051) & (stats['sol_obj'] <= 2052)])
        obj_milp = stats_df.loc[stats_df["iter_type"] == "LB-MILP", "sol_obj"]
        obj_miqp = stats_df.loc[stats_df["iter_type"] == "BR-MIQP", "sol_obj"]
        obj_nlp = stats_df.loc[stats_df["iter_type"] == "NLP", ["iter_nr", "sol_obj"]].groupby("iter_nr").min()
        lb_sequence = stats_df.loc[stats_df["iter_type"] == "NLP", ["iter_nr", "lb"]].groupby("iter_nr").min()
        ub_sequence = stats_df.loc[stats_df["iter_type"] == "NLP", ["iter_nr", "ub"]].groupby("iter_nr").min()

        plt.figure(figsize=(5,3))
        plt.plot(obj_nlp.values[:-1], marker='.', label='$J(y_k)$')
        plt.plot(obj_miqp.values[:-1], marker='.', label='$V_{\mathrm{MIQP}, k}$')
        plt.plot(obj_milp.values[:-1], marker='.', label='$V_{\mathrm{MILP}, k}$')
        plt.plot(ub_sequence.values[:-1], label='UB')
        plt.plot(lb_sequence.values[:-1], label='LB')
        plt.ylim(-2000, 1e4)
        plt.xlim(0,)
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join("results", filename+"_iteration_values.pdf"), bbox_inches='tight')

        # Plot wall-time of each solver

        grouped_stats_df = stats_df.loc[stats_df["iter_type"] == "NLP", stats_df.columns.tolist()].groupby("iter_nr").min("sol_obj")

        fig, ax = plt.subplots(1, 1, figsize=(4.5, 1.5))
        ax = sns.lineplot(grouped_stats_df[[elm for elm in stats_df.columns.tolist() if ".time_wall" in elm]]/3600, ax=ax, linewidth=1.5, alpha=0.8)
        ax = sns.lineplot(grouped_stats_df[[elm for elm in stats_df.columns.tolist() if ".time_wall" in elm]].sum(axis=1)/3600, ax=ax, color='gray', label="Sum")
        legend_handles, _= ax.get_legend_handles_labels()
        ax.legend(legend_handles, ["NLP", "BR-MIQP", "LB-MILP", "F-NLP", "Sum"],loc='center right', ncol=1, bbox_to_anchor=(1.33, 0.5), fontsize=9)
        ax.set(xlabel=r"Iteration")
        ax.set(ylabel=r"Cumulative runtime (h)")
        ax.set(xlim=[0, grouped_stats_df.shape[0]])
        ax.set(ylim=[0, 13])
        plt.savefig(os.path.join("results", filename+"_runtime.pdf"), bbox_inches='tight')


    # Plot the optimal trajectory for state and binary variables
    # Construct the stcs problem
    timing = Timing(simplified=simplified)
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

    prob, data, s = create_stcs_problem(simplified=simplified)

    meta = prob.meta
    state = to_0d(best_sol_x)[meta.idx_state].reshape(-1, meta.n_state)
    state = np.vstack([meta.initial_state, state[:-1, :]])
    control_u = to_0d(best_sol_x)[meta.idx_control].reshape(-1, meta.n_continuous_control)
    control_b = to_0d(best_sol_x)[meta.idx_bin_control].reshape(-1, meta.n_discrete_control)

    state = pd.DataFrame(state)
    state.columns = ['T_hts_0', 'T_hts_1', 'T_hts_2', 'T_hts_3',
        "T_lts","T_fpsc","T_fpsc_s","T_vtsc","T_vtsc_s","T_pscf","T_pscr",
        "T_shx_psc_0", "T_shx_psc_1", "T_shx_psc_2", "T_shx_psc_3",
        "T_shx_ssc_0", "T_shx_ssc_1", "T_shx_ssc_2", "T_shx_ssc_3"]
    control_u = pd.DataFrame(control_u)
    control_u.columns = ["v_ppsc", "p_mpsc", "v_pssc", "P_g", "mdot_o_hts_b", "mdot_i_hts_b"]
    control_b = pd.DataFrame(control_b)
    control_b.columns = ["b_ac", "b_fc", "b_hp"]
    daytime_array = daytime_array[:-1]

    # Plot optimal trajectory
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
    ax0a.plot(daytime_array, params.I_fpsc[:-1]/1e3, color = "darkorange", label = "$I_\mathsf{fpsc}$", linestyle='-.')
    ax0a.plot(daytime_array, params.I_vtsc[:-1]/1e3, color = "gold", label = "$I_\mathsf{vtsc}$", linestyle='-.')
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

    axs[4].plot(daytime_array, params.T_amb[:-1], "-.", label = "$T_\mathsf{amb}$")
    axs[4].set_ylim(10, 30)
    axs[4].set_ylabel("Temp. (°C)")
    ax4a = axs[4].twinx()
    ax4a.plot(daytime_array, params.Qdot_c[:-1]/1000, "-.", color='tab:orange', label = "$\dot{Q}_\mathsf{lc}$")
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
    fig.align_ylabels(axs)

    fig.savefig(os.path.join("results", filename+"_plot_traj.pdf"), bbox_inches='tight')
    plt.show()


def plot_unstable_ocp(stats_df, best_iter_idx, best_sol_obj, best_sol_x, solution_method):
    # Construct the unstable ocp problem
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
    fig.savefig(os.path.join("results", filename+"_plot_traj.pdf"), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    try:
        filename = sys.argv[1]
    except:
        filename = "2025-07-09_18:40:11_cia_stcs_generic.pkl"
        print(f"Filename not provided, loading the results with filename {filename}")

    file_path = os.path.join(GlobalSettings.OUT_DIR, filename)
    with open(file_path, 'rb') as f:
        stats = pickle.load(f)

    stats_df = pd.DataFrame(stats)
    if 'cia+s-b-miqp' in filename:
        stats_df = stats_df.loc[stats_df.index>1]
        solution_method='s-b-miqp'
    elif "cia" in filename:
        stats_df = stats_df.loc[stats_df.index>1]
        solution_method='cia'
    elif 's-b-miqp' in filename:
        stats_df = stats_df.loc[stats_df.index>1]
        solution_method='s-b-miqp'
    elif 'bonmin' in filename:
        stats_df = stats_df
        solution_method='bonmin'

    if solution_method == 'bonmin':
        best_iter_idx = 0
        best_sol_obj = stats_df["sol_obj"]
        best_sol_x = stats_df.loc[stats_df.index == best_iter_idx, "sol_x"].iloc[0]
    else:
        best_iter_idx = stats_df.loc[(stats_df['success'] == True) & (stats_df["iter_type"] == "NLP")].sort_values("sol_obj").index[0]
        best_sol_obj = stats_df.loc[stats_df.index == best_iter_idx, "sol_obj"]
        best_sol_x = stats_df.loc[stats_df.index == best_iter_idx, "sol_x"].iloc[0]

    print(f"\n Best objective: {best_sol_obj} \n")
    print(f"\n Computation time: {stats[-1]['time']} \n")

    if "stcs" in filename:
        if "simplified" in filename:
            plot_stcs(stats_df, best_iter_idx, best_sol_obj, best_sol_x, solution_method, simplified=True)
        else:
            plot_stcs(stats_df, best_iter_idx, best_sol_obj, best_sol_x, solution_method, simplified=False)
    elif "unstable_ocp" in filename:
        plot_unstable_ocp(stats_df, best_iter_idx, best_sol_obj, best_sol_x, solution_method)
    else:
        raise ValueError("This script creates plots for optimal control problems, available inputs: 'stcs', 'unstable_ocp'")