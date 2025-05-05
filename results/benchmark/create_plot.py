from sys import argv
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib import lines
from matplotlib import colors
import matplotlib
import pandas as pd
import numpy as np
import os
from datetime import datetime


LINESTYLES = [*lines.lineStyles.keys()][:4] * 2
MCOLORS = [*colors.TABLEAU_COLORS.keys()]


os.makedirs("figures", exist_ok=True)

def latexify(fig_width=None, fig_height=None):
    """
    Set up matplotlib's RC params for LaTeX plotting.

    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """
    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf
    if fig_width is None:
        fig_width = 5  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    params = {
        # "backend": "ps",
        "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
        "axes.labelsize": 8,  # fontsize for x and y labels (was 8)
        "axes.titlesize": 8,
        "lines.linewidth": 1,
        "legend.fontsize": 8,  # was 8
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
        "figure.subplot.bottom": 0.15,
        "figure.subplot.top": 0.98,
        "figure.subplot.right": 0.95,
        "figure.subplot.left": 0.11,
        "legend.loc": 'lower right',
    }

    matplotlib.rcParams.update(params)


def compute_ratio(val_min, val_ref, corr=True):
    """Compute ratio."""
    ratios = []
    for imin, jval in zip(val_min, val_ref):
        imin, jval = float(imin), float(jval)
        if corr and imin < 1.0:
            corr = - imin + 1.0
            imin += corr
            jval += corr
        if np.isnan(jval):
            ratios.append(np.inf)
        else:
            if imin==0:
                ratios.append(np.nan)
            else:
                ratios.append(max(jval / imin, 1.0))

    return ratios


def collect_bins_plot(values, name, style, color, min_val=None, max_val=None, ax=None):
    """Collect in bins."""
    values = values[values < 1e5]
    res = values.value_counts().sort_index().cumsum()
    keys = res.keys().to_list()
    values = res.values.tolist()
    if min_val is not None:
        keys = [min_val] + keys
        values = [0] + values

    if max_val is not None:
        keys.append(max_val)
        values.append(values[-1])

    print(name, values)
    if ax:
        ax.plot(keys, values, style, color=color, label=name)
    else:
        plt.plot(keys, values, style, color=color, label=name)


def to_float(val):
    """Convert to float."""
    if type(val) == str:
        if ("Objective" in val) or ("feasible" in val) or ("Error" in val) or ("Calling" in val) or ("g_val" in val) or ("CRASH" in val) or ("FAILED" in val) or ("empty" in val) or ("basic_string" in val) or ("No objective" in val) or ("Suffix values" in val) or ("for indices" in val) or ("has no attribute" in val):
            return np.inf
        elif val == "-inf":
            return np.inf
        elif val == "NAN":
            return np.inf
        else:
            val = float(val)
            if val > 1e20:
                return np.inf
            else:
                return float(val)
    else:
        return float(val)

def create_performance_profile(df, solver_columns, problem_column=None, tau_max=10, num_points=100,
                              log_scale=True, name="perf_plot", title="title", legend_labels=[],
                              xlabel="Within this factor of the best", ylabel="Fraction of problems solved"):
    """
    Create a performance profile plot comparing multiple solvers with support for both
    positive and negative values, and properly handling infinite values.
    Assumes smaller values are better.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the results
    solver_columns : list of str
        List of column names containing the solver results to compare
    problem_column : str, optional
        Column name identifying the problems. If None, the index is used
    tau_max : float, default=10
        Maximum value of the performance ratio to display
    num_points : int, default=100
        Number of points to evaluate the cumulative distribution
    log_scale : bool, default=True
        Whether to use a logarithmic scale for the x-axis
    title : str, default="Performance Profile"
        Plot title
    xlabel : str, default="Performance ratio τ"
        X-axis label
    ylabel : str, default="Probability P(r_{p,s} ≤ τ)"
        Y-axis label
    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    # Make a copy of the data to avoid modifying the original
    data = df.copy()

    # Use index as problem identifier if problem_column is not provided
    if problem_column is None:
        problems = data.index
    else:
        problems = data[problem_column].unique()
    n_problems = len(problems)
    n_solvers = len(solver_columns)
    # Initialize performance ratios matrix
    perf_ratios = np.full((n_problems, n_solvers), np.nan)
    # Process each problem
    for i, problem in enumerate(problems):
        if problem_column is None:
            problem_data = data.loc[problem, solver_columns]
            if isinstance(problem_data, pd.Series):
                problem_data = problem_data.to_frame().T
        else:
            problem_data = data[data[problem_column] == problem][solver_columns]

        # Extract values for all solvers for this problem
        values = []
        valid_solvers = []
        for j, solver in enumerate(solver_columns):
            if solver in problem_data.columns:
                val = problem_data[solver].iloc[0]
                # Filter out NaN and infinite values
                if pd.notna(val) and not np.isinf(val):
                    values.append(val)
                    valid_solvers.append(solver)

        # If no valid values, skip this problem
        if len(values) == 0:
            continue
        values = np.array(values)
        # Handle negative values by transforming to a positive scale while preserving order
        min_value = np.min(values)
        # If all values are negative or mix of negative and positive
        if min_value < 0:
            # Transform values to ensure the best performance is positive
            transformed_values = values - min_value + 1
            # Find the best (smallest) transformed value
            best_perf = np.min(transformed_values)
            # Calculate performance ratios using transformed values
            for idx, solver in enumerate(valid_solvers):
                # print(f"{transformed_values[idx]=}, {best_perf=}")
                solver_j = solver_columns.index(solver)
                perf_ratios[i, solver_j] = transformed_values[idx] / best_perf
        else:
            # For all positive values, use standard calculation
            best_perf = np.min(values)
            # Calculate performance ratios
            for idx, solver in enumerate(valid_solvers):
                solver_j = solver_columns.index(solver)
                # print(f"{values[idx]=}, {best_perf=}")
                perf_ratios[i, solver_j] = values[idx] / best_perf
        # Mark missing, invalid, or infinite values as infinity in performance ratio
        for j, solver in enumerate(solver_columns):
            if solver not in valid_solvers:
                perf_ratios[i, j] = np.inf

    # Create range of tau values (performance ratios)
    if log_scale:
        tau_values = np.logspace(0, np.log10(tau_max), num_points)
    else:
        tau_values = np.linspace(1, tau_max, num_points)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(3, 2))

    # Plot performance profiles for each solver
    for j, solver in enumerate(solver_columns):
        # For each tau value, calculate the fraction of problems where the solver's
        # performance ratio is less than or equal to tau
        profile = [np.sum(perf_ratios[:, j] <= tau) / n_problems for tau in tau_values]
        ax.plot(tau_values, profile, marker='', label=legend_labels[j],
                color=MCOLORS[j], linestyle=LINESTYLES[j])

    # Configure the plot
    if log_scale:
        ax.set_xscale('log')
    ax.set_xlim(1, tau_max)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_yticks(np.linspace(0,1,6))
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # if title=="Objective":
    ax.legend(ncols=2, columnspacing=0.7, handlelength=1.5)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"figures/{datetime.now().strftime('%m-%d')}_{name}.pdf")

    return fig, ax


if len(argv) != 3:
    print("Usage: python create_plot.py <data_file.csv> <key>")
    print("key: cvx or noncvx")
    exit(1)

latexify(6, 4)
data = pd.read_csv(argv[1])
key = argv[2]
assert (key == "cvx" or key == "noncvx")
total_entries = data.shape[0]  # int(input("Amount (e.g. 120):"))


solvers = [f"{key}_bonmin", f"{key}_shot_st", f"{key}_sbmiqp",]
solver_names = ["Bonmin", "SHOT", "S-B-MIQP",]
solvers_obj = [f"{solver}.obj" for solver in solvers]
solvers_calctime = [
    solver + ".calc_time" if "shot" not in solver else f"{solver}.calctime"
    for solver in solvers
]

data[solvers_calctime] = data[solvers_calctime].map(to_float)
data[solvers_obj] = data[solvers_obj].map(to_float)
data['min.calctime'] = np.min(data[solvers_calctime], axis=1)
data.set_index("name", inplace=True)
for solver in solvers:
    if "shot" in solver:
        data[f'{solver}.calctime'][data[f'{solver}.obj'] == np.inf] = np.inf


# New plots:
create_performance_profile(data, solvers_calctime, tau_max=1e5, name=f'{key}_calc_time_profile', title="Wall time", legend_labels=solver_names)
create_performance_profile(data, solvers_obj, log_scale=True, tau_max=1e5, name=f'{key}_obj_profile', title="Objective", legend_labels=solver_names)

plt.show()
raise
# Plot in the arXiv paper:
for s, sc in zip(solvers, solvers_calctime):
    data[f'{s}.ratio_time'] = compute_ratio(
        data["min.calctime"], data[sc], corr=False
    )
    data[f'{s}.ratio_obj'] = compute_ratio(
        data["primalbound"], data[f'{s}.obj']
    )

data.to_csv("/tmp/result.csv")


fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
print(data.keys())
for s, name, style, color in zip(solvers, solver_names, LINESTYLES, MCOLORS):
    subdata = data.query(f"`{s}.ratio_obj` < 1e5")
    collect_bins_plot(subdata[f"{s}.ratio_obj"], name, style, color, min_val=1, max_val=1e5, ax=axs[0])
axs[0].set_xlabel("Objective value ratio")
axs[0].set_xscale('log')

axs[0].axhline(total_entries, linestyle="-", color='k', linewidth=1, alpha=0.7)

axs[0].set_xlim(0.9, 1e3)

if key == "cvx":
    axs[0].set_ylim(0, 120)
elif key == "noncvx":
    axs[0].set_ylim(0, 160)
axs[0].grid(linewidth=1, linestyle=":")

for s, name, style, color in zip(solvers, solver_names, LINESTYLES, MCOLORS):
    subdata = data.query(f"`{s}.ratio_obj` < 1.01")
    collect_bins_plot(subdata[f"{s}.ratio_time"], name, style, color, min_val=1, max_val=1e3, ax=axs[1])
axs[1].set_xlabel("Time performance index $t / t_{\\min}$")
axs[1].set_xscale('log')

axs[1].axhline(total_entries, linestyle="-", color='k', linewidth=1, alpha=0.7)

axs[1].set_xlim(0.9, 1e3)

if key == "cvx":
    axs[0].set_ylim(0, 120)
elif key == "noncvx":
    axs[0].set_ylim(0, )

axs[1].grid(linewidth=1, linestyle=":")

axs[0].set_ylabel("Problems solved")
axs[0].legend(loc="lower right")
# axs[0].legend(loc="upper left")

plt.tight_layout()
plt.savefig(f"figures/{datetime.now().strftime('%m-%d')}_benchmark_merged_{key}.pdf")
plt.show()
