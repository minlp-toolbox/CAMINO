from sys import argv
from math import sqrt
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import os


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
        "axes.labelsize": 9,  # fontsize for x and y labels (was 9)
        "axes.titlesize": 9,
        "lines.linewidth": 2,
        "legend.fontsize": 9,  # was 9
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
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
        if ("Objective" in val) or ("feasible" in val) or ("Error" in val) or ("Calling" in val) or ("g_val" in val) or ("CRASH" in val) or ("FAILED" in val) or ("empty" in val) or ("basic_string" in val) or ("No objective" in val) or ("Suffix values" in val):
            return np.inf
        elif val == "NAN":
            return np.inf
        else:
            return float(val)
    else:
        return float(val)

def create_performance_profile(df, solver_columns, problem_column=None, tau_max=10, num_points=100,
                              log_scale=True, name="perf_plot",
                              xlabel="Performance ratio", ylabel="Fraction of problems solved"):
    """
    Create a performance profile plot comparing multiple solvers.
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

    # Calculate performance ratios for each problem
    for i, problem in enumerate(problems):
        if problem_column is None:
            problem_data = data.loc[problem, solver_columns]
            if isinstance(problem_data, pd.Series):
                problem_data = problem_data.to_frame().T
        else:
            problem_data = data[data[problem_column] == problem][solver_columns]

        # Skip problems with all NaN or non-positive values
        if problem_data.isna().all().all() or (problem_data <= 0).all().all():
            continue

        # Get best performance among all solvers for this problem
        best_perf = problem_data.min().min()

        # If best performance is 0 or NaN, skip this problem
        if best_perf <= 0 or np.isnan(best_perf) or np.isinf(best_perf):
            continue

        # Calculate performance ratios
        for j, solver in enumerate(solver_columns):
            value = problem_data[solver].iloc[0]
            # Skip if solver failed (NaN or non-positive value)
            if np.isnan(value) or value <= 0 or np.isinf(value):
                perf_ratios[i, j] = np.inf
            else:
                perf_ratios[i, j] = value / best_perf

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
        ax.plot(tau_values, profile, marker='', linewidth=2, label=solver)

    # Configure the plot
    if log_scale:
        ax.set_xscale('log')
    ax.set_xlim(1, tau_max)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"figures/{name}.pdf")

    return fig, ax


latexify(6, 4)
data = pd.read_csv(argv[1])
key = argv[2]
assert (key == "cvx" or key == "noncvx")
total_entries = data.shape[0]  # int(input("Amount (e.g. 120):"))

# solvers = ["bonmin", f"shot_{key}prob", f"{key}_sbmiqp"]
# solver_names = ["Bonmin", "SHOT", "sbmiqp"]
# solvers_calctime = ['bonmin.calc_time', f'shot_{key}prob.calctime', f"{key}_sbmiqp.calc_time"]

solvers = [f"shot_{key}prob", f"{key}_sbmiqp"]
solver_names = ["SHOT", "sbmiqp"]
solvers_calctime = [f'shot_{key}prob.calctime', f"{key}_sbmiqp.calc_time"]
solvers_obj = [f'shot_{key}prob.obj', f"{key}_sbmiqp.obj"]


data[solvers_calctime] = data[solvers_calctime].map(to_float)
data[solvers_obj] = data[solvers_obj].map(to_float)
data['min.calctime'] = np.min(data[solvers_calctime], axis=1)
data.set_index("name", inplace=True)


# New plots:
create_performance_profile(data, solvers_calctime, tau_max=1e5, name=f'{key}_calc_time_profile')
create_performance_profile(data, solvers_obj, tau_max=1e5, name=f'{key}_obj_profile')

# Plot in the arXiv paper:
for s, sc in zip(solvers, solvers_calctime):
    data[f'{s}.ratio_time'] = compute_ratio(
        data["min.calctime"], data[sc], corr=False
    )
    data[f'{s}.ratio_obj'] = compute_ratio(
        data["primalbound"], data[f'{s}.obj']
    )

data.to_csv("/tmp/result.csv")

# if len(argv) > 2:
#     s, d = solvers, data
#     print("Aah, debugging mode! Use prints like:")
#     print("p d.query(\"'trmi.ratio_obj` < 1.01\")")
#     print("s: " + ", ".join(solvers))
#     breakpoint()
styles = [
    ":",
    "--",
    "-.",
    "-",
]
colors = ["blue", "green", "orange", "red"]

fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
print(data.keys())
for s, name, style, color in zip(solvers, solver_names, styles, colors):
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

for s, name, style, color in zip(solvers, solver_names, styles, colors):
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
plt.savefig(f"figures/benchmark_merged_{key}_1.pdf")
plt.show()
