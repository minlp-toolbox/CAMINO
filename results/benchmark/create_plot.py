from sys import argv
from math import sqrt
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np


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
        "axes.labelsize": 10,  # fontsize for x and y labels (was 10)
        "axes.titlesize": 10,
        "lines.linewidth": 2,
        "legend.fontsize": 10,  # was 10
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
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
    if val == "NAN":
        return np.inf
    else:
        return float(val)


latexify(6, 4)
data = pd.read_csv(argv[1])
key = "conv"
print(data)
scale = 120  # int(input("Amount (e.g. 120):"))

solvers = ["bonmin", "shot"]
solver_names = ["Bonmin", "SHOT"]
solvers_calctime = ['bonmin.solvertime', 'shot.calctime']

if "trmi.solvertime" in data:
    solvers.append("trmi")
    solver_names.append("S-B-MIQP")
    solvers_calctime.append('trmi.solvertime')

if "trmi+.solvertime" in data:
    solvers.append("trmi+")
    solver_names.append("S-B-MIQP with $\mathrm{LB} = V_k$")
    solvers_calctime.append('trmi+.solvertime')


# data.query("convex == True")
values = np.array(data[solvers_calctime]).tolist()
data['min.calctime'] = [np.min(
    [to_float(a) for a in args]
) for args in values]

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

fig, axs = plt.subplots(1, 2, figsize=(6, 3))
print(data.keys())
for s, name, style, color in zip(solvers, solver_names, styles, colors):
    subdata = data.query(f"`{s}.ratio_obj` < 1e5")
    collect_bins_plot(subdata[f"{s}.ratio_obj"], name, style, color, min_val=1, max_val=1e5, ax=axs[0])
axs[0].set_xlabel("Objective value ratio")
axs[0].set_xscale('log')
axs[0].set_xlim(0.9, 1e3)
# axs[0].set_ylim(0, 160)
axs[0].set_ylim(0, 120)

for s, name, style, color in zip(solvers, solver_names, styles, colors):
    subdata = data.query(f"`{s}.ratio_obj` < 1.01")
    collect_bins_plot(subdata[f"{s}.ratio_time"], name, style, color, min_val=1, max_val=1e3, ax=axs[1])
axs[1].set_ylabel("Problems solved")
axs[1].set_xlabel("Time performance index $t / t_{\\min}$")
axs[1].set_xscale('log')
axs[1].set_xlim(0.9, 1e3)
# axs[1].set_ylim(0, 160)
axs[1].set_ylim(0, 120)

axs[0].legend(loc="lower right")
# axs[0].legend(loc="upper left")

plt.tight_layout()
# plt.savefig(f"benchmark_comb_conv.pdf")
plt.show()
