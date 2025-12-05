import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# plot settings
plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 13})
plt.rc(
    "text.latex", preamble=r"\usepackage{amsfonts,amssymb,amsthm,amsmath,times}"
)
plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 13})
plt.rc("axes", **{"labelsize": 14, "titlesize": 15})
colors = {"prop_W": "C0", "prop_Z": "C1", "prop_Q": "C2"}

# plot imputed props
props = pd.read_pickle(
    "experiments/NHANES/results/NHANES_imputed_props_05-05-25.pkl"
)
sns.histplot(
    pd.melt(props),
    x="value",
    hue="variable",
    common_bins=False,
    binwidth=0.0125,
    palette=colors,
    element="step",
    alpha=0.3,
    linewidth=1.4,
)
plt.xlabel(r"Propensity score estimate")
plt.legend(
    labels=[
        r"$\widehat{P}(T_i=1 \mid \mathbf{W}_i)$",
        r"$\widehat{P}(T_i=1 \mid \mathbf{W}_i^\top\hat{\theta}\,)$",
        r"$\widehat{P}(T_i=1 \mid \widehat{g}(1,\mathbf{W}_i))$",
    ],
)
plt.savefig("plots/propensities_imputed.pdf", bbox_inches="tight")
plt.show()

# plot removed props
props = pd.read_pickle(
    "experiments/NHANES/results/NHANES_removed_props_05-05-25.pkl"
)
sns.histplot(
    pd.melt(props),
    x="value",
    hue="variable",
    common_bins=False,
    binwidth=0.0125,
    palette=colors,
    element="step",
    alpha=0.3,
    linewidth=1.4,
)
plt.xlabel(r"Propensity score estimate")
plt.legend(
    labels=[
        r"$\widehat{P}(T_i=1 \mid \mathbf{W}_i)$",
        r"$\widehat{P}(T_i=1 \mid \mathbf{W}_i^\top\hat{\theta}\,)$",
        r"$\widehat{P}(T_i=1 \mid \widehat{g}(1,\mathbf{W}_i))$",
    ]
)
plt.savefig("plots/propensities_removed.pdf", bbox_inches="tight")
plt.show()

# table imputed
df = pd.read_pickle(
    "experiments/NHANES/results/NHANES_imputed_results_05-05-25.pkl"
)

df["BS CI"] = df.apply(
    lambda x: (r"$({:.3f}, {:.3f})$".format(x["BS lower"], x["BS upper"])),
    axis=1,
)

df.reset_index().drop(columns=["BS var", "BS lower", "BS upper"]).sort_values(
    by="BS se"
).round(3).to_latex(
    "plots/nhanes_table_imputed.tex",
    header=["Estimator", "Estimate", "BS se", "BS CI"],
    index=False,
    column_format="lrrr",
    formatters={
        "estimate": lambda x: r"${:.3f}$".format(x),
        "BS se": lambda x: r"${:.3f}$".format(x),
    },
)

# table removed
df = pd.read_pickle(
    "experiments/NHANES/results/NHANES_removed_results_05-05-25.pkl"
)
df["BS CI"] = df.apply(
    lambda x: (r"$({:.3f}, {:.3f})$".format(x["BS lower"], x["BS upper"])),
    axis=1,
)
df.reset_index().drop(columns=["BS var", "BS lower", "BS upper"]).sort_values(
    by="BS se"
).round(3).to_latex(
    "plots/nhanes_table_removed.tex",
    header=["Estimator", "Estimate", "BS se", "BS CI"],
    index=False,
    float_format="%.3f",
    column_format="lrrr",
    formatters={
        "estimate": lambda x: r"${:.3f}$".format(x),
        "BS se": lambda x: r"${:.3f}$".format(x),
    },
)
